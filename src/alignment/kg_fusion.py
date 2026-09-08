"""Two-phase Knowledge Graph Fusion.

Phase 1: Per-event extraction (independent, no cross-event alignment)
Phase 2: Global entity resolution + Neo4j assembly

This replaces the incremental alignment approach with a proper two-phase
architecture that eliminates order dependency in entity alignment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.alignment.cross_event import _NON_ALIGNABLE_TYPES, _STAKEHOLDER_TYPES, _alignment_type
from src.alignment.semantic_aligner import SemanticAligner, UnionFind
from src.alignment.llm_verifier import LLMVerifier
from src.core.config import get_config
from src.core.models import EntityNode, EventKnowledgeSubgraph, OntologyClass
from src.utils.text import normalize_for_matching

logger = logging.getLogger(__name__)


def _collect_alignable_entities(
    subgraphs: dict[str, EventKnowledgeSubgraph],
) -> list[tuple[str, str, EntityNode]]:
    """Collect all alignable entities from all event subgraphs.

    Returns list of (event_id, entity_id_in_subgraph, entity_node).
    """
    entities: list[tuple[str, str, EntityNode]] = []
    for event_id, sg in subgraphs.items():
        for node in sg.nodes:
            if node.entity_type not in _NON_ALIGNABLE_TYPES:
                entities.append((event_id, node.id, node))
    return entities


def _global_entity_resolution(
    entities: list[tuple[str, str, EntityNode]],
    threshold: float = 0.8,
    llm_verify_top_k: int = 5,
) -> dict[str, str]:
    """Resolve entities across all events globally.

    Returns a mapping: original_entity_id -> canonical_entity_id
    Entities mapped to the same canonical ID should be merged.
    """
    if not entities:
        return {}

    # Step 1: Group by alignment type
    by_type: dict[str, list[tuple[int, EntityNode]]] = {}
    for idx, (event_id, eid, node) in enumerate(entities):
        atype = _alignment_type(node)
        by_type.setdefault(atype, []).append((idx, node))

    # Global UnionFind over all entities
    n = len(entities)
    uf = UnionFind(n)

    # Cache similarity matrices per type so they are computed only once
    # and reused by both the threshold merge and the LLM boundary verification.
    type_sim_cache: dict[str, tuple[list[int], list[EntityNode], np.ndarray]] = {}

    # Step 2: Within each type, do name-based blocking + semantic matching
    aligner = SemanticAligner(threshold=threshold)
    for atype, typed_entities in by_type.items():
        if len(typed_entities) < 2:
            continue

        indices = [idx for idx, _ in typed_entities]
        nodes = [node for _, node in typed_entities]
        names = [node.name for node in nodes]

        # 2a. Name-based exact blocking (aggressive normalization match)
        name_groups: dict[str, list[int]] = {}
        for local_i, node in enumerate(nodes):
            norm = normalize_for_matching(node.name)
            name_groups.setdefault(norm, []).append(local_i)

        for norm, members in name_groups.items():
            if len(members) > 1:
                for m in members[1:]:
                    uf.union(indices[members[0]], indices[m])

        # 2b. Semantic similarity for remaining unmatched pairs
        try:
            sim_matrix = aligner._compute_hybrid_scores(names)
            type_sim_cache[atype] = (indices, nodes, sim_matrix)

            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    if float(sim_matrix[a, b]) >= threshold:
                        uf.union(indices[a], indices[b])
        except Exception as e:
            logger.warning(f"Semantic alignment failed for type {atype}: {e}")

    # Step 3: LLM verification for boundary pairs (reuses cached sim matrices)
    if llm_verify_top_k > 0:
        _llm_verify_boundary(entities, uf, threshold, llm_verify_top_k, type_sim_cache)

    # Step 4: Build canonical ID mapping
    # For each group, pick the entity with the longest name as canonical
    root_to_members: dict[int, list[int]] = {}
    for i in range(n):
        root = uf.find(i)
        root_to_members.setdefault(root, []).append(i)

    id_mapping: dict[str, str] = {}
    for members in root_to_members.values():
        if len(members) == 1:
            event_id, eid, node = entities[members[0]]
            id_mapping[eid] = eid
            continue

        # Pick canonical: highest support_count, then longest name
        best_idx = max(members, key=lambda m: (
            entities[m][2].support_count,
            len(entities[m][2].name),
        ))
        canonical_eid = entities[best_idx][1]

        for m in members:
            event_id, eid, node = entities[m]
            id_mapping[eid] = canonical_eid

    # Log merge statistics
    merge_count = sum(1 for members in root_to_members.values() if len(members) > 1)
    merged_entities = sum(len(m) for m in root_to_members.values() if len(members) > 1)
    logger.info(
        f"[Phase 2] Global resolution: {n} entities, "
        f"{merge_count} merge groups, {merged_entities} entities merged"
    )

    return id_mapping


def _llm_verify_boundary(
    entities: list[tuple[str, str, EntityNode]],
    uf: UnionFind,
    threshold: float,
    top_k: int,
    type_sim_cache: dict[str, tuple[list[int], list[EntityNode], np.ndarray]],
) -> None:
    """Use LLM to verify boundary cases (similarity in [0.55, threshold)).

    Reuses the similarity matrices cached during step 2b to avoid recomputing
    embeddings and BM25 scores. The ``top_k`` budget is distributed across type
    groups proportionally so that no single type monopolises the LLM calls.
    """
    low, high = 0.55, threshold

    # Collect boundary pairs per type, with global indices
    per_type_pairs: dict[str, list[tuple[int, int, float]]] = {}
    total_boundary = 0

    for atype, (indices, nodes, sim_matrix) in type_sim_cache.items():
        pairs: list[tuple[int, int, float]] = []
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                sim = float(sim_matrix[a, b])
                if low <= sim < high:
                    # Skip if already in same group
                    if uf.find(indices[a]) != uf.find(indices[b]):
                        pairs.append((indices[a], indices[b], sim))
        if pairs:
            # Sort within type by similarity descending
            pairs.sort(key=lambda x: x[2], reverse=True)
            per_type_pairs[atype] = pairs
            total_boundary += len(pairs)

    if total_boundary == 0:
        return

    # Distribute top_k budget across types proportionally to their boundary count.
    # Each type gets at least 1 slot if it has boundary pairs and top_k > 0.
    selected_pairs: list[tuple[int, int, float]] = []
    if total_boundary <= top_k:
        # Fewer boundary pairs than budget — verify all
        for pairs in per_type_pairs.values():
            selected_pairs.extend(pairs)
    else:
        remaining = top_k
        type_items = list(per_type_pairs.items())
        for i, (atype, pairs) in enumerate(type_items):
            if remaining <= 0:
                break
            # Proportional allocation, at least 1
            alloc = max(1, round(top_k * len(pairs) / total_boundary))
            alloc = min(alloc, remaining, len(pairs))
            selected_pairs.extend(pairs[:alloc])
            remaining -= alloc

    # Sort final selection by similarity descending so highest-confidence
    # boundary pairs are verified first.
    selected_pairs.sort(key=lambda x: x[2], reverse=True)

    logger.info(
        f"[Phase 2] LLM verifying {len(selected_pairs)}/{total_boundary} boundary pairs "
        f"(budget={top_k}, types={len(per_type_pairs)})"
    )

    verifier = LLMVerifier()
    for idx_a, idx_b, sim in selected_pairs:
        node_a = entities[idx_a][2]
        node_b = entities[idx_b][2]
        same = verifier.verify(node_a, node_b, sim)
        if same:
            uf.union(idx_a, idx_b)
            logger.info(
                f"[Phase 2] LLM merged: '{node_a.name}' ≈ '{node_b.name}' (sim={sim:.3f})"
            )


def _apply_resolution(
    subgraphs: dict[str, EventKnowledgeSubgraph],
    id_mapping: dict[str, str],
) -> dict[str, EventKnowledgeSubgraph]:
    """Apply entity resolution mapping to all subgraphs.

    For each entity whose ID is mapped to a canonical ID:
    1. Update the entity's ID to the canonical ID
    2. Update all edges that reference the old ID
    3. Remove self-loop edges (subject_id == object_id) created by merging
    4. Deduplicate edges that became identical after ID remapping
    """
    for event_id, sg in subgraphs.items():
        # Build reverse lookup: canonical_id -> list of entities to merge
        node_map: dict[str, EntityNode] = {}
        for node in sg.nodes:
            canonical_id = id_mapping.get(node.id, node.id)
            if canonical_id in node_map:
                # Merge into canonical
                existing = node_map[canonical_id]
                existing.evidence.extend(node.evidence)
                existing.source_doc_ids = list(set(
                    existing.source_doc_ids + node.source_doc_ids
                ))
                existing.support_count += node.support_count
                existing.confidence = round(max(existing.confidence, node.confidence), 2)
                if len(node.name) > len(existing.name):
                    existing.name = node.name
            else:
                node.id = canonical_id
                node_map[canonical_id] = node

        # Update node list (deduplicated)
        sg.nodes = list(node_map.values())

        # Update edge subject/object IDs, then drop self-loops and duplicates
        seen_edges: set[tuple[str, str, str]] = set()
        kept_edges = []
        for edge in sg.edges:
            edge.subject_id = id_mapping.get(edge.subject_id, edge.subject_id)
            edge.object_id = id_mapping.get(edge.object_id, edge.object_id)
            # Skip self-loops created by entity merging
            if edge.subject_id == edge.object_id:
                continue
            # Skip duplicate edges (same subject, relation, object)
            key = (edge.subject_id, edge.predicate, edge.object_id)
            if key in seen_edges:
                # Merge evidence into the already-kept edge
                for kept in kept_edges:
                    if (kept.subject_id, kept.predicate, kept.object_id) == key:
                        # EvidenceItem is not hashable; dedupe by evidence_id
                        seen_ev_ids = {ev.evidence_id for ev in kept.evidence}
                        for ev in edge.evidence:
                            if ev.evidence_id not in seen_ev_ids:
                                kept.evidence.append(ev)
                                seen_ev_ids.add(ev.evidence_id)
                        kept.source_doc_ids = list(set(
                            kept.source_doc_ids + edge.source_doc_ids
                        ))
                        kept.confidence = round(max(kept.confidence, edge.confidence), 2)
                        break
                continue
            seen_edges.add(key)
            kept_edges.append(edge)
        sg.edges = kept_edges

    return subgraphs


def fuse_subgraphs(
    subgraphs: dict[str, EventKnowledgeSubgraph],
    threshold: float = 0.8,
    llm_verify_top_k: int = 5,
) -> dict[str, EventKnowledgeSubgraph]:
    """Two-phase Knowledge Graph Fusion.

    Phase 1 is already done (per-event extraction, subgraphs provided).
    This function does Phase 2: global entity resolution + assembly.

    Args:
        subgraphs: event_id -> EventKnowledgeSubgraph (from Phase 1)
        threshold: Similarity threshold for entity merging
        llm_verify_top_k: Number of boundary pairs to verify with LLM

    Returns:
        Resolved subgraphs with canonical entity IDs
    """
    logger.info(f"[Phase 2] Fusing {len(subgraphs)} event subgraphs")

    # Collect all alignable entities
    entities = _collect_alignable_entities(subgraphs)
    logger.info(f"[Phase 2] Collected {len(entities)} alignable entities")

    # Global entity resolution
    id_mapping = _global_entity_resolution(entities, threshold, llm_verify_top_k)

    # Apply resolution to subgraphs
    resolved = _apply_resolution(subgraphs, id_mapping)

    # Log statistics
    total_nodes = sum(len(sg.nodes) for sg in resolved.values())
    total_edges = sum(len(sg.edges) for sg in resolved.values())
    logger.info(
        f"[Phase 2] Fusion complete: {total_nodes} nodes, {total_edges} edges "
        f"across {len(resolved)} events"
    )

    return resolved


def write_to_neo4j(
    subgraphs: dict[str, EventKnowledgeSubgraph],
    *,
    clear_existing: bool = False,
) -> None:
    """Write resolved subgraphs to Neo4j without deleting unrelated data.

    Set ``clear_existing`` only for an intentionally destructive full rebuild.
    """
    from src.storage.neo4j_store import Neo4jStore
    store = Neo4jStore()
    try:
        if clear_existing:
            store.clear_database()

        for event_id, sg in subgraphs.items():
            store.add_event_subgraph(sg)

        logger.info(f"[Phase 2] Neo4j: wrote {len(subgraphs)} event subgraphs")
    except Exception as e:
        logger.warning(f"[Phase 2] Neo4j write failed: {e}")
    finally:
        store.close()
