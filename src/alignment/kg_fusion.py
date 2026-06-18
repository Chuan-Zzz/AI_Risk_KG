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

from src.alignment.cross_event import _NON_ALIGNABLE_TYPES, _STAKEHOLDER_TYPES, _alignment_type
from src.alignment.semantic_aligner import SemanticAligner, UnionFind
from src.alignment.llm_verifier import LLMVerifier
from src.core.config import get_config
from src.core.models import EntityNode, EventKnowledgeSubgraph, OntologyClass
from src.utils.text import normalize_entity_name

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

    # Step 2: Within each type, do name-based blocking + semantic matching
    for atype, typed_entities in by_type.items():
        if len(typed_entities) < 2:
            continue

        indices = [idx for idx, _ in typed_entities]
        nodes = [node for _, node in typed_entities]
        names = [node.name for node in nodes]

        # 2a. Name-based exact blocking (normalized name match)
        name_groups: dict[str, list[int]] = {}
        for local_i, node in enumerate(nodes):
            norm = normalize_entity_name(node.name).lower()
            name_groups.setdefault(norm, []).append(local_i)

        for norm, members in name_groups.items():
            if len(members) > 1:
                for m in members[1:]:
                    uf.union(indices[members[0]], indices[m])

        # 2b. Semantic similarity for remaining unmatched pairs
        try:
            aligner = SemanticAligner(threshold=threshold)
            sim_matrix = aligner._compute_hybrid_scores(names)

            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    if float(sim_matrix[a, b]) >= threshold:
                        uf.union(indices[a], indices[b])
        except Exception as e:
            logger.warning(f"Semantic alignment failed for type {atype}: {e}")

    # Step 3: LLM verification for boundary pairs
    if llm_verify_top_k > 0:
        _llm_verify_boundary(entities, uf, threshold, llm_verify_top_k)

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
        member_entities = [entities[m] for m in members]
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
    merged_entities = sum(len(m) for m in root_to_members.values() if len(m) > 1)
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
) -> None:
    """Use LLM to verify boundary cases (similarity in [0.55, threshold))."""
    low, high = 0.55, threshold

    by_type: dict[str, list[tuple[int, EntityNode]]] = {}
    for idx, (event_id, eid, node) in enumerate(entities):
        atype = _alignment_type(node)
        by_type.setdefault(atype, []).append((idx, node))

    boundary_pairs: list[tuple[int, int, float]] = []

    for atype, typed_entities in by_type.items():
        if len(typed_entities) < 2:
            continue

        indices = [idx for idx, _ in typed_entities]
        nodes = [node for _, node in typed_entities]
        names = [node.name for node in nodes]

        try:
            aligner = SemanticAligner(threshold=high)
            sim_matrix = aligner._compute_hybrid_scores(names)

            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    sim = float(sim_matrix[a, b])
                    if low <= sim < high:
                        # Skip if already in same group
                        if uf.find(indices[a]) != uf.find(indices[b]):
                            boundary_pairs.append((indices[a], indices[b], sim))
        except Exception:
            continue

    if not boundary_pairs:
        return

    # Sort by similarity descending, take top_k
    boundary_pairs.sort(key=lambda x: x[2], reverse=True)
    boundary_pairs = boundary_pairs[:top_k]

    logger.info(f"[Phase 2] LLM verifying {len(boundary_pairs)} boundary pairs")

    verifier = LLMVerifier()
    for idx_a, idx_b, sim in boundary_pairs:
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

        # Update edge subject/object IDs
        for edge in sg.edges:
            edge.subject_id = id_mapping.get(edge.subject_id, edge.subject_id)
            edge.object_id = id_mapping.get(edge.object_id, edge.object_id)

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
) -> None:
    """Write all resolved subgraphs to Neo4j."""
    try:
        from src.storage.neo4j_store import Neo4jStore
        store = Neo4jStore()
        store.clear_database()

        for event_id, sg in subgraphs.items():
            store.add_event_subgraph(sg)

        store.close()
        logger.info(f"[Phase 2] Neo4j: wrote {len(subgraphs)} event subgraphs")
    except Exception as e:
        logger.warning(f"[Phase 2] Neo4j write failed: {e}")
