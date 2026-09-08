"""Stage 3: Event Evidence Aggregation - Merge multi-report evidence."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from src.agent.state import PipelineState
from src.core.config import get_config
from src.core.embedder import compute_similarity_matrix, embed_texts
from src.core.models import EntityNode, EvidenceItem, ExtractionMode, OntologyClass
from src.utils.text import merge_entities, normalize_entity_name

logger = logging.getLogger(__name__)


def _ordered_unique_doc_ids(documents: list) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for doc in documents:
        if doc.doc_id in seen:
            continue
        seen.add(doc.doc_id)
        result.append(doc.doc_id)
    return result


def _find_near_duplicate_reports(
    documents: list,
    threshold: float = 0.85,
    max_representative: int = 8,
) -> list[str]:
    """Find near-duplicate reports using BGE-M3 embeddings.

    Returns list of doc_ids to keep (representative reports).
    """
    unique_doc_ids = _ordered_unique_doc_ids(documents)
    if len(unique_doc_ids) <= max_representative:
        return unique_doc_ids

    # Extract titles and summaries
    texts = []
    doc_ids = []
    for doc in documents:
        parts = []
        if doc.title:
            parts.append(doc.title)
        if doc.summary:
            parts.append(doc.summary)
        if parts:
            texts.append(" ".join(parts))
            doc_ids.append(doc.doc_id)

    if len(texts) < 2:
        return doc_ids[:max_representative]

    # Compute embeddings
    to_remove: set[str] = set()
    try:
        embeddings = np.asarray(embed_texts(texts))
        if embeddings.size == 0:
            return doc_ids[:max_representative]

        # Compute similarity matrix
        sim_matrix = compute_similarity_matrix(embeddings)

        # Find near-duplicates
        seen = set()

        for i in range(len(doc_ids)):
            if doc_ids[i] in seen or doc_ids[i] in to_remove:
                continue

            # Find similar reports (excluding self)
            similarities = sim_matrix[i]
            similar_indices = [
                j for j in range(len(doc_ids))
                if j != i and similarities[j] >= threshold
            ]

            # Keep the most representative (earliest) and remove others
            if similar_indices:
                # Keep the first one (earliest)
                kept = doc_ids[i]
                seen.add(kept)
                to_remove.update(doc_ids[j] for j in similar_indices if doc_ids[j] != kept)

        logger.info(f"[Stage 3] Vectorization dedup: removed {len(to_remove)} near-duplicates")
    except Exception as e:
        logger.warning(f"[Stage 3] Vectorization failed: {e}, using simple dedup")

    # Return representative set
    representatives = []
    for doc_id in doc_ids:
        if doc_id in to_remove or doc_id in representatives:
            continue
        representatives.append(doc_id)
    if not representatives:
        representatives = unique_doc_ids
    return representatives[:max_representative]


def _detect_conflicts(
    entities: list[EntityNode],
    documents: list,
    threshold: float = 0.7,
) -> list[dict]:
    """Detect conflicting entity claims across reports.

    Conflicts include:
    1. Same entity name, different types (e.g., "Google" as AISystem vs Stakeholder)
    2. Contradictory evidence statements
    3. Very low support count (< 2) for high-profile entities

    Returns list of conflict dicts with details.
    """
    if not entities:
        return []

    conflicts = []

    # Group by normalized name
    name_to_entities: dict[str, list[EntityNode]] = {}
    for e in entities:
        norm_name = normalize_entity_name(e.name).lower()
        name_to_entities.setdefault(norm_name, []).append(e)

    # Check for type conflicts
    for norm_name, entity_list in name_to_entities.items():
        if len(entity_list) < 2:
            continue

        # Check if same name has different types
        types = {e.entity_type for e in entity_list}
        if len(types) > 1:
            conflicts.append({
                "type": "type_conflict",
                "entity_name": norm_name,
                "types": [t.value for t in types],
                "entities": [e.name for e in entity_list],
                "reason": f"Same name refers to different ontology classes: {types}",
            })

    # Check for low support count conflicts
    low_support = [e for e in entities if e.support_count < 2]
    if low_support and len(documents) >= 3 and len(entities) > 1:
        conflicts.append({
            "type": "low_support",
            "entities": [e.name for e in low_support],
            "reason": f"{len(low_support)} entities have support_count < 2",
        })

    # Check for entity type conflicts: same entity name assigned different types
    name_to_types: dict[str, set[str]] = {}
    for e in entities:
        name_key = e.name.lower().strip()
        name_to_types.setdefault(name_key, set()).add(e.entity_type.value)
    for name, types in name_to_types.items():
        if len(types) > 1:
            conflicts.append({
                "type": "type_conflict",
                "entity": name[:100],
                "types": list(types),
                "reason": f"Entity '{name[:50]}' assigned conflicting types: {types}",
            })

    logger.info(f"[Stage 3] Detected {len(conflicts)} conflicts")
    return conflicts


def _merge_entities(all_entities: list[EntityNode]) -> list[EntityNode]:
    return merge_entities(all_entities)


def _aggregate_small(entities: list[EntityNode], documents: list) -> dict[str, Any]:
    merged = _merge_entities(entities)
    core = _categorize_entities(merged)
    evidence_sents = []
    for e in merged:
        for ev in e.evidence:
            evidence_sents.append({"doc_id": ev.source_doc_id, "sentence": ev.evidence_sentence})

    support_stats = {}
    for e in merged:
        support_stats[e.name] = e.support_count

    conflicts = _detect_conflicts(merged, documents)

    return {
        "representative_reports": _ordered_unique_doc_ids(documents),
        "core_entities": core,
        "key_evidence_sentences": evidence_sents[:50],
        "support_statistics": support_stats,
        "conflicts": conflicts,
    }


def _aggregate_medium(entities: list[EntityNode], documents: list) -> dict[str, Any]:
    merged = _merge_entities(entities)
    core = _categorize_entities(merged)

    # Use vectorization to select representative reports
    cfg = get_config()
    max_rep = cfg.get("aggregation.max_representative", 8)
    vector_threshold = cfg.get("aggregation.vector_threshold", 0.85)

    representatives = _find_near_duplicate_reports(documents, threshold=vector_threshold, max_representative=max_rep)

    # If vectorization didn't reduce enough, add entity-rich reports
    if len(representatives) < max_rep:
        entity_names = {e.name for e in merged}
        for doc in documents:
            if doc.doc_id not in set(representatives):
                content_lower = doc.content.lower()
                if any(n.lower() in content_lower for n in entity_names):
                    representatives.append(doc.doc_id)
                    if len(representatives) >= max_rep:
                        break

    dedup_representatives: list[str] = []
    for doc_id in representatives:
        if doc_id not in dedup_representatives:
            dedup_representatives.append(doc_id)

    evidence_sents = []
    for e in merged:
        for ev in e.evidence:
            evidence_sents.append({"doc_id": ev.source_doc_id, "sentence": ev.evidence_sentence})

    support_stats = {e.name: e.support_count for e in merged}

    evidence_by_sentence: dict[str, dict] = {}
    for s in evidence_sents:
        sent = s["sentence"]
        if sent not in evidence_by_sentence:
            evidence_by_sentence[sent] = s
    evidence_counter = Counter(s["sentence"] for s in evidence_sents)
    evidence_sents_dedup = [
        evidence_by_sentence[sent]
        for sent, _ in evidence_counter.most_common(30)
        if sent in evidence_by_sentence
    ]

    conflicts = _detect_conflicts(merged, documents)

    return {
        "representative_reports": dedup_representatives[:max_rep],
        "core_entities": core,
        "key_evidence_sentences": evidence_sents_dedup,
        "support_statistics": support_stats,
        "conflicts": conflicts,
    }


def _aggregate_large(entities: list[EntityNode], documents: list) -> dict[str, Any]:
    merged = _merge_entities(entities)
    core = _categorize_entities(merged)

    chunk_size = 12
    chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]

    # Use vectorization to select representative reports from each chunk
    local_packages = []
    for chunk in chunks:
        # Vectorization dedup within chunk
        representatives = _find_near_duplicate_reports(
            chunk,
            threshold=0.85,
            max_representative=3,  # Keep 3 per chunk
        )

        chunk_entities = []
        for e in merged:
            if any(did in e.source_doc_ids for d in chunk for did in [d.doc_id]):
                chunk_entities.append(e)

        local_packages.append({
            "doc_ids": representatives,
            "entities": chunk_entities,
        })

    global_support: dict[str, int] = {}

    for pkg in local_packages:
        for e in pkg["entities"]:
            # Accumulate support_count across chunks instead of taking max
            global_support[e.name] = global_support.get(e.name, 0) + e.support_count

    evidence_sents = []
    for e in merged:
        for ev in e.evidence:
            evidence_sents.append({"doc_id": ev.source_doc_id, "sentence": ev.evidence_sentence})

    evidence_counter = Counter(s["sentence"] for s in evidence_sents)
    evidence_sents_dedup = [
        {"doc_id": "", "sentence": sent}
        for sent, _ in evidence_counter.most_common(40)
    ]
    for s in evidence_sents_dedup:
        for es in evidence_sents:
            if es["sentence"] == s["sentence"]:
                s["doc_id"] = es["doc_id"]
                break

    # Detect conflicts
    conflicts = _detect_conflicts(merged, documents)

    # Build global representative set
    representatives = []
    for pkg in local_packages:
        if pkg["doc_ids"]:
            representatives.extend(pkg["doc_ids"])
            if len(representatives) >= 15:
                break

    dedup_representatives: list[str] = []
    for doc_id in representatives:
        if doc_id not in dedup_representatives:
            dedup_representatives.append(doc_id)

    return {
        "representative_reports": dedup_representatives[:15],
        "core_entities": core,
        "key_evidence_sentences": evidence_sents_dedup,
        "support_statistics": global_support,
        "conflicts": conflicts,
    }


def _categorize_entities(entities: list[EntityNode]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {
        "ai_systems": [],
        "ai_models": [],
        "ai_techniques": [],
        "ai_capabilities": [],
        "stakeholders": [],
        "regulations": [],
        "standards": [],
    }
    type_map = {
        OntologyClass.AI_SYSTEM: "ai_systems",
        OntologyClass.AI_MODEL: "ai_models",
        OntologyClass.AI_TECHNIQUE: "ai_techniques",
        OntologyClass.AI_CAPABILITY: "ai_capabilities",
        OntologyClass.STAKEHOLDER: "stakeholders",
        OntologyClass.REGULATION: "regulations",
        OntologyClass.STANDARD: "standards",
    }
    for e in entities:
        cat = type_map.get(e.entity_type)
        if cat:
            categories[cat].append(e.name)
    return categories


def aggregation_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    report_count = state.get("report_count", 0)
    doc_level_entities = state.get("doc_level_entities", {})
    documents = state.get("documents", [])

    all_entities: list[EntityNode] = []
    for entities in doc_level_entities.values():
        all_entities.extend(entities)

    logger.info(f"[Stage 3] Aggregating {len(all_entities)} entities from {report_count} reports")

    if report_count <= 5:
        package = _aggregate_small(all_entities, documents)
        strategy = "small"
    elif report_count <= 30:
        package = _aggregate_medium(all_entities, documents)
        strategy = "medium"
    else:
        package = _aggregate_large(all_entities, documents)
        strategy = "large"

    total_core = sum(len(v) for v in package["core_entities"].values())
    logger.info(
        f"[Stage 3] Strategy={strategy}, core_entities={total_core}, "
        f"representative_reports={len(package['representative_reports'])}"
    )

    return {
        "event_evidence_package": package,
        "representative_reports": package["representative_reports"],
        "core_entities": package["core_entities"],
        "support_statistics": package["support_statistics"],
        "conflicts": package["conflicts"],
        "current_stage": "aggregation",
        "stages_completed": state.get("stages_completed", []) + ["aggregation"],
    }
