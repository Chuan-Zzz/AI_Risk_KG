"""Stage 4 Layer 1: Entity Reuse - Three-layer entity alignment.

Layer 1: Name normalization (alias substring matching)
Layer 2: Semantic similarity (bge-m3 embedding)
Layer 3: LLM verification (boundary cases only)
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import PipelineState
from src.alignment.llm_verifier import LLMVerifier
from src.alignment.semantic_aligner import SemanticAligner
from src.core.config import get_config
from src.core.models import (
    EntityNode,
    ExtractionMode,
    OntologyClass,
)
from src.utils.text import generate_id, merge_entities, normalize_entity_name

logger = logging.getLogger(__name__)

_NAME_TYPE_NORMALIZATION = {
    ("llm", OntologyClass.AI_SYSTEM): OntologyClass.AI_MODEL,
    ("large language model", OntologyClass.AI_SYSTEM): OntologyClass.AI_MODEL,
    ("foundation model", OntologyClass.AI_SYSTEM): OntologyClass.AI_MODEL,
    ("chatbot", OntologyClass.AI_SYSTEM): OntologyClass.AI_SYSTEM,
    ("algorithm", OntologyClass.AI_SYSTEM): OntologyClass.AI_TECHNIQUE,
}

# Stakeholder subtypes — normalize to Stakeholder for alignment key
_STAKEHOLDER_TYPES = frozenset({
    OntologyClass.STAKEHOLDER,
    OntologyClass.AI_DEVELOPER,
    OntologyClass.AI_PROVIDER,
    OntologyClass.AI_DEPLOYER,
    OntologyClass.AI_USER,
    OntologyClass.REGULATOR,
    OntologyClass.AFFECTED_ACTOR,
})


def _alignment_type(entity: EntityNode) -> str:
    """Type key for alignment — stakeholder subtypes normalize to Stakeholder."""
    if entity.entity_type in _STAKEHOLDER_TYPES:
        return "Stakeholder"
    return entity.entity_type.value


def _load_aliases_from_reused(entities: list[EntityNode]) -> dict[str, str]:
    """Build alias map: shorter names that are substrings of longer names → canonical.

    Groups by alignment type (stakeholder subtypes normalize to Stakeholder)
    so that "Google" as AIProvider and "Google" as AIDeployer can match.
    """
    aliases: dict[str, str] = {}
    by_type: dict[str, list[EntityNode]] = {}
    for e in entities:
        atype = _alignment_type(e)
        by_type.setdefault(atype, []).append(e)

    for atype, group in by_type.items():
        sorted_group = sorted(group, key=lambda e: len(e.name), reverse=True)
        for canonical in sorted_group:
            canon_lower = normalize_entity_name(canonical.name).lower()
            for candidate in sorted_group:
                cand_lower = normalize_entity_name(candidate.name).lower()
                if cand_lower == canon_lower:
                    continue
                if cand_lower in canon_lower and len(cand_lower) >= 2:
                    if cand_lower not in aliases:
                        aliases[cand_lower] = normalize_entity_name(canonical.name).lower()

    return aliases


def _normalize_type(entity: EntityNode) -> OntologyClass:
    name_key = normalize_entity_name(entity.name).lower()
    type_key = (name_key, entity.entity_type)
    return _NAME_TYPE_NORMALIZATION.get(type_key, entity.entity_type)


def _alias_key_fn(alias_map: dict[str, str]):
    """Return a key function that resolves aliases."""
    def key_fn(entity: EntityNode) -> str:
        name_lower = normalize_entity_name(entity.name).lower()
        return alias_map.get(name_lower, name_lower)
    return key_fn


def _merge_group(entities: list[EntityNode], indices: list[int]) -> None:
    """Merge entities at `indices` into the first one in-place."""
    if len(indices) < 2:
        return
    canonical = entities[indices[0]]
    for idx in indices[1:]:
        other = entities[idx]
        canonical.evidence.extend(other.evidence)
        canonical.source_doc_ids = list(set(canonical.source_doc_ids + other.source_doc_ids))
        canonical.support_count = len(canonical.source_doc_ids)
        canonical.confidence = round(max(canonical.confidence, other.confidence), 2)
        if len(other.name) > len(canonical.name):
            canonical.name = other.name
        other.id = ""  # mark for removal


def entity_reuse_node(state: PipelineState) -> dict[str, Any]:
    event_id = state.get("event_id", "unknown")
    doc_level_entities = state.get("doc_level_entities", {})

    all_entities: list[EntityNode] = []
    for entities in doc_level_entities.values():
        all_entities.extend(entities)

    if not all_entities:
        return {
            "reused_entities": [],
            "current_stage": "entity_reuse",
            "stages_completed": state.get("stages_completed", []) + ["entity_reuse"],
        }

    cfg = get_config()
    sim_threshold = cfg.get("alignment.similarity_threshold", 0.8)

    # === Layer 1: Name normalization (alias substring) ===
    alias_map = _load_aliases_from_reused(all_entities)
    if alias_map:
        logger.info(f"[Stage 4 L1] Alias map: {len(alias_map)} entries")

    merged = merge_entities(
        all_entities,
        type_fn=_normalize_type,
        key_fn=_alias_key_fn(alias_map),
    )
    logger.info(f"[Stage 4 L1] After name normalization: {len(merged)} entities")

    # === Layer 2: Semantic similarity alignment ===
    try:
        aligner = SemanticAligner(threshold=sim_threshold)
        groups = aligner.align(merged)

        if groups:
            for group_indices in groups:
                _merge_group(merged, group_indices)
            merged = [e for e in merged if e.id != ""]
            logger.info(f"[Stage 4 L1] Semantic alignment merged {len(groups)} groups")

        # === Layer 3: LLM verification for boundary pairs ===
        llm_top_k = cfg.get("alignment.llm_verify_top_k", 3)
        early_stop = cfg.get("alignment.early_stopping_threshold", 0.55)
        boundary_pairs = aligner.get_candidate_pairs(
            merged, low=early_stop, high=sim_threshold,
        )

        if boundary_pairs and llm_top_k > 0:
            boundary_pairs = boundary_pairs[:llm_top_k]
            verifier = LLMVerifier()
            results = []
            for i, j, sim in boundary_pairs:
                same = verifier.verify(merged[i], merged[j], sim)
                results.append((i, j, same))
                logger.info(f"[Stage 4 L1] LLM verify: '{merged[i].name}' vs '{merged[j].name}' (sim={sim:.3f}) -> {same}")

            # Merge LLM-verified pairs
            from src.alignment.semantic_aligner import UnionFind
            verified_indices = {idx for i, j, _ in boundary_pairs for idx in (i, j)}
            index_list = sorted(verified_indices)
            local_map = {orig: local for local, orig in enumerate(index_list)}
            uf = UnionFind(len(index_list))
            for i, j, same in results:
                if same:
                    uf.union(local_map[i], local_map[j])

            root_groups: dict[int, list[int]] = {}
            for orig_idx in index_list:
                root = uf.find(local_map[orig_idx])
                root_groups.setdefault(root, []).append(orig_idx)

            for members in root_groups.values():
                if len(members) > 1:
                    _merge_group(merged, members)
            merged = [e for e in merged if e.id != ""]
    except Exception as e:
        logger.warning(f"[Stage 4 L1] Semantic/LLM alignment failed, using name-only: {e}")

    # Regenerate IDs
    for entity in merged:
        entity.id = generate_id(event_id, entity.name, entity.entity_type.value)

    logger.info(f"[Stage 4 L1] Final: {len(merged)} entities for event {event_id}")

    return {
        "reused_entities": merged,
        "current_stage": "entity_reuse",
        "stages_completed": state.get("stages_completed", []) + ["entity_reuse"],
    }
