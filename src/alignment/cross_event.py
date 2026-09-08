"""Cross-event entity alignment via global entity registry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.alignment.bm25_index import BM25Indexer
from src.core.config import get_config
from src.core.embedder import compute_similarity, embed_texts
from src.core.models import EntityNode, OntologyClass
from src.utils.text import normalize_entity_name, normalize_for_matching

logger = logging.getLogger(__name__)

# Stakeholder subtypes that should be merged across roles.
# "Google" as AIProvider and "Google" as AIDeployer are the same real-world entity.
# NOTE: AffectedActor is intentionally excluded — it is an event-specific role
# (e.g. "the wrongly arrested individual") and is listed in _NON_ALIGNABLE_TYPES.
_STAKEHOLDER_TYPES = frozenset({
    OntologyClass.STAKEHOLDER,
    OntologyClass.AI_DEVELOPER,
    OntologyClass.AI_PROVIDER,
    OntologyClass.AI_DEPLOYER,
    OntologyClass.AI_USER,
    OntologyClass.REGULATOR,
})

# Types that are NOT real-world entities and should NOT create cross-event connections.
# These are inferred abstractions (lifecycle phases, domains, risk concepts).
_NON_ALIGNABLE_TYPES = frozenset({
    OntologyClass.AI_RISK_INCIDENT,
    OntologyClass.RISK_SOURCE,
    OntologyClass.RISK,
    OntologyClass.HAZARD,
    OntologyClass.THREAT,
    OntologyClass.VULNERABILITY,
    OntologyClass.CONSEQUENCE,
    OntologyClass.IMPACT,
    OntologyClass.RISK_CONTROL,
    OntologyClass.AFFECTED_ACTOR,  # event-specific role, not a real-world entity
    OntologyClass.ROLE_ASSIGNMENT,
    OntologyClass.PURPOSE,
    OntologyClass.AI_LIFECYCLE_PHASE,
    OntologyClass.DOMAIN,
    OntologyClass.NEWS_REPORT,
    OntologyClass.INFORMATION_SOURCE,
    OntologyClass.EVIDENCE,
    OntologyClass.KNOWLEDGE_STATEMENT,
    OntologyClass.EXTRACTION_MODE,
    OntologyClass.TREND,
    OntologyClass.OBLIGATION,
    OntologyClass.COMPLIANCE_REQUIREMENT,
    OntologyClass.ENFORCEMENT_ACTION,
    OntologyClass.CODE_OF_CONDUCT,
    OntologyClass.DOCUMENTATION,
    OntologyClass.TECHNICAL_DOCUMENTATION,
    OntologyClass.VERIFICATION_TEST,
})


def _alignment_type(entity: EntityNode) -> str:
    """Return the type key for alignment purposes.

    Stakeholder subtypes normalize to "Stakeholder" so that
    "Google" as AIProvider and "Google" as AIDeployer match.
    """
    if entity.entity_type in _STAKEHOLDER_TYPES:
        return "Stakeholder"
    return entity.entity_type.value


class EntityRegistry:
    """Global registry that maps canonical entity names to stable IDs across events.

    Uses hybrid alignment (semantic + BM25) for cross-event entity matching.
    Only aligns real-world entities (AISystem, AIModel, Stakeholder, etc.).
    Inferred abstractions (AILifecyclePhase, Domain, Risk, etc.) are NOT aligned.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_hybrid: bool = True,
    ) -> None:
        self.threshold = threshold
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.use_hybrid = use_hybrid

        # key: (alignment_type, canonical_name_lower) -> EntityNode
        self._registry: dict[tuple[str, str], EntityNode] = {}
        self._vectors: dict[tuple[str, str], list[float]] = {}

    @classmethod
    def from_config(cls) -> "EntityRegistry":
        config = get_config()
        return cls(
            threshold=config.get("alignment.cross_event_threshold", 0.85),
            semantic_weight=config.get("alignment.semantic_weight", 0.6),
            bm25_weight=config.get("alignment.bm25_weight", 0.4),
            use_hybrid=config.get("alignment.use_hybrid", True),
        )

    def _key(self, entity: EntityNode) -> tuple[str, str]:
        return (_alignment_type(entity), normalize_entity_name(entity.name).lower())

    def _normalize_name(self, name: str) -> str:
        return normalize_for_matching(name)

    def register(self, entity: EntityNode) -> EntityNode:
        """Register an entity. If a matching entity exists, return the existing one
        (merged). Otherwise, register and return the entity with a stable ID.

        Non-alignable types (Risk, Domain, LifecyclePhase, etc.) are always
        registered as unique — they never merge across events.
        """
        if entity.entity_type in _NON_ALIGNABLE_TYPES:
            return entity

        key = self._key(entity)

        # Exact match
        if key in self._registry:
            existing = self._registry[key]
            existing.evidence.extend(entity.evidence)
            existing.source_doc_ids = list(set(existing.source_doc_ids + entity.source_doc_ids))
            existing.support_count += entity.support_count
            existing.confidence = round(max(existing.confidence, entity.confidence), 2)
            return existing

        # Semantic match — check against all same-type entities
        existing_match = self._find_semantic_match(entity)
        if existing_match is not None:
            existing_match.evidence.extend(entity.evidence)
            existing_match.source_doc_ids = list(set(existing_match.source_doc_ids + entity.source_doc_ids))
            existing_match.support_count += entity.support_count
            existing_match.confidence = round(max(existing_match.confidence, entity.confidence), 2)
            self._registry[key] = existing_match
            return existing_match

        # New entity — register it
        self._registry[key] = entity
        return entity

    def _find_semantic_match(self, entity: EntityNode) -> EntityNode | None:
        """Find a semantically similar entity in the registry (same alignment type)."""
        atype = _alignment_type(entity)
        candidates = [
            (k, e) for k, e in self._registry.items()
            if k[0] == atype
        ]
        if not candidates:
            return None

        names = [entity.name] + [e.name for _, e in candidates]

        vectors = embed_texts(names)
        query_vec = vectors[0]

        semantic_scores = [0.0]
        for i in range(1, len(vectors)):
            sim = compute_similarity(query_vec, vectors[i])
            semantic_scores.append(sim)

        if self.use_hybrid:
            bm25 = BM25Indexer()
            bm25.build_index(names)
            bm25_scores = bm25.get_scores(entity.name)

            total_weight = self.semantic_weight + self.bm25_weight
            sw = self.semantic_weight / total_weight if total_weight > 0 else 0.6
            bw = self.bm25_weight / total_weight if total_weight > 0 else 0.4
        else:
            bm25_scores = np.zeros(len(names))
            sw, bw = 1.0, 0.0

        best_score = 0.0
        best_entity = None
        for i, (k, e) in enumerate(candidates, start=1):
            hybrid_score = sw * semantic_scores[i] + bw * bm25_scores[i]

            # Apply exact-match bonus only when semantic similarity >= 0.5
            if (normalize_for_matching(entity.name) == normalize_for_matching(e.name)
                    and semantic_scores[i] >= 0.5):
                hybrid_score = max(hybrid_score, 0.85)

            if hybrid_score > best_score:
                best_score = hybrid_score
                best_entity = e

        if best_score >= self.threshold and best_entity is not None:
            logger.info(
                f"[Registry] Hybrid match: '{entity.name}' ≈ '{best_entity.name}' (score={best_score:.3f})"
            )
            return best_entity

        return None

    def register_batch(self, entities: list[EntityNode]) -> list[EntityNode]:
        result: list[EntityNode] = []
        for entity in entities:
            resolved = self.register(entity)
            result.append(resolved)
        return result

    def save(self, path: Path) -> None:
        data = {}
        for (etype, name), entity in self._registry.items():
            data[f"{etype}|{name}"] = entity.model_dump(mode="json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key_str, entity_data in data.items():
            etype, name = key_str.split("|", 1)
            entity = EntityNode(**entity_data)
            self._registry[(etype, name)] = entity

    @property
    def size(self) -> int:
        return len(self._registry)
