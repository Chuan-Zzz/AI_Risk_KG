"""Hybrid entity alignment using semantic embeddings + BM25."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.alignment.bm25_index import BM25Indexer
from src.core.config import get_config
from src.core.embedder import compute_similarity_matrix, embed_texts
from src.core.models import EntityNode, OntologyClass
from src.utils.text import normalize_for_matching

logger = logging.getLogger(__name__)

# Stakeholder subtypes that should be aligned together.
# AffectedActor is excluded — it is event-specific (see _NON_ALIGNABLE_TYPES).
_STAKEHOLDER_TYPES = frozenset({
    OntologyClass.STAKEHOLDER,
    OntologyClass.AI_DEVELOPER,
    OntologyClass.AI_PROVIDER,
    OntologyClass.AI_DEPLOYER,
    OntologyClass.AI_USER,
    OntologyClass.REGULATOR,
})


def _alignment_type(entity: EntityNode) -> str:
    """Type key for alignment — stakeholder subtypes normalize to Stakeholder."""
    if entity.entity_type in _STAKEHOLDER_TYPES:
        return "Stakeholder"
    return entity.entity_type.value


class UnionFind:
    """Disjoint-set / Union-Find with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class SemanticAligner:
    """Hybrid entity aligner using semantic embeddings + BM25.

    Three-layer alignment:
    1. Name normalization (exact match)
    2. Hybrid similarity (semantic + BM25)
    3. LLM verification (boundary pairs)
    """

    def __init__(
        self,
        threshold: float = 0.8,
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        exact_match_bonus: float = 0.3,
        use_hybrid: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """Initialize hybrid aligner.

        Args:
            threshold: Final hybrid score threshold for merging
            semantic_weight: Weight for semantic similarity (0-1)
            bm25_weight: Weight for BM25 score (0-1)
            exact_match_bonus: Bonus added when normalized names match exactly
            use_hybrid: Enable hybrid mode (false = pure semantic)
            bm25_k1: BM25 term frequency saturation parameter
            bm25_b: BM25 document length normalization
        """
        self.threshold = threshold
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.exact_match_bonus = exact_match_bonus
        self.use_hybrid = use_hybrid
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        # Normalize weights
        total_weight = self.semantic_weight + self.bm25_weight
        if total_weight > 0:
            self.semantic_weight /= total_weight
            self.bm25_weight /= total_weight

    @classmethod
    def from_config(cls) -> "SemanticAligner":
        """Create aligner from config file."""
        config = get_config()
        return cls(
            threshold=config.get("alignment.similarity_threshold", 0.8),
            semantic_weight=config.get("alignment.semantic_weight", 0.6),
            bm25_weight=config.get("alignment.bm25_weight", 0.4),
            exact_match_bonus=config.get("alignment.exact_match_bonus", 0.3),
            use_hybrid=config.get("alignment.use_hybrid", True),
            bm25_k1=config.get("alignment.bm25_k1", 1.5),
            bm25_b=config.get("alignment.bm25_b", 0.75),
        )

    def _compute_hybrid_scores(
        self,
        names: list[str],
    ) -> np.ndarray:
        """Compute hybrid similarity matrix (semantic + BM25).

        Args:
            names: List of entity names

        Returns:
            Symmetric similarity matrix [0, 1]
        """
        n = len(names)

        # 1. Semantic similarity
        embeddings = embed_texts(names)
        semantic_sim = compute_similarity_matrix(embeddings)

        if not self.use_hybrid:
            # Still apply exact-match bonus on pure semantic scores
            for i in range(n):
                for j in range(i + 1, n):
                    norm_i = normalize_for_matching(names[i])
                    norm_j = normalize_for_matching(names[j])
                    if norm_i and norm_i == norm_j:
                        # Only apply bonus when semantic similarity is
                        # moderately high — very low semantic similarity
                        # with a normalized name match usually means the
                        # strings collide by accident (e.g. "AI Inc" vs
                        # "AI-Inc" referring to different organisations).
                        if float(semantic_sim[i, j]) >= 0.5:
                            semantic_sim[i, j] = max(float(semantic_sim[i, j]), 0.85)
                            semantic_sim[j, i] = semantic_sim[i, j]
            return semantic_sim

        # 2. BM25 similarity
        bm25 = BM25Indexer(k1=self.bm25_k1, b=self.bm25_b)
        bm25.build_index(names)
        bm25_sim = bm25.get_pairwise_scores(names)

        # 3. Weighted fusion
        hybrid_sim = self.semantic_weight * semantic_sim + self.bm25_weight * bm25_sim

        # 4. Exact match bonus — only when semantic similarity is >= 0.5,
        #    to avoid merging entities whose normalised names collide by
        #    accident but are semantically unrelated.
        for i in range(n):
            for j in range(i + 1, n):
                norm_i = normalize_for_matching(names[i])
                norm_j = normalize_for_matching(names[j])
                if norm_i and norm_i == norm_j:
                    if float(semantic_sim[i, j]) >= 0.5:
                        hybrid_sim[i, j] = max(float(hybrid_sim[i, j]), 0.85)
                        hybrid_sim[j, i] = hybrid_sim[i, j]

        return hybrid_sim

    def align(
        self,
        entities: list[EntityNode],
    ) -> list[list[int]]:
        """Return groups of indices that should be merged.

        Each group contains indices into `entities` that refer to the same
        real-world entity, based on hybrid similarity.
        """
        if len(entities) < 2:
            return []

        # Group by alignment type — stakeholder subtypes merge together
        by_type: dict[str, list[int]] = {}
        for i, e in enumerate(entities):
            by_type.setdefault(_alignment_type(e), []).append(i)

        groups: list[list[int]] = []

        for atype, indices in by_type.items():
            if len(indices) < 2:
                continue

            typed_entities = [entities[i] for i in indices]
            names = [e.name for e in typed_entities]

            # Compute hybrid similarity
            sim_matrix = self._compute_hybrid_scores(names)

            uf = UnionFind(len(indices))
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    if sim_matrix[a, b] >= self.threshold:
                        uf.union(a, b)

            # Collect groups with >1 member
            root_to_members: dict[int, list[int]] = {}
            for local_i in range(len(indices)):
                root = uf.find(local_i)
                root_to_members.setdefault(root, []).append(local_i)

            for members in root_to_members.values():
                if len(members) > 1:
                    groups.append([indices[m] for m in members])

        return groups

    def get_candidate_pairs(
        self,
        entities: list[EntityNode],
        low: float = 0.7,
        high: float = 0.9,
    ) -> list[tuple[int, int, float]]:
        """Return pairs in the boundary zone for LLM verification."""
        if len(entities) < 2:
            return []

        # Use _alignment_type so stakeholder subtypes (AIProvider, AIDeveloper, ...)
        # are grouped together — otherwise "Google" as AIProvider and "Google" as
        # AIDeployer would never be compared.
        by_type: dict[str, list[int]] = {}
        for i, e in enumerate(entities):
            by_type.setdefault(_alignment_type(e), []).append(i)

        pairs: list[tuple[int, int, float]] = []

        for etype, indices in by_type.items():
            if len(indices) < 2:
                continue

            typed_entities = [entities[i] for i in indices]
            names = [e.name for e in typed_entities]

            # Compute hybrid similarity
            sim_matrix = self._compute_hybrid_scores(names)

            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    sim = float(sim_matrix[a, b])
                    if low <= sim < high:
                        pairs.append((indices[a], indices[b], sim))

        return pairs


def align_entities(
    entities: list[EntityNode],
    threshold: float = 0.8,
) -> list[list[int]]:
    """Convenience function: return merge groups."""
    aligner = SemanticAligner.from_config()
    aligner.threshold = threshold
    return aligner.align(entities)
