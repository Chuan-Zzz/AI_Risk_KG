"""Alignment module for entity normalization and merging."""

from __future__ import annotations

from .bm25_index import BM25Indexer
from .semantic_aligner import SemanticAligner, align_entities
from .llm_verifier import LLMVerifier, verify_pairs
from .cross_event import EntityRegistry

__all__ = [
    "BM25Indexer",
    "SemanticAligner",
    "align_entities",
    "LLMVerifier",
    "verify_pairs",
    "EntityRegistry",
]
