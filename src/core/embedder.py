"""Embedding service using BGE-M3 for semantic entity alignment."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src.core.config import get_config

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        cfg = get_config()
        model_name = cfg.get("embedding.model", "BAAI/bge-m3")
        device = cfg.get("embedding.device", "cpu")
        logger.info(f"Loading embedding model: {model_name} on {device}")
        _model = SentenceTransformer(model_name, device=device)
        logger.info(f"Embedding model loaded, dim={_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (len(texts), dim)."""
    if not texts:
        return np.array([])
    model = _get_model()
    cfg = get_config()
    batch_size = cfg.get("embedding.batch_size", 32)
    embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings)


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity. Vectors should be L2-normalized."""
    return vectors @ vectors.T


def compute_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))
