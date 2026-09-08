"""Embedding service for semantic entity alignment."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from src.core.config import get_config

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to keep cosine similarity behavior consistent."""
    if embeddings.size == 0:
        return embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _get_local_model() -> Any:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                cfg = get_config()
                model_name = cfg.get("embedding.model", "BAAI/bge-m3")
                device = cfg.get("embedding.device", "cpu")
                logger.info(f"Loading embedding model: {model_name} on {device}")
                _model = SentenceTransformer(model_name, device=device)
                logger.info(f"Embedding model loaded, dim={_model.get_sentence_embedding_dimension()}")
    return _model


def _get_openai_client() -> Any:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from openai import OpenAI

                cfg = get_config()
                base_url = str(cfg.get("embedding.base_url", "")).rstrip("/")
                api_key = cfg.get("embedding.api_key", "") or "EMPTY"
                timeout = cfg.get("embedding.timeout", 180)

                logger.info(f"Connecting to embedding API: {base_url or 'default OpenAI endpoint'}")
                _model = OpenAI(
                    base_url=base_url or None,
                    api_key=api_key,
                    max_retries=3,
                    timeout=timeout,
                )
    return _model


def _get_provider() -> str:
    provider = str(get_config().get("embedding.provider", "local")).strip().lower()
    if provider not in {"local", "openai"}:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    return provider


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (len(texts), dim)."""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)

    cfg = get_config()
    batch_size = cfg.get("embedding.batch_size", 32)
    provider = _get_provider()

    if provider == "local":
        model = _get_local_model()
        embeddings = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    client = _get_openai_client()
    model_name = cfg.get("embedding.model", "text-embedding-3-large")

    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model_name, input=batch)
        ordered = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        batch_array = np.asarray(ordered, dtype=np.float32)
        all_embeddings.append(batch_array)

    embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([], dtype=np.float32)
    return _normalize_embeddings(embeddings)


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
