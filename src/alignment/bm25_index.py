"""BM25 indexer for entity name matching."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BM25Indexer:
    """BM25 indexer for entity name similarity matching.

    Handles technical entity names with custom tokenization:
    - Split on hyphens, underscores, dots: "GPT-4" -> ["gpt", "4"]
    - Split camelCase: "ChatGPT" -> ["chat", "gpt"]
    - Preserve numbers with decimals: "Claude 3.5" -> ["claude", "3.5"]
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        """Initialize BM25 indexer.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization (default: 0.75)
            epsilon: Floor value for IDF to prevent negative scores
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Index state
        self._names: list[str] = []
        self._tokenized: list[list[str]] = []
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0
        self._df: dict[str, int] = {}  # document frequency
        self._idf: dict[str, float] = {}  # inverse document frequency
        self._doc_freqs: list[dict[str, int]] = []  # term freq per doc

    def tokenize(self, text: str) -> list[str]:
        """Tokenize entity name with special handling for technical names.

        Examples:
            "GPT-4" -> ["gpt", "4"]
            "ChatGPT" -> ["chat", "gpt"]
            "Claude 3.5" -> ["claude", "3.5"]
            "OpenAI_API" -> ["open", "ai", "api"]
        """
        if not text:
            return []

        # Normalize
        text = text.lower().strip()

        # Replace underscores and dots with spaces
        text = re.sub(r"[_.]", " ", text)

        # Split camelCase: "ChatGPT" -> "Chat GPT"
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Split on hyphens but keep the parts
        text = re.sub(r"-", " ", text)

        # Split numbers from letters: "GPT4" -> "GPT 4"
        text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

        # Tokenize
        tokens = text.split()

        # Filter empty tokens
        tokens = [t for t in tokens if t]

        return tokens

    def build_index(self, names: list[str]) -> None:
        """Build BM25 index from entity names.

        Args:
            names: List of entity names to index
        """
        self._names = names
        self._tokenized = [self.tokenize(name) for name in names]
        self._doc_len = [len(tokens) for tokens in self._tokenized]
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 1.0

        # Compute document frequencies
        self._df: dict[str, int] = {}
        self._doc_freqs: list[dict[str, int]] = []

        for tokens in self._tokenized:
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self._doc_freqs.append(freq)

            # Update document frequency (each term counted once per doc)
            for token in set(tokens):
                self._df[token] = self._df.get(token, 0) + 1

        # Compute IDF
        n_docs = len(names)
        self._idf: dict[str, float] = {}
        for term, df in self._df.items():
            idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            self._idf[term] = max(idf, self.epsilon)

        logger.debug(
            f"BM25 index built: {n_docs} docs, {len(self._df)} terms, avgdl={self._avgdl:.2f}"
        )

    def get_scores(self, query: str) -> np.ndarray:
        """Get BM25 scores for a query against all indexed documents.

        Args:
            query: Query string (entity name)

        Returns:
            Array of normalized scores [0, 1] for each document
        """
        if not self._names:
            return np.array([])

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return np.zeros(len(self._names))

        scores = np.zeros(len(self._names))

        for i, doc_freq in enumerate(self._doc_freqs):
            score = 0.0
            doc_len = self._doc_len[i]

            for term in query_tokens:
                if term not in self._idf:
                    continue

                tf = doc_freq.get(term, 0)
                if tf == 0:
                    continue

                idf = self._idf[term]
                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self._avgdl
                )
                score += idf * numerator / denominator

            scores[i] = score

        # Normalize to [0, 1]
        max_score = scores.max() if len(scores) > 0 else 1.0
        if max_score > 0:
            scores = scores / max_score

        return scores

    def get_pairwise_scores(self, queries: list[str]) -> np.ndarray:
        """Get pairwise BM25 scores between queries and indexed documents.

        Args:
            queries: List of query strings (same as indexed names for self-comparison)

        Returns:
            Symmetric matrix of normalized scores [0, 1]
        """
        n = len(queries)
        matrix = np.zeros((n, n))

        for i, query in enumerate(queries):
            scores = self.get_scores(query)
            matrix[i] = scores

        # Make symmetric (average of both directions)
        matrix = (matrix + matrix.T) / 2

        return matrix

    def get_name(self, idx: int) -> str:
        """Get indexed name by index."""
        return self._names[idx] if 0 <= idx < len(self._names) else ""
