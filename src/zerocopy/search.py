"""Semantic search utilities."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Sequence

from .logging import get_logger

log = get_logger(__name__)


def _normalize(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector[:]
    return [value / norm for value in vector]


class TextEncoder:
    """Deterministic text encoder placeholder."""

    def __init__(self, embedding_dim: int = 512) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        self.embedding_dim = embedding_dim

    def encode(self, text: str) -> List[float]:
        hasher = hashlib.sha256(text.encode("utf-8"))
        digest = hasher.digest()
        repeated = (digest * (self.embedding_dim // len(digest) + 1))[: self.embedding_dim]
        vector = [byte / 255.0 for byte in repeated]
        return _normalize(vector)


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    score: float


class VectorIndex:
    """In-memory vector index with cosine similarity."""

    def __init__(self, embedding_dim: int = 512) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        self.embedding_dim = embedding_dim
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []

    def add(self, chunk_id: str, vector: List[float]) -> None:
        if len(vector) != self.embedding_dim:
            raise ValueError("embedding dimension mismatch")
        self._ids.append(chunk_id)
        self._vectors.append(_normalize(vector))
        log.debug("index.add", chunk_id=chunk_id, total=len(self._ids))

    def add_bulk(self, chunk_ids: Sequence[str], vectors: Sequence[List[float]]) -> None:
        for chunk_id, vector in zip(chunk_ids, vectors):
            self.add(chunk_id, vector)

    def query(self, vector: List[float], top_k: int = 5) -> List[SearchResult]:
        if not self._vectors:
            return []
        query_vec = _normalize(vector)
        scores = []
        for stored in self._vectors:
            score = sum(a * b for a, b in zip(stored, query_vec))
            scores.append(score)
        paired = list(zip(self._ids, scores))
        paired.sort(key=lambda item: item[1], reverse=True)
        top = paired[:top_k]
        results = [SearchResult(chunk_id=chunk_id, score=float(score)) for chunk_id, score in top]
        log.debug("index.query", top_k=top_k, results=len(results))
        return results

    def __len__(self) -> int:
        return len(self._ids)


__all__ = ["TextEncoder", "VectorIndex", "SearchResult"]
