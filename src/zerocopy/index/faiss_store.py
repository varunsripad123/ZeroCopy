"""FAISS-backed vector store with persistence helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback when faiss is unavailable
    faiss = None


@dataclass(slots=True)
class SearchResult:
    """Container for search responses."""

    chunk_id: str
    score: float


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit length along axis 1."""

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class _FaissIndex:
    """Thin wrapper to abstract FAISS vs numpy implementations."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = None
        if faiss is not None:  # pragma: no cover - heavy dependency path
            self._index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        if self._index is not None:  # pragma: no branch - either faiss or numpy
            self._index.add(vectors)

    def search(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is not None:  # pragma: no cover - heavy dependency path
            scores, indices = self._index.search(vector, k)
            return scores, indices
        # numpy fallback: compute dot products manually
        # `vector` will be shape (1, dim) and we expect to be called with
        # `_vectors` stored on the parent for fallback use.
        raise RuntimeError("search requires numpy fallback handler")

    def reset(self) -> None:
        if self._index is not None:  # pragma: no cover - heavy dependency path
            self._index.reset()


class FaissStore:
    """Persisted FAISS index using inner-product similarity."""

    def __init__(self, dim: int, metric: str = "ip") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        metric = metric.lower()
        if metric != "ip":
            raise ValueError("only inner-product metric is supported")
        self.dim = dim
        self.metric = metric
        self._ids: List[str] = []
        self._vectors: np.ndarray | None = None
        self._index = _FaissIndex(dim)

    def _ensure_matrix(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = np.empty((0, self.dim), dtype="float32")
        return self._vectors

    def add(self, ids: Sequence[str], vecs: np.ndarray) -> None:
        if len(ids) == 0:
            return
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError("vector dimensionality mismatch")
        if vecs.shape[0] != len(ids):
            raise ValueError("ids and vectors length mismatch")

        vecs = np.asarray(vecs, dtype="float32")
        vecs = _normalize_matrix(vecs)

        self._ids.extend([str(_id) for _id in ids])
        matrix = self._ensure_matrix()
        self._vectors = np.concatenate([matrix, vecs], axis=0)

        if self._index._index is not None:  # pragma: no branch - dispatch based on backend
            self._index.add(vecs)

    def search(self, vec: np.ndarray, k: int) -> List[SearchResult]:
        if k <= 0:
            return []
        if self._vectors is None or not self._ids:
            return []

        vector = np.asarray(vec, dtype="float32")
        if vector.ndim == 1:
            vector = vector[np.newaxis, :]
        if vector.shape[1] != self.dim:
            raise ValueError("query dimension mismatch")
        vector = _normalize_matrix(vector)

        if self._index._index is not None:  # pragma: no branch
            scores, indices = self._index.search(vector, k)
            top_indices = indices[0]
            top_scores = scores[0]
        else:
            matrix = self._ensure_matrix()
            scores_all = matrix @ vector.T  # shape (n, 1)
            order = np.argsort(-scores_all[:, 0])[:k]
            top_indices = order
            top_scores = scores_all[:, 0][order]

        results: List[SearchResult] = []
        for raw_idx, raw_score in zip(top_indices, top_scores):
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self._ids):
                continue
            chunk_id = self._ids[idx]
            results.append(SearchResult(chunk_id=chunk_id, score=float(raw_score)))
            if len(results) >= k:
                break
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        vectors = self._ensure_matrix().astype("float32")
        np.savez_compressed(
            target,
            dim=np.array([self.dim], dtype="int32"),
            metric=np.array([self.metric], dtype=np.str_),
            ids=np.array(self._ids, dtype=np.str_),
            vectors=vectors,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FaissStore":
        data = np.load(Path(path))
        dim = int(data["dim"][0])
        metric_arr = data["metric"]
        metric = str(metric_arr[0]) if metric_arr.ndim > 0 else str(metric_arr)
        store = cls(dim=dim, metric=metric)
        ids_array = data["ids"].tolist()
        store._ids = [str(item) for item in ids_array]
        vectors = np.asarray(data["vectors"], dtype="float32")
        if vectors.size:
            store._vectors = _normalize_matrix(vectors)
            if store._index._index is not None:  # pragma: no branch
                store._index.add(store._vectors)
        else:
            store._vectors = np.empty((0, dim), dtype="float32")
        return store

    def __len__(self) -> int:
        return len(self._ids)


__all__ = ["FaissStore", "SearchResult"]
