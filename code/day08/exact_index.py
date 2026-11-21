"""Exact (brute-force) vector index in numpy with metadata filtering.

Exact search is O(N*d) per query. For a few hundred thousand vectors of 384-1536 dimensions that is
still milliseconds-to-tens-of-ms on one core with BLAS, and it is the ground truth that any
approximate index is measured against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Metadata = dict[str, Any]


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    metadata: Metadata = field(default_factory=dict)


class ExactIndex:
    def __init__(self, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self._ids: list[str] = []
        self._meta: list[Metadata] = []
        self._vecs = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], vectors: np.ndarray, metadata: Sequence[Metadata] | None = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors of shape (n, {self.dim}), got {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if len(set(ids)) != len(ids) or set(ids) & set(self._ids):
            raise ValueError("duplicate ids")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms[norms > 0], 1.0, atol=1e-3):
            raise ValueError("vectors must be L2-normalised (cosine == dot product relies on it)")
        self._ids.extend(ids)
        self._meta.extend(metadata if metadata is not None else [{} for _ in ids])
        self._vecs = np.vstack([self._vecs, vectors])

    def search(self, query: np.ndarray, k: int = 5, where: Callable[[Metadata], bool] | None = None) -> list[Hit]:
        """Top-k by cosine similarity. `where` is applied as a *pre-filter*, so k results are returned
        whenever at least k vectors pass the filter (unlike post-filtering an ANN top-k)."""
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query has dim {query.shape[0]}, index has dim {self.dim}")
        if k < 1:
            raise ValueError("k must be >= 1")
        candidates = np.arange(len(self._ids))
        if where is not None:
            candidates = np.array([i for i in candidates if where(self._meta[i])], dtype=int)
        if candidates.size == 0:
            return []
        scores = self._vecs[candidates] @ query
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(self._ids[candidates[i]], float(scores[i]), self._meta[candidates[i]]) for i in top]
