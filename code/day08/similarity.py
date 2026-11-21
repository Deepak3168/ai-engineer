"""Similarity metrics. For L2-normalised vectors cosine, dot product and Euclidean distance are
monotonically related, which is why the rest of the code base normalises once and uses dot products."""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(a @ b / (na * nb))


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def euclidean_sq(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d @ d)
