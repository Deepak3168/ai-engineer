"""Embedding providers behind one interface (Day 8).

* `HashEmbedder`   - deterministic, offline, no dependencies beyond numpy. It is a
                     hashed bag of words + character trigrams. It captures *lexical*
                     similarity, not meaning - which is exactly the point of using it
                     as a teaching baseline and as a stable test double.
* `OpenAICompatEmbedder` - any OpenAI-compatible /embeddings endpoint.

Vectors are always L2-normalised, so cosine similarity == dot product.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), rows L2-normalised."""
        ...


def normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def cosine_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """(q, d) similarity matrix for L2-normalised inputs."""
    return queries @ docs.T


def _bucket(token: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
    return h % dim, 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias


class HashEmbedder:
    def __init__(self, dim: int = 384, use_trigrams: bool = True) -> None:
        self.dim = dim
        self.use_trigrams = use_trigrams

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                idx, sign = _bucket("w:" + tok, self.dim)
                out[row, idx] += sign
                if self.use_trigrams and len(tok) > 3:
                    padded = f"#{tok}#"
                    for i in range(len(padded) - 2):
                        idx, sign = _bucket("t:" + padded[i : i + 3], self.dim)
                        out[row, idx] += 0.3 * sign
        return normalize(out)


class OpenAICompatEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dim: int | None = None,
        batch_size: int = 64,
        timeout_s: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("EMBED_API_KEY") or os.environ.get("LLM_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("EMBED_API_KEY (or LLM_API_KEY) is not set")
        self.base_url = base_url or os.environ.get("EMBED_BASE_URL") or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self.dim = dim or 0  # learned from the first response when unknown
        self.batch_size = batch_size
        from openai import OpenAI

        self._sdk = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_s, max_retries=0)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            resp = self._sdk.embeddings.create(model=self.model, input=batch)
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        arr = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        self.dim = arr.shape[1] if arr.size else self.dim
        return normalize(arr)


def get_embedder() -> Embedder:
    """Real embedder when EMBED_API_KEY is set (and not LESSON_OFFLINE), else HashEmbedder."""
    if os.environ.get("EMBED_API_KEY") and not os.environ.get("LESSON_OFFLINE"):
        return OpenAICompatEmbedder()
    return HashEmbedder()
