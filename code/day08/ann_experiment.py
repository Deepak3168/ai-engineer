"""Measured ANN-vs-exact experiment: recall@k and latency of pgvector HNSW as `ef_search` grows.

Data is synthetic (Gaussian clusters on the unit sphere) because the point is the *mechanism*, not any
claim about your embeddings: real embedding distributions differ and you must re-measure on your own.
Ground truth = exact cosine top-k computed in numpy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from common.embeddings import normalize
from common.infra import temp_schema
from day08.chunking import Chunk
from day08.pgvector_store import PgVectorStore


def synthetic_vectors(n: int, dim: int, n_queries: int, n_clusters: int = 50, noise: float = 0.8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    assign = rng.integers(0, n_clusters, size=n + n_queries)
    pts = centers[assign] + noise * rng.normal(size=(n + n_queries, dim))
    pts = normalize(pts.astype(np.float32))
    return pts[:n], pts[n:]


def exact_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    sims = queries @ corpus.T
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def recall_at_k(found: list[list[int]], truth: np.ndarray) -> float:
    k = truth.shape[1]
    return float(np.mean([len(set(f) & set(t.tolist())) / k for f, t in zip(found, truth, strict=True)]))


@dataclass(frozen=True)
class Row:
    setting: str
    recall: float
    ms_per_query: float


def run_experiment(
    n: int = 20000, dim: int = 128, n_queries: int = 100, k: int = 10, ef_values: tuple[int, ...] = (10, 20, 40, 100, 200, 400),
    m: int = 16, ef_construction: int = 64, noise: float = 0.8, seed: int = 0,
) -> tuple[list[Row], float]:
    """Returns (rows, index_build_seconds). Needs Postgres with pgvector; uses an isolated temp schema."""
    corpus, queries = synthetic_vectors(n, dim, n_queries, noise=noise, seed=seed)
    truth = exact_topk(corpus, queries, k)
    rows: list[Row] = []
    with temp_schema(require_pgvector=True) as (url, schema):
        store = PgVectorStore(url, schema, dim)
        store.create_table()
        chunks = [Chunk(id=f"v{i}", doc_id=f"v{i}", index=0, text="", start=0, end=0) for i in range(n)]
        store.upsert(chunks, corpus)
        t0 = time.perf_counter()
        store.create_hnsw_index(m=m, ef_construction=ef_construction)
        build_s = time.perf_counter() - t0

        conn = store.open()  # one connection: we want query time, not connection-setup time

        def measure(label: str, **kw: object) -> Row:
            found: list[list[int]] = []
            t = time.perf_counter()
            for q in queries:
                found.append([int(h.id[1:]) for h in store.search(q, k, conn=conn, **kw)])  # type: ignore[arg-type]
            ms = (time.perf_counter() - t) * 1000 / len(queries)
            return Row(label, recall_at_k(found, truth), ms)

        try:
            rows.append(measure("exact (seq scan)", exact=True))
            rows.extend(measure(f"hnsw ef_search={ef}", ef_search=ef) for ef in ef_values)
        finally:
            conn.close()
    return rows, build_s


def format_table(rows: list[Row]) -> str:
    lines = [f"{'setting':<22}{'recall@k':>10}{'ms/query':>10}"]
    lines += [f"{r.setting:<22}{r.recall:>10.3f}{r.ms_per_query:>10.2f}" for r in rows]
    return "\n".join(lines)
