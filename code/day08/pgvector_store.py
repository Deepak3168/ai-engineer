"""pgvector-backed chunk store with an optional HNSW index.

Verified against pgvector (extension 0.8.0) and the `pgvector` Python package 0.4+ on psycopg 3:
  * `<=>` is cosine *distance* (1 - cosine similarity); `vector_cosine_ops` is the matching operator class.
  * `hnsw.ef_search` is a per-session/transaction GUC (default 40) - it bounds the candidate list, so it
    must be >= LIMIT or fewer rows come back.
  * `hnsw.iterative_scan` (added in pgvector 0.8.0) keeps scanning the index until enough rows survive a WHERE filter.
  * Planner switch `enable_indexscan = off` forces a sequential scan + sort, i.e. exact search over the same table.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from day08.chunking import Chunk
from day08.exact_index import Hit

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name


class PgVectorStore:
    """All operations open a short-lived connection with `search_path = <schema>, public`
    (`public` keeps the `vector` type/operators visible). Production code would use a pool."""

    def __init__(self, url: str, schema: str, dim: int, table: str = "chunks") -> None:
        self.url, self.schema, self.dim, self.table = url, _ident(schema), dim, _ident(table)

    def _connect(self) -> psycopg.Connection[Any]:
        conn = psycopg.connect(self.url, options=f"-c search_path={self.schema},public", autocommit=True)
        register_vector(conn)
        return conn

    def open(self) -> psycopg.Connection[Any]:
        """A long-lived connection with the right search_path and vector adapters registered."""
        return self._connect()

    def create_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.table} (
                    id text PRIMARY KEY, doc_id text NOT NULL, idx int NOT NULL, text text NOT NULL,
                    title text NOT NULL DEFAULT '', tags text[] NOT NULL DEFAULT '{{}}',
                    embedding vector({int(self.dim)}) NOT NULL)"""
            )

    def create_hnsw_index(self, m: int = 16, ef_construction: int = 64) -> None:
        with self._connect() as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} USING hnsw "
                f"(embedding vector_cosine_ops) WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        rows = [
            (c.id, c.doc_id, c.index, c.text, c.title, c.tags, np.asarray(v, dtype=np.float32))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (id, doc_id, idx, text, title, tags, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET doc_id=EXCLUDED.doc_id, idx=EXCLUDED.idx, text=EXCLUDED.text,
                        title=EXCLUDED.title, tags=EXCLUDED.tags, embedding=EXCLUDED.embedding""",
                rows,
            )
        return len(rows)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])  # type: ignore[index]

    def search(
        self, query: np.ndarray, k: int = 5, *, tag: str | None = None, ef_search: int | None = None, exact: bool = False,
        iterative_scan: str | None = None, conn: psycopg.Connection[Any] | None = None,
    ) -> list[Hit]:
        """Top-k by cosine similarity. `exact=True` disables index scans (sequential scan + sort).
        Pass `conn` (from `open()`) to reuse a connection, e.g. when timing queries. `tag` is a WHERE filter; with an HNSW index it is applied *after* the index returns its
        ef_search candidates, so a selective filter can return fewer than k rows (see lesson)."""
        where = "WHERE tags @> ARRAY[%s]::text[]" if tag else ""
        params: list[Any] = [np.asarray(query, dtype=np.float32)]
        if tag:
            params.append(tag)
        params += [np.asarray(query, dtype=np.float32), k]
        sql = (
            f"SELECT id, 1 - (embedding <=> %s) AS score, doc_id, title, tags FROM {self.table} {where} "
            "ORDER BY embedding <=> %s LIMIT %s"
        )
        if iterative_scan not in (None, "strict_order", "relaxed_order"):
            raise ValueError("iterative_scan must be None, 'strict_order' or 'relaxed_order'")
        own = conn is None
        c = self._connect() if conn is None else conn
        try:
            with c.transaction():
                if exact:
                    c.execute("SET LOCAL enable_indexscan = off")
                if ef_search is not None:
                    c.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                if iterative_scan:
                    c.execute(f"SET LOCAL hnsw.iterative_scan = {iterative_scan}")
                with c.cursor(row_factory=dict_row) as cur:
                    rows = cur.execute(sql, params).fetchall()
        finally:
            if own:
                c.close()
        return [Hit(r["id"], float(r["score"]), {"doc_id": r["doc_id"], "title": r["title"], "tags": r["tags"]}) for r in rows]

    def explain(self, query: np.ndarray, k: int = 5, *, exact: bool = False) -> str:
        with self._connect() as conn, conn.transaction():
            if exact:
                conn.execute("SET LOCAL enable_indexscan = off")
            rows = conn.execute(
                f"EXPLAIN SELECT id FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
                (np.asarray(query, dtype=np.float32), k),
            ).fetchall()
        return "\n".join(r[0] for r in rows)
