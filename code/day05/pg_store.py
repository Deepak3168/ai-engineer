"""Postgres checkpoint store: append-only history, optimistic concurrency via the primary key."""

from __future__ import annotations

import re

import psycopg
from psycopg import errors as pg_errors

from day05.state import RunState
from day05.store import CheckpointConflict, parse_state, validate_run_id

_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

DDL = """
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    run_id         text        NOT NULL,
    seq            integer     NOT NULL,
    status         text        NOT NULL,
    schema_version integer     NOT NULL,
    state          jsonb       NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS agent_checkpoints_status_idx ON agent_checkpoints (status);
"""


class PostgresCheckpointStore:
    """Every checkpoint is a new row `(run_id, seq)`.

    * Two workers resuming the same run both try to insert seq N+1; the primary key lets exactly
      one win and the other gets `CheckpointConflict` instead of silently interleaving histories.
    * History is kept (useful for debugging and time-travel); `prune` bounds growth.
    * `schema` lets tests (and multi-tenant setups) isolate tables; it is validated because
      identifiers cannot be bound as parameters.
    """

    def __init__(self, url: str, schema: str | None = None) -> None:
        if schema is not None and not _SCHEMA_RE.fullmatch(schema):
            raise ValueError(f"invalid schema name {schema!r}")
        self.url = url
        self.schema = schema

    def _connect(self) -> psycopg.Connection:
        options = f"-c search_path={self.schema},public" if self.schema else None
        return psycopg.connect(self.url, autocommit=True, options=options, connect_timeout=5)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(DDL)

    def save(self, state: RunState) -> None:
        validate_run_id(state.run_id)
        with self._connect() as conn:
            latest = conn.execute("SELECT max(seq) FROM agent_checkpoints WHERE run_id = %s", (state.run_id,)).fetchone()
            if latest and latest[0] is not None and latest[0] >= state.seq:
                raise CheckpointConflict(f"run {state.run_id}: stored seq {latest[0]} >= new seq {state.seq}")
            try:
                conn.execute(
                    "INSERT INTO agent_checkpoints (run_id, seq, status, schema_version, state) "
                    "VALUES (%s, %s, %s, %s, %s::jsonb)",
                    (state.run_id, state.seq, state.status.value, state.schema_version, state.model_dump_json()),
                )
            except pg_errors.UniqueViolation as exc:
                raise CheckpointConflict(f"run {state.run_id}: seq {state.seq} already written") from exc

    def load(self, run_id: str, seq: int | None = None) -> RunState | None:
        """Latest checkpoint, or a specific `seq` for time-travel debugging."""
        validate_run_id(run_id)
        with self._connect() as conn:
            if seq is None:
                row = conn.execute(
                    "SELECT state FROM agent_checkpoints WHERE run_id = %s ORDER BY seq DESC LIMIT 1", (run_id,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT state FROM agent_checkpoints WHERE run_id = %s AND seq = %s", (run_id, seq)
                ).fetchone()
        return parse_state(row[0]) if row else None

    def list_runs(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT run_id FROM agent_checkpoints ORDER BY run_id").fetchall()
        return [r[0] for r in rows]

    def history(self, run_id: str) -> list[tuple[int, str]]:
        """(seq, status) for every checkpoint, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT seq, status FROM agent_checkpoints WHERE run_id = %s ORDER BY seq", (validate_run_id(run_id),)
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def prune(self, run_id: str, keep_last: int = 20) -> int:
        """Delete all but the newest `keep_last` checkpoints of a run; returns rows deleted."""
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1")
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM agent_checkpoints WHERE run_id = %s AND seq <= "
                "(SELECT max(seq) FROM agent_checkpoints WHERE run_id = %s) - %s",
                (validate_run_id(run_id), run_id, keep_last),
            )
            return cur.rowcount
