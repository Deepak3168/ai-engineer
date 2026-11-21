"""Checkpoint stores: the durable half of the runtime.

Contract (shared by the file and Postgres implementations):
  * `save(state)` persists a full snapshot. `state.seq` must be strictly greater than the
    latest stored seq, otherwise `CheckpointConflict` (two workers drove the same run).
  * `load(run_id)` returns the latest snapshot, migrated to the current schema, or None.
  * A torn or corrupt snapshot raises `CorruptCheckpoint`; it is never silently ignored.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from day05.migrate import migrate
from day05.state import RunState

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class CheckpointConflict(Exception):
    """Another writer already stored a checkpoint with this seq (or a later one)."""


class CorruptCheckpoint(Exception):
    """Stored data could not be parsed/validated as a RunState."""


def validate_run_id(run_id: str) -> str:
    """Run ids become file names, so keep them boring. Also blocks path traversal."""
    if not _RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(f"invalid run_id {run_id!r}: use 1-128 chars of letters, digits, '.', '_' or '-'")
    return run_id


def parse_state(raw: str | dict | bytes) -> RunState:
    try:
        doc = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if not isinstance(doc, dict):
            raise TypeError(f"expected a JSON object, got {type(doc).__name__}")
        return RunState.model_validate(migrate(doc))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise CorruptCheckpoint(f"cannot read checkpoint: {exc}") from exc


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, state: RunState) -> None: ...

    def load(self, run_id: str) -> RunState | None: ...

    def list_runs(self) -> list[str]: ...


class FileCheckpointStore:
    """One JSON file per run, replaced atomically.

    Write to a temp file in the same directory, fsync, then `os.replace` (atomic on POSIX
    and Windows for same-volume paths). A crash leaves either the old or the new snapshot,
    never half of one. Conflict detection here is read-then-write, so it is NOT race-free
    between concurrent processes; use Postgres when more than one worker can touch a run.
    """

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.dir / f"{validate_run_id(run_id)}.json"

    def save(self, state: RunState) -> None:
        path = self._path(state.run_id)
        current = self.load(state.run_id)
        if current is not None and current.seq >= state.seq:
            raise CheckpointConflict(f"run {state.run_id}: stored seq {current.seq} >= new seq {state.seq}")
        payload = state.model_dump_json(indent=None)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{state.run_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self, run_id: str) -> RunState | None:
        path = self._path(run_id)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return parse_state(raw)

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.json"))
