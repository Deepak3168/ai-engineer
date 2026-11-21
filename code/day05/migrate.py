"""Checkpoint schema migration.

Checkpoints outlive the code that wrote them: a run paused on Friday is resumed by
Monday's deploy. So every stored document carries `schema_version`, and loading goes
through an explicit chain of pure upgrade functions *before* Pydantic validates.

v1 -> v2 (the example change):
  * status "done" was renamed "completed"
  * `tool_outputs: {call_id: str}` became `tool_records: {call_id: ToolRecord}`
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from day05.state import SCHEMA_VERSION


class UnsupportedSchema(Exception):
    """The checkpoint is newer than this code, or has no usable version. Never guess."""


def _v1_to_v2(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    if out.get("status") == "done":
        out["status"] = "completed"
    records: dict[str, Any] = {}
    for call_id, content in (out.pop("tool_outputs", None) or {}).items():
        # v1 did not record the tool name; keep a visible marker rather than inventing one.
        records[call_id] = {"call_id": call_id, "name": "unknown(v1)", "status": "finished", "ok": True,
                            "content": content}
    out["tool_records"] = records
    out["schema_version"] = 2
    return out


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {1: _v1_to_v2}


def migrate(doc: dict[str, Any]) -> dict[str, Any]:
    version = doc.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise UnsupportedSchema(f"checkpoint has no valid schema_version: {version!r}")
    if version > SCHEMA_VERSION:
        raise UnsupportedSchema(
            f"checkpoint schema v{version} is newer than this code (v{SCHEMA_VERSION}); refusing to load"
        )
    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise UnsupportedSchema(f"no migration from schema v{version}")
        doc = step(doc)
        version = doc["schema_version"]
    return doc
