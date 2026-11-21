# Day 5: Agent state, checkpoints, and execution lifecycle

> **Track:** Foundations · **Time:** ~40 min · **Builds on:** [Day 3](day-03-agent-loops-and-react.md), [Day 4](day-04-planning-and-task-decomposition.md)

## 1. Learning objectives

By the end of this lesson you will be able to:

- Model a run as an explicit state machine with legal transitions, and explain why `running` is the only state a crash leaves behind.
- Define agent state as plain serialisable data with nothing live inside it.
- Checkpoint after every LLM response and every tool result, and resume from the latest checkpoint in a new process.
- Prove that a resume does not repeat completed side effects, and explain what remains ambiguous when a crash happens *during* a side effect.
- Prevent two workers from driving the same run (optimistic concurrency by sequence number).
- Version and migrate checkpoints so a run paused under old code resumes under new code.

## 2. Why this matters in real-world agent engineering

An agent run is a long, expensive, partly irreversible computation: it makes model calls that cost money and tool calls that change the world. Workers get deployed over, OOM-killed, and preempted. If your only copy of the run is a Python list in memory, a crash after the agent filed a ticket but before it wrote its answer means one of two bad outcomes: the run is lost, or it is restarted from scratch and files the ticket again.

The second outcome is the dangerous one, and it is what a naive "just retry the job" policy produces. Durable execution is not an optimisation for agents that call write tools. It is a correctness requirement.

## 3. Mental model

> **The checkpoint is the run. The process is a disposable worker that happens to be executing it.**

Every worker holds only a cache of the state. Anything not yet checkpointed is allowed to vanish; anything checkpointed must be enough to continue. Writing the loop so that it could stop after any line and be resumed is the discipline.

## 4. Architecture and execution flow

```mermaid
stateDiagram-v2
    [*] --> created
    created --> running
    created --> cancelled
    running --> completed
    running --> failed
    running --> cancelled
    failed --> running: resume() retries
    failed --> cancelled
    completed --> [*]
    cancelled --> [*]
```

```mermaid
sequenceDiagram
    participant W as Worker
    participant S as Checkpoint store
    participant M as Model
    participant T as Tool
    W->>S: save(created), save(running)
    loop until final answer
        W->>M: chat(messages)
        M-->>W: assistant message
        W->>S: save (after LLM response)
        W->>S: save intent (only for non-idempotent tools)
        W->>T: execute
        T-->>W: result
        W->>S: save (after tool result)
    end
    W->>S: save(completed)
    Note over W,S: kill -9 anywhere: new worker loads latest checkpoint and continues
```

## 5. Core concepts

**State is data.** `RunState` is a Pydantic model: run id, task, status, the message list, step count, token usage, a ledger of tool calls, and a monotonically increasing `seq`. Nothing in it refers to a socket, a closure or an SDK object. That is what lets *any* process pick it up.

**Lifecycle as a state machine.** `created`, `running`, `completed`, `failed`, `cancelled`, with an explicit table of legal transitions. Illegal moves raise `IllegalTransition` and do not mutate state. `failed -> running` is a deliberate retry edge; `completed` and `cancelled` are final.

**A crash is invisible to the state machine.** Nothing gets to write `failed` when the process is killed, so a crashed run still says `running`. That is the whole reason resume has to work from `running`, and why production systems add a *lease* or heartbeat to decide when a `running` run is actually orphaned (see Production considerations).

**Where to checkpoint.** After each LLM response (the response cost money; do not buy it twice) and after each tool result. That makes the resume point at most one in-flight operation behind.

**The tool ledger.** Each tool call gets a `ToolRecord` with status `started` or `finished`. On resume:

- `finished` call: **replay**. The recorded result is reused and the tool is not called again.
- no record: execute normally.
- `started` but not `finished` (crashed mid-call): if the tool is idempotent, run it again; if not, **do not guess**. Record `outcome_unknown` and tell the model so.

**Write-ahead intent.** For non-idempotent tools the runner saves the `started` record *before* running the tool. Without that, a crash mid-call leaves no trace and the runner cannot tell "never ran" from "ran and the result was lost".

**The unavoidable ambiguity.** If the process dies after the tool's side effect but before the checkpoint, nothing in your process can know whether it happened. The intent record turns that from "silently duplicated" into an explicit `outcome_unknown` the model can reason about (for example by calling a read tool to check). True exactly-once needs the *downstream system* to deduplicate, which is what idempotency keys are for ([Day 23](day-23-reliability-retries-timeouts-and-idempotency.md)).

**Optimistic concurrency.** Each save must carry a `seq` strictly greater than the stored one. If two workers resume the same run, the loser's save fails with `CheckpointConflict`. With Postgres this is enforced by the primary key `(run_id, seq)`, so exactly one concurrent insert wins.

**Schema versioning.** Checkpoints outlive the code that wrote them. Each document carries `schema_version`, and loading runs a chain of pure upgrade functions *before* Pydantic validation. A checkpoint from a newer schema is refused rather than guessed at.

## 6. Implementation from scratch

1. **`state.py`**: `RunState`, `RunStatus`, the `TRANSITIONS` table, `ToolRecord`, and `pending_calls()`, which finds tool calls in the last assistant message that have no tool message yet. That one method is what makes resume correct when the crash happened between two tool calls of the same assistant turn.
2. **`store.py`**: the `CheckpointStore` protocol and `FileCheckpointStore`. It writes to a temp file in the same directory, `fsync`s, then `os.replace`s, so a crash leaves the old or the new snapshot, never half of one. Run ids become file names, so they are validated to block path traversal.
3. **`pg_store.py`**: `PostgresCheckpointStore` appends one row per checkpoint, giving history and time-travel (`load(run_id, seq=...)`) for debugging, plus `prune`.
4. **`migrate.py`**: the version chain, with a worked v1 to v2 example (a renamed status and a restructured tool ledger).
5. **`runner.py`**: `DurableAgent`, the Day 3 loop with the invariants above. `start`, `resume` and `cancel` are the whole public API.

Simplifications to know about: the file store's conflict check is read-then-write, so it is **not** race-free across processes (use Postgres when more than one worker can touch a run); and the runner executes tool calls sequentially so the ledger stays simple.

## 7. Complete runnable code

```text
code/day05/
├── state.py        RunState, lifecycle, tool ledger
├── store.py        protocol + atomic file store
├── pg_store.py     Postgres store (append-only, optimistic concurrency)
├── migrate.py      schema versions
├── runner.py       DurableAgent
├── ops_tools.py    tools with an observable side-effect ledger
├── demo.py         four scenarios, offline
└── test_*.py       34 tests (5 need Postgres)
```

```bash
docker compose up -d
export DATABASE_URL=postgresql://agent:agent@localhost:5432/agents
PYTHONPATH=code LESSON_OFFLINE=1 python -m day05.demo
pytest code/day05
```

<!-- code:begin path="code/day05/state.py" -->
**`code/day05/state.py`**

```python
"""Run state: an explicit lifecycle plus everything needed to resume after a crash.

The state is plain data (Pydantic -> JSON). Nothing in it refers to a live object,
socket or closure, so any process can pick it up later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from common.llm import Message, ToolCall, Usage

SCHEMA_VERSION = 2


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Legal transitions. `failed -> running` is an explicit retry; completed/cancelled are final.
TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.FAILED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}
TERMINAL = frozenset({RunStatus.COMPLETED, RunStatus.CANCELLED})


class IllegalTransition(Exception):
    """Raised when code tries to move a run along an edge the state machine does not allow."""


def utc_now() -> datetime:
    return datetime.now(UTC)


class ToolRecord(BaseModel):
    """Ledger entry for one tool call. `started` without `finished` means 'crashed while running'."""

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["started", "finished"] = "started"
    ok: bool | None = None
    content: str | None = None
    error_type: str | None = None  # includes "outcome_unknown" (see runner.py)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None


class RunState(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    task: str
    status: RunStatus = RunStatus.CREATED
    seq: int = 0  # monotonically increasing checkpoint number; the store uses it to detect writers racing
    messages: list[Message] = Field(default_factory=list)
    steps: int = 0  # number of LLM calls made so far
    usage: Usage = Field(default_factory=Usage)
    tool_records: dict[str, ToolRecord] = Field(default_factory=dict)
    final_answer: str | None = None
    error: str | None = None
    resume_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def transition(self, new: RunStatus) -> None:
        if new not in TRANSITIONS[self.status]:
            raise IllegalTransition(f"{self.status.value} -> {new.value} is not allowed for run {self.run_id}")
        self.status = new
        self.updated_at = utc_now()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    def pending_calls(self) -> list[ToolCall]:
        """Tool calls of the last assistant message that have no tool message yet."""
        idx = len(self.messages) - 1
        while idx >= 0 and self.messages[idx].role == "tool":  # results follow their assistant turn
            idx -= 1
        if idx < 0:
            return []
        assistant = self.messages[idx]
        if assistant.role != "assistant" or not assistant.tool_calls:
            return []
        answered = {m.tool_call_id for m in self.messages[idx + 1 :] if m.role == "tool"}
        return [c for c in assistant.tool_calls if c.id not in answered]
```
<!-- code:end -->

<!-- code:begin path="code/day05/store.py" -->
**`code/day05/store.py`**

```python
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
```
<!-- code:end -->

<!-- code:begin path="code/day05/pg_store.py" -->
**`code/day05/pg_store.py`**

```python
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
```
<!-- code:end -->

<!-- code:begin path="code/day05/migrate.py" -->
**`code/day05/migrate.py`**

```python
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
```
<!-- code:end -->

<!-- code:begin path="code/day05/runner.py" -->
**`code/day05/runner.py`**

```python
"""A durable agent runner: the Day 3 loop, but every step is checkpointed and resumable.

Invariants
  1. State is saved *after* each LLM response and *after* each tool result.
  2. A non-idempotent tool gets an `started` intent record saved BEFORE it runs.
  3. On resume, finished tool calls are never executed again; their recorded result is reused.
  4. A call found `started` but not `finished` (crash mid-tool) is re-run only if the tool is
     idempotent; otherwise the model is told the outcome is unknown. We refuse to guess.

What cannot be fixed here: a crash between "tool did its side effect" and "checkpoint saved"
is inherently ambiguous. The intent record narrows it to an explicit 'outcome unknown' state;
true exactly-once needs the *tool's* system to be idempotent (Day 23).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from common.llm import LLMClient, LLMError, ToolCall, system, tool_result, user
from common.tools import ToolRegistry
from day05.state import RunState, RunStatus, ToolRecord, utc_now
from day05.store import CheckpointStore

EventHook = Callable[[dict[str, Any]], None]
UNKNOWN_OUTCOME = (
    "Error: the process crashed while this tool was running, so it is UNKNOWN whether it took effect. "
    "Do not repeat it blindly: verify the current state with a read-only tool first."
)


class RunNotFound(Exception):
    pass


class DurableAgent:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        store: CheckpointStore,
        system_prompt: str,
        *,
        max_steps: int = 10,
        on_event: EventHook | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.store = store
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self._emit = on_event or (lambda _e: None)

    # -- public API ---------------------------------------------------------
    def start(self, task: str, run_id: str | None = None) -> RunState:
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        if self.store.load(run_id) is not None:
            raise ValueError(f"run {run_id} already exists; use resume()")
        state = RunState(run_id=run_id, task=task, messages=[system(self.system_prompt), user(task)])
        self._checkpoint(state)  # a run exists durably before any work happens
        return self._drive(state)

    def resume(self, run_id: str) -> RunState:
        state = self.store.load(run_id)
        if state is None:
            raise RunNotFound(run_id)
        if state.is_terminal:
            return state
        state.resume_count += 1
        self._emit({"type": "resume", "run_id": run_id, "from_seq": state.seq, "status": state.status.value})
        return self._drive(state)

    def cancel(self, run_id: str) -> RunState:
        state = self.store.load(run_id)
        if state is None:
            raise RunNotFound(run_id)
        state.transition(RunStatus.CANCELLED)
        self._checkpoint(state)
        return state

    # -- internals ----------------------------------------------------------
    def _checkpoint(self, state: RunState) -> None:
        state.seq += 1
        state.updated_at = utc_now()
        self.store.save(state)
        self._emit({"type": "checkpoint", "run_id": state.run_id, "seq": state.seq, "status": state.status.value})

    def _fail(self, state: RunState, error: str) -> RunState:
        state.error = error
        state.transition(RunStatus.FAILED)
        self._checkpoint(state)
        return state

    def _drive(self, state: RunState) -> RunState:
        if state.status in (RunStatus.CREATED, RunStatus.FAILED):
            state.error = None
            state.transition(RunStatus.RUNNING)
            self._checkpoint(state)
        schemas = self.registry.schemas() or None
        while not state.is_terminal and state.status == RunStatus.RUNNING:
            pending = state.pending_calls()
            if pending:
                self._run_tools(state, pending)
                continue
            if state.messages[-1].role == "assistant":  # answered without tool calls: finished
                return self._complete(state)
            if state.steps >= self.max_steps:
                return self._fail(state, f"no final answer within {self.max_steps} steps")
            try:
                resp = self.client.chat(state.messages, tools=schemas)
            except LLMError as exc:  # expected provider failure -> durable 'failed', retryable via resume()
                return self._fail(state, f"{type(exc).__name__}: {exc}")
            # Anything else (a bug, SIGKILL, power loss) propagates: the last checkpoint stays as-is.
            state.steps += 1
            state.usage.prompt_tokens += resp.usage.prompt_tokens
            state.usage.completion_tokens += resp.usage.completion_tokens
            state.messages.append(resp.message)
            self._checkpoint(state)
        return state

    def _complete(self, state: RunState) -> RunState:
        state.final_answer = state.messages[-1].content
        state.transition(RunStatus.COMPLETED)
        self._checkpoint(state)
        return state

    def _run_tools(self, state: RunState, pending: list[ToolCall]) -> None:
        for call in pending:
            record = state.tool_records.get(call.id)
            tool = self.registry.get(call.name)
            if record is not None and record.status == "finished":
                # Replay: the recorded result is the truth; the tool is NOT called again.
                self._emit({"type": "tool_replayed", "tool": call.name, "call_id": call.id})
                content = record.content or ""
            elif record is not None and tool is not None and not tool.idempotent:
                # Crashed mid-call and re-running could duplicate the side effect.
                record.status, record.ok, record.error_type = "finished", False, "outcome_unknown"
                record.content, record.finished_at = UNKNOWN_OUTCOME, utc_now()
                self._emit({"type": "tool_outcome_unknown", "tool": call.name, "call_id": call.id})
                content = UNKNOWN_OUTCOME
            else:
                content = self._execute(state, call, tool_is_unsafe=tool is not None and not tool.idempotent)
            state.messages.append(tool_result(call, content))
            self._checkpoint(state)

    def _execute(self, state: RunState, call: ToolCall, *, tool_is_unsafe: bool) -> str:
        record = ToolRecord(call_id=call.id, name=call.name, arguments=call.arguments)
        state.tool_records[call.id] = record
        if tool_is_unsafe:
            self._checkpoint(state)  # durable intent BEFORE the side effect
        outcome = self.registry.execute(call)
        record.status, record.ok, record.error_type = "finished", outcome.ok, outcome.error_type
        record.content, record.finished_at = outcome.content, utc_now()
        self._emit({"type": "tool_executed", "tool": call.name, "call_id": call.id, "ok": outcome.ok})
        return outcome.content
```
<!-- code:end -->

<!-- code:begin path="code/day05/ops_tools.py" -->
**`code/day05/ops_tools.py`**

```python
"""Ops tools for the Day 5 runs, with an observable side-effect ledger.

`TicketLedger` counts how many times a write *actually executed*. Crash/resume tests assert
on that counter: a correct resume leaves it unchanged for already-completed calls.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from common.fixtures import get_incident_record, metrics_for
from common.tools import ToolRegistry


class TicketLedger:
    def __init__(self) -> None:
        self.created: list[dict[str, str]] = []
        self._by_key: dict[str, str] = {}

    def create(self, title: str, severity: str, key: str | None = None) -> str:
        if key is not None and key in self._by_key:
            return self._by_key[key]  # deduplicated: no second side effect
        ticket_id = f"TCK-{len(self.created) + 1:03d}"
        self.created.append({"id": ticket_id, "title": title, "severity": severity})
        if key is not None:
            self._by_key[key] = ticket_id
        return ticket_id


class IncidentArgs(BaseModel):
    incident_id: str = Field(description="Incident id such as INC-1042")


class MetricArgs(BaseModel):
    service: str
    metric: str = Field(description="error_rate | p95_latency_ms | cpu_pct")
    minutes: int = Field(default=10, ge=1, le=60)


class TicketArgs(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    severity: str = Field(pattern="^SEV[1-3]$")


class KeyedTicketArgs(TicketArgs):
    idempotency_key: str = Field(min_length=8, description="Same key => same ticket, never a duplicate")


def build_registry(ledger: TicketLedger) -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("get_incident", "Fetch an incident record by id.", IncidentArgs)
    def get_incident(incident_id: str) -> dict:
        return dict(get_incident_record(incident_id))

    @reg.tool("query_metrics", "Return the last N per-minute values of a metric for a service.", MetricArgs)
    def query_metrics(service: str, metric: str, minutes: int) -> list[dict[str, float]]:
        return metrics_for(service, metric, 20, minutes)

    # Non-idempotent write: running it twice creates two tickets. The runner must never replay it blindly.
    @reg.tool("create_ticket", "Create a follow-up ticket (NOT idempotent).", TicketArgs, side_effect="write")
    def create_ticket(title: str, severity: str) -> str:
        return ledger.create(title, severity)

    # Idempotent write: safe to re-execute after a crash because the key deduplicates.
    @reg.tool("create_ticket_keyed", "Create a follow-up ticket; safe to retry with the same key.",
              KeyedTicketArgs, side_effect="write", idempotent=True)
    def create_ticket_keyed(title: str, severity: str, idempotency_key: str) -> str:
        return ledger.create(title, severity, key=idempotency_key)

    return reg
```
<!-- code:end -->

<!-- code:begin path="code/day05/demo.py" -->
**`code/day05/demo.py`**

```python
"""Day 5 demo: crash a run mid-flight, resume it from disk, and prove no write was repeated. Offline.

    PYTHONPATH=code LESSON_OFFLINE=1 python -m day05.demo

The "crash" is an exception that is not an LLMError, so the runner (correctly) does not catch it. It stands
in for SIGKILL, an OOM kill or a deploy. Only what reached the checkpoint store survives, as in production.
"""

from __future__ import annotations

import tempfile

from common.llm import Message, ScriptedLLM, ToolCall
from day05.migrate import migrate
from day05.ops_tools import TicketLedger, build_registry
from day05.runner import DurableAgent
from day05.state import RunStatus
from day05.store import CheckpointConflict, FileCheckpointStore


class SimulatedCrash(Exception):
    """Not an LLMError: the runner must let it propagate, exactly like a killed process."""


def agent(llm: ScriptedLLM, registry, store) -> DurableAgent:
    return DurableAgent(llm, registry, store, "You are the Ops Report Agent.")


def ticket(call_id: str, name: str = "create_ticket", **extra: str) -> Message:
    return ScriptedLLM.tool_call(name, call_id=call_id, title="Add index on orders", severity="SEV2", **extra)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def scenario_resume_after_crash(store: FileCheckpointStore) -> None:
    banner("1. Crash after a write finished, resume in a 'new process'")
    ledger = TicketLedger()
    registry = build_registry(ledger)
    first = ScriptedLLM([ScriptedLLM.tool_call("get_incident", "c1", incident_id="INC-1042"), ticket("c2"),
                         SimulatedCrash("worker killed")])
    try:
        agent(first, registry, store).start("Investigate INC-1042 and file a follow-up ticket", "run-1")
    except SimulatedCrash as exc:
        print(f"process died: {exc}")
    on_disk = store.load("run-1")
    print(f"on disk: status={on_disk.status.value} seq={on_disk.seq} steps={on_disk.steps} tickets_created={len(ledger.created)}")
    print("tool ledger:", {c: (r.name, r.status) for c, r in on_disk.tool_records.items()})

    second = ScriptedLLM(["Filed TCK-001 for INC-1042."])
    state = agent(second, registry, store).resume("run-1")
    print(f"resumed: status={state.status.value} resume_count={state.resume_count} tickets_created={len(ledger.created)}")
    sent = [m.content for m in second.calls[0]["messages"] if m.role == "tool"]
    print("model saw recorded tool results, not re-executions:", [c[:40] for c in sent])


def scenario_unknown_outcome(store: FileCheckpointStore) -> None:
    banner("2. Crash INSIDE a non-idempotent tool: the honest answer is 'unknown'")
    ledger = TicketLedger()
    registry = build_registry(ledger)
    real = registry.execute

    def crash_after_effect(call: ToolCall):
        real(call)  # the ticket IS created...
        raise SimulatedCrash("...then the worker dies before the result is checkpointed")

    registry.execute = crash_after_effect  # type: ignore[method-assign]
    try:
        agent(ScriptedLLM([ticket("c1")]), registry, store).start("File a ticket", "run-2")
    except SimulatedCrash:
        pass
    registry.execute = real  # type: ignore[method-assign]
    print(f"tickets actually created before resume: {len(ledger.created)}")

    def next_reply(messages) -> str:
        note = next(m.content for m in messages if m.role == "tool")
        return "I could not confirm the ticket; verifying before retrying." if "UNKNOWN" in (note or "") else "?"

    state = agent(ScriptedLLM([next_reply]), registry, store).resume("run-2")
    record = state.tool_records["c1"]
    print(f"record: ok={record.ok} error_type={record.error_type}")
    print(f"tickets after resume (no blind retry): {len(ledger.created)} | final: {state.final_answer}")


def scenario_two_workers(store: FileCheckpointStore) -> None:
    banner("3. Two workers resume the same run: the second write is rejected")
    run = store.load("run-1")
    stale = run.model_copy(deep=True)
    run.seq += 1
    store.save(run)
    try:
        store.save(stale.model_copy(update={"seq": run.seq}))
    except CheckpointConflict as exc:
        print("rejected:", exc)


def scenario_migration() -> None:
    banner("4. An old (v1) checkpoint written by last month's code")
    old = {"schema_version": 1, "run_id": "legacy", "task": "t", "status": "done", "messages": [],
           "tool_outputs": {"c1": "TCK-007"}}
    new = migrate(old)
    print("status:", new["status"], "| schema_version:", new["schema_version"], "| tool_records:", list(new["tool_records"]))
    assert RunStatus(new["status"]) == RunStatus.COMPLETED


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        scenario_resume_after_crash(store)
        scenario_unknown_outcome(store)
        scenario_two_workers(store)
        scenario_migration()


if __name__ == "__main__":
    main()
```
<!-- code:end -->

<!-- code:begin path="code/day05/test_runner.py" -->
**`code/day05/test_runner.py`**

```python
"""Behavioural tests for DurableAgent, including real crash-and-resume scenarios."""

from __future__ import annotations

import pytest

from common.llm import LLMResponse, Message, ScriptedLLM, ToolCall, TransientLLMError
from common.tools import ToolRegistry
from day05.ops_tools import TicketLedger, build_registry
from day05.runner import DurableAgent, RunNotFound
from day05.state import IllegalTransition, RunStatus
from day05.store import FileCheckpointStore


class SimulatedCrash(Exception):
    """Stands in for SIGKILL / OOM / deploy: not an LLMError, so the runner must not catch it."""


def agent(llm, registry, store, **kw) -> DurableAgent:
    return DurableAgent(llm, registry, store, "You are the Ops Report Agent.", **kw)


def ticket(call_id: str, name: str = "create_ticket", **extra) -> Message:
    args = {"title": "Add index on orders", "severity": "SEV2", **extra}
    return ScriptedLLM.tool_call(name, call_id=call_id, **args)


def test_happy_path_checkpoints_every_step(tmp_path):
    ledger = TicketLedger()
    events: list[dict] = []
    llm = ScriptedLLM([ScriptedLLM.tool_call("get_incident", "c1", incident_id="INC-1042"), ticket("c2"), "Done: TCK-001"])
    state = agent(llm, build_registry(ledger), FileCheckpointStore(tmp_path), on_event=events.append).start("go", "run-a")
    assert state.status == RunStatus.COMPLETED and state.final_answer == "Done: TCK-001"
    assert len(ledger.created) == 1
    seqs = [e["seq"] for e in events if e["type"] == "checkpoint"]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # created, running, llm1, tool1, llm2, intent(write), tool2, llm3, completed
    assert state.seq == 9
    assert FileCheckpointStore(tmp_path).load("run-a").status == RunStatus.COMPLETED


def test_crash_between_steps_then_resume_does_not_repeat_write(tmp_path):
    """The core Day 5 test: kill the run after a write completed, resume in a 'new process'."""
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)
    crashing = ScriptedLLM([ticket("c1"), SimulatedCrash("worker killed")])
    with pytest.raises(SimulatedCrash):
        agent(crashing, registry, store).start("file a ticket", "run-b")
    assert len(ledger.created) == 1

    on_disk = store.load("run-b")
    assert on_disk.status == RunStatus.RUNNING  # a crash is invisible to the state machine
    assert on_disk.tool_records["c1"].status == "finished"

    # "New process": fresh agent, fresh LLM client, same store. Only the checkpoint survives.
    fresh = ScriptedLLM(["Ticket TCK-001 filed."])
    state = agent(fresh, registry, store).resume("run-b")
    assert state.status == RunStatus.COMPLETED
    assert len(ledger.created) == 1, "completed write must not run again"
    assert state.resume_count == 1
    sent = fresh.calls[0]["messages"]
    assert any(m.role == "tool" and m.content == "TCK-001" for m in sent), "model sees the recorded result"


def test_crash_after_llm_response_before_tools_runs_pending_tools_once(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)

    class CrashBeforeTools(ToolRegistry):
        pass

    # Crash inside the registry on the first call, before any tool executed.
    real = registry.execute
    state_flag = {"armed": True}

    def flaky_execute(call: ToolCall):
        if state_flag["armed"]:
            raise SimulatedCrash("killed before tool ran")
        return real(call)

    registry.execute = flaky_execute  # type: ignore[method-assign]
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([ticket("c1")]), registry, store).start("t", "run-c")
    assert ledger.created == []
    # The intent record was durably written before the crash, so the outcome is declared unknown.
    assert store.load("run-c").tool_records["c1"].status == "started"
    state_flag["armed"] = False
    state = agent(ScriptedLLM(["ok"]), registry, store).resume("run-c")
    assert state.status == RunStatus.COMPLETED
    assert ledger.created == [], "non-idempotent call with unknown outcome is not blindly re-run"
    assert state.tool_records["c1"].error_type == "outcome_unknown"


def test_crash_mid_tool_idempotent_tool_is_retried_safely(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)
    real = registry.execute

    def crash_after_effect(call: ToolCall):
        real(call)  # the side effect happens...
        raise SimulatedCrash("...then the worker dies before checkpointing")

    registry.execute = crash_after_effect  # type: ignore[method-assign]
    key = "key-inc-1042-ticket"
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([ticket("c1", "create_ticket_keyed", idempotency_key=key)]), registry, store).start("t", "run-d")
    assert len(ledger.created) == 1  # effect happened, checkpoint did not
    registry.execute = real  # type: ignore[method-assign]
    state = agent(ScriptedLLM(["done"]), registry, store).resume("run-d")
    assert state.status == RunStatus.COMPLETED
    assert len(ledger.created) == 1, "same idempotency key -> deduplicated on re-execution"
    assert state.tool_records["c1"].ok is True


def test_replay_uses_recorded_result_when_record_is_finished(tmp_path):
    """A finished record whose message was not appended is replayed from the ledger, not re-executed."""
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([ticket("c1"), SimulatedCrash("x")]), registry, store).start("t", "run-e")
    s = store.load("run-e")
    s.messages.pop()  # simulate a checkpoint taken after the record but before the message
    s.seq += 1
    store.save(s)
    events: list[dict] = []
    state = agent(ScriptedLLM(["ok"]), registry, store, on_event=events.append).resume("run-e")
    assert [e["type"] for e in events if e["type"].startswith("tool_")] == ["tool_replayed"]
    assert len(ledger.created) == 1 and state.status == RunStatus.COMPLETED


def test_llm_error_fails_durably_and_resume_retries(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    state = agent(ScriptedLLM([TransientLLMError("upstream 503")]), registry, store).start("t", "run-f")
    assert state.status == RunStatus.FAILED and "upstream 503" in state.error
    state = agent(ScriptedLLM(["recovered"]), registry, store).resume("run-f")
    assert state.status == RunStatus.COMPLETED and state.error is None


def test_step_budget_survives_resume(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    loop = [ScriptedLLM.tool_call("get_incident", f"c{i}", incident_id="INC-1042") for i in range(5)]
    state = agent(ScriptedLLM(loop), registry, store, max_steps=2).start("t", "run-g")
    assert state.status == RunStatus.FAILED and state.steps == 2
    assert "no final answer" in state.error


def test_terminal_runs_are_not_rerun_and_cancel_is_final(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    llm = ScriptedLLM(["hi"])
    a = agent(llm, registry, store)
    done = a.start("t", "run-h")
    again = a.resume("run-h")
    assert again.seq == done.seq and llm.remaining == 0
    with pytest.raises(IllegalTransition):
        a.cancel("run-h")
    with pytest.raises(ValueError):
        a.start("t", "run-h")
    with pytest.raises(RunNotFound):
        a.resume("nope")


def test_usage_accumulates_across_a_crash(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    resp = LLMResponse(message=ScriptedLLM.tool_call("get_incident", "c1", incident_id="INC-1042"),
                       usage={"prompt_tokens": 100, "completion_tokens": 10})
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([resp, SimulatedCrash("x")]), registry, store).start("t", "run-i")
    final = LLMResponse(message=Message(role="assistant", content="ok"), usage={"prompt_tokens": 120, "completion_tokens": 5})
    state = agent(ScriptedLLM([final]), registry, store).resume("run-i")
    assert state.usage.total_tokens == 235 and state.steps == 2
```
<!-- code:end -->

<!-- code:begin path="code/day05/test_pg_store.py" -->
**`code/day05/test_pg_store.py`**

```python
from __future__ import annotations

import pytest

from common.infra import postgres_available, temp_schema
from common.llm import ScriptedLLM
from day05.ops_tools import TicketLedger, build_registry
from day05.pg_store import PostgresCheckpointStore
from day05.runner import DurableAgent
from day05.state import RunState, RunStatus
from day05.store import CheckpointConflict

pytestmark = [
    pytest.mark.infra,
    pytest.mark.skipif(not postgres_available(), reason="Postgres not reachable (set DATABASE_URL)"),
]


@pytest.fixture()
def store():
    with temp_schema() as (url, schema):
        s = PostgresCheckpointStore(url, schema)
        s.ensure_schema()
        yield s


class Crash(Exception):
    pass


def test_roundtrip_history_and_time_travel(store):
    s = RunState(run_id="pg-1", task="t", seq=1)
    store.save(s)
    s.seq = 2
    s.transition(RunStatus.RUNNING)
    store.save(s)
    assert store.load("pg-1").status == RunStatus.RUNNING
    assert store.load("pg-1", seq=1).status == RunStatus.CREATED
    assert store.history("pg-1") == [(1, "created"), (2, "running")]
    assert store.load("none") is None and store.list_runs() == ["pg-1"]


def test_two_writers_same_seq_exactly_one_wins(store):
    a = RunState(run_id="pg-2", task="t", seq=1)
    store.save(a)
    w1 = RunState(run_id="pg-2", task="worker-1", seq=2)
    w2 = RunState(run_id="pg-2", task="worker-2", seq=2)
    store.save(w1)
    with pytest.raises(CheckpointConflict):
        store.save(w2)
    assert store.load("pg-2").task == "worker-1"


def test_prune_keeps_newest(store):
    for i in range(1, 8):
        store.save(RunState(run_id="pg-3", task="t", seq=i))
    assert store.prune("pg-3", keep_last=3) == 4
    assert [s for s, _ in store.history("pg-3")] == [5, 6, 7]
    with pytest.raises(ValueError):
        store.prune("pg-3", keep_last=0)


def test_bad_schema_name_rejected():
    with pytest.raises(ValueError):
        PostgresCheckpointStore("postgresql://x", 'x"; DROP SCHEMA public; --')


def test_crash_and_resume_through_postgres(store):
    ledger = TicketLedger()
    registry = build_registry(ledger)
    first = ScriptedLLM([ScriptedLLM.tool_call("create_ticket", "c1", title="Add index", severity="SEV2"), Crash("kill")])
    with pytest.raises(Crash):
        DurableAgent(first, registry, store, "sys").start("t", "pg-run")
    state = DurableAgent(ScriptedLLM(["TCK-001 filed"]), registry, store, "sys").resume("pg-run")
    assert state.status == RunStatus.COMPLETED and len(ledger.created) == 1
    assert store.history("pg-run")[-1][1] == "completed"
```
<!-- code:end -->

### Real demo output

```text
=== 1. Crash after a write finished, resume in a 'new process' ===
process died: worker killed
on disk: status=running seq=7 steps=2 tickets_created=1
tool ledger: {'c1': ('get_incident', 'finished'), 'c2': ('create_ticket', 'finished')}
resumed: status=completed resume_count=1 tickets_created=1
model saw recorded tool results, not re-executions: ['{"id": "INC-1042", "title": "Checkout AP', 'TCK-001']

=== 2. Crash INSIDE a non-idempotent tool: the honest answer is 'unknown' ===
tickets actually created before resume: 1
record: ok=False error_type=outcome_unknown
tickets after resume (no blind retry): 1 | final: I could not confirm the ticket; verifying before retrying.

=== 3. Two workers resume the same run: the second write is rejected ===
rejected: run run-1: stored seq 10 >= new seq 10

=== 4. An old (v1) checkpoint written by last month's code ===
status: completed | schema_version: 2 | tool_records: ['c1']
```

Note the first scenario: the run on disk says `running` even though the process is dead. Scenario 2 is the important one: the ticket *was* created, the checkpoint was not written, and the runner refuses to retry blindly. The "model" here is scripted, so the reply is predetermined. What the demo verifies is the runner's behaviour: the unknown-outcome message reaches the model and no second ticket is created.

## 8. Implementation decisions explained

| Decision | Alternative | Why this one | When to choose differently |
|---|---|---|---|
| Snapshot of full state per checkpoint | Event log with replay | Simple, resume is one read | Very long runs where snapshot size grows; use an event log plus periodic snapshots |
| Checkpoint after LLM response and tool result | Only at run end | Resume loses at most one in-flight operation | Tiny cheap runs where restarting is fine |
| Write-ahead intent only for non-idempotent tools | For every tool | Avoids a write per read-only call | If you want a complete audit of attempted calls, record intent for all |
| `outcome_unknown` instead of retry | Always retry | Duplicating a ticket or payment is worse than asking | Tools that are truly idempotent are retried automatically |
| `seq` and primary key | Row locks | Simple, no lock held across a model call | Long-held leases for exclusive ownership (below) |
| Append-only Postgres history | Upsert latest | Time-travel debugging and audit | Storage-constrained setups; use `prune` |
| Migrations before Pydantic validation | Optional fields everywhere | Explicit, testable, refuses unknown futures | Never skip versioning for durable state |
| Crash exceptions are not caught by the runner | Catch everything and mark failed | A killed process cannot mark itself failed; catching masks bugs | Wrap at the worker boundary, not in the runner |

## 9. Failure scenarios and edge cases

- **Crash between steps.** Latest checkpoint is consistent; resume continues. Tested in `test_crash_between_steps_then_resume_does_not_repeat_write`.
- **Crash after the LLM response, before its tools ran.** `pending_calls()` returns the unanswered calls and they execute exactly once.
- **Crash mid-tool, idempotent tool.** Re-executed; deduplicated downstream by the key.
- **Crash mid-tool, non-idempotent tool.** `outcome_unknown`, surfaced to the model. Not resolvable inside the runner; see Day 23.
- **Torn or corrupt snapshot.** Raises `CorruptCheckpoint`; never silently ignored. The atomic-rename write makes torn files unlikely, but disks and humans corrupt files.
- **Two workers on one run.** Conflict error for the loser. Not handled: *which* worker should have owned it. That needs a lease.
- **Provider error.** `LLMError` becomes a durable `failed` that `resume()` can retry; the step budget survives resume (`test_step_budget_survives_resume`).
- **Checkpoint from newer code.** Refused (`UnsupportedSchema`). A rollback of a deploy therefore cannot silently misread new data.
- **Large state.** Messages grow with every step. Snapshots get slower and larger; Day 7 covers compaction, and `prune` bounds history rows.
- **Resume after the world changed.** A paused run resumed days later may act on stale facts. Nothing here detects that; long-lived runs need an explicit freshness policy.

## 10. Production considerations

**Security.** Checkpoints contain the full conversation: prompts, tool outputs, possibly customer data and secrets that leaked into tool results. Treat the store as sensitive: encrypt at rest, restrict access, set retention, and redact before persisting where you can. Validate ids that become file paths (done) and never load checkpoints from untrusted sources, since the state drives what tools are called on resume.

**Reliability.** Add a lease or heartbeat so a worker can tell a live `running` run from an orphaned one. Without it, "resume everything that says running" will start a second worker on a run that is merely slow. The sequence-number check is your last line of defence, not your ownership protocol.

**Observability.** Emit an event per checkpoint with run id, `seq` and status (the runner does). Track resume counts per run: a run that resumes repeatedly is crashing at the same point.

**Scalability.** One row per checkpoint multiplies write load by steps per run. Batch where correctness allows, prune old rows, and keep large tool outputs out of the snapshot (store a reference).

**Cost.** Checkpointing after the LLM response is what stops a restart from paying for the same completion twice. The extra writes are small next to a model call; measure before optimising them away.

## 11. Hands-on exercises

1. **Kill it for real.** Run a loop that sleeps inside a tool, `kill -9` the process, then resume from the file store in a new process. *Acceptance:* the resumed run completes and the ledger shows the original calls as replayed, not re-executed.
2. **Add a state.** Add a `waiting_approval` status (preview of Day 17) with legal transitions in and out. *Acceptance:* tests for the new edges, and that illegal edges still raise.
3. **Write a migration.** Add `schema_version` 3 with a new required field, write `_v2_to_v3`, and keep the old v1 migration test passing. *Acceptance:* a v1 document migrates through both steps.
4. **Provoke a conflict.** Start two threads resuming the same Postgres run. *Acceptance:* exactly one proceeds; the other gets `CheckpointConflict`, and the final history has no duplicate `seq`.

## 12. Challenge task

Add a **lease**: a `claimed_by` worker id and `lease_expires_at` on the run, renewed by a heartbeat thread while the worker is alive. A worker may resume a `running` run only if its lease has expired. *Acceptance criteria:* two workers racing for an expired lease result in exactly one owner; a live worker's run is never stolen while heartbeating; a test simulates a stalled worker (heartbeat stops) and shows takeover and then a correct resume; the stale worker's late checkpoint write is rejected. *Hint:* do the claim in a single conditional `UPDATE ... WHERE lease_expires_at < now() RETURNING`, and think about clock skew between workers.

## 13. Knowledge check

1. After `kill -9`, what status does the run have on disk, and why?
   <details><summary>Answer</summary>`running`. The dying process cannot write a terminal status; a crash is invisible to the state machine, so resume must start from `running`.</details>
2. Why save an intent record before running a non-idempotent tool?
   <details><summary>Answer</summary>Without it, a crash mid-call leaves no trace, so on resume the runner cannot distinguish "never ran" from "ran, result lost". With it, the runner can report `outcome_unknown` instead of duplicating the effect.</details>
3. A finished tool record exists for call `c2`. What does resume do with `c2`?
   <details><summary>Answer</summary>It replays the recorded result into the conversation and does not call the tool.</details>
4. Why can't the runner achieve exactly-once for a ticket write by itself?
   <details><summary>Answer</summary>A crash between the external side effect and the checkpoint is undetectable from inside the process. Only the downstream system deduplicating (idempotency keys) can close that gap.</details>
5. Two workers resume the same run on Postgres. What happens?
   <details><summary>Answer</summary>Both try to insert the next `seq`; the primary key `(run_id, seq)` admits one, and the other gets `CheckpointConflict`.</details>
6. Why run migrations before Pydantic validation?
   <details><summary>Answer</summary>Old documents would fail validation against the new model. A pure upgrade chain brings them to the current shape first, and unknown future versions are refused.</details>
7. Why does the file store need a caveat about multiple processes?
   <details><summary>Answer</summary>Its conflict check is read-then-write, which two processes can interleave. Postgres enforces it atomically with the primary key.</details>

## 14. References

- [Temporal: Durable execution](https://docs.temporal.io/): the general idea behind resumable workflows, and why side effects are the hard part.
- [PostgreSQL: Concurrency control](https://www.postgresql.org/docs/16/mvcc.html): background for the optimistic-concurrency approach using primary keys.
- [Pydantic: Serialization](https://docs.pydantic.dev/latest/concepts/serialization/): how `model_dump_json` and validation round-trip state.
- [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace): the atomic rename behind the file store.
- [Stripe: Idempotent requests](https://docs.stripe.com/api/idempotent_requests): the downstream half of exactly-once, previewing Day 23.

## 15. After this lesson you should be able to…

- [ ] Explain why a crashed run still says `running`, and design for that.
- [ ] Implement a lifecycle state machine with legal transitions.
- [ ] Checkpoint after each LLM response and tool result, and resume in a new process.
- [ ] Show with a test that completed side effects are not repeated on resume.
- [ ] Explain precisely what stays ambiguous after a mid-tool crash, and how idempotency keys resolve it.
- [ ] Migrate an old checkpoint schema safely and refuse a newer one.

Next: [Day 6](day-06-short-term-and-long-term-memory.md) separates what the agent holds in the working conversation from what it remembers across runs.
