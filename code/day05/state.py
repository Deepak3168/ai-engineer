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
