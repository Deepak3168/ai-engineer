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

