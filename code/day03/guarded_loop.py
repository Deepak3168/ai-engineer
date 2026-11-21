"""`common.loop.AgentLoop` plus the guards a production loop needs.

`AgentLoop` stops for three reasons: the model answers, `max_steps`, or a token budget. A model that is
stuck inside those limits - same search again and again, or ten different searches that all return nothing -
burns every step before anyone notices. This loop adds:

  * repeat-call detection   identical (tool, arguments) beyond N executions is blocked, not run
  * per-tool call caps      e.g. at most 3 `search_runbooks` per run
  * no-progress detection   K consecutive steps that produced no *new* successful information stop the run
  * an abort threshold      after M blocked attempts the nudges are clearly not working: stop
  * an optional explicit final tool, and an optional tool-free wrap-up call for a partial answer

Invariant kept on every exit path: each assistant `tool_calls` id has exactly one tool message, so the
returned history is always a valid conversation (resumable on Day 5, replayable on Day 22).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any, Literal

from day02.execution import execute_batch
from pydantic import BaseModel, Field

from common.llm import LLMClient, LLMError, Message, ToolCall, system, tool_result, user
from common.loop import RunResult, Step
from common.tools import SideEffect, ToolRegistry, ToolResult

GuardedStatus = Literal[
    "completed", "max_steps", "budget_exceeded", "llm_error", "loop_detected", "no_progress"
]
EventHook = Callable[[dict[str, Any]], None]


class GuardConfig(BaseModel):
    max_steps: int = Field(default=10, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_identical_calls: int = Field(default=2, ge=1, description="Executions allowed per identical (tool, args)")
    max_calls_per_tool: dict[str, int] = Field(default_factory=dict)
    max_blocked: int = Field(default=3, ge=1, description="Blocked attempts tolerated before the run is aborted")
    no_progress_steps: int = Field(default=3, ge=1)
    allow_side_effects: tuple[SideEffect, ...] = ("read",)
    final_tool: str | None = None
    wrap_up: bool = False


class GuardedRunResult(RunResult):
    status: GuardedStatus  # type: ignore[assignment]  # widened: adds loop_detected / no_progress
    blocked: int = 0
    executed_by_tool: dict[str, int] = Field(default_factory=dict)
    partial_answer: str | None = None


def _blocked(reason: str, message: str) -> ToolResult:
    return ToolResult(ok=False, error_type=reason, content=f"Error: {message}")


def _call_key(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class GuardedLoop:
    def __init__(self, client: LLMClient, registry: ToolRegistry, system_prompt: str,
                 config: GuardConfig | None = None, on_event: EventHook | None = None) -> None:
        self.client, self.registry, self.system_prompt = client, registry, system_prompt
        self.config = config or GuardConfig()
        self._emit = on_event or (lambda _e: None)

    # ------------------------------------------------------------------ guards
    def _decide(self, call: ToolCall, attempts: Counter[str], executed: Counter[str]) -> ToolResult | None:
        """Return a synthetic result if the call must NOT run, else None (and count it as scheduled)."""
        cfg = self.config
        key = _call_key(call)
        attempts[key] += 1
        if attempts[key] > cfg.max_identical_calls:
            return _blocked("repeat_blocked", f"you already called '{call.name}' with exactly these arguments "
                            f"{cfg.max_identical_calls} times and the result will not change. Change the arguments, "
                            "use a different tool, or give your final answer with what you have.")
        cap = cfg.max_calls_per_tool.get(call.name)
        if cap is not None and executed[call.name] >= cap:
            return _blocked("tool_cap", f"call budget for '{call.name}' ({cap}) is used up. "
                            "Work with the evidence you have and answer.")
        executed[call.name] += 1
        return None

    def _run_calls(self, calls: Sequence[ToolCall], attempts: Counter[str], executed: Counter[str]
                   ) -> tuple[list[ToolResult], list[bool]]:
        results: list[ToolResult | None] = [None] * len(calls)
        to_run: list[int] = []
        for i, call in enumerate(calls):
            results[i] = self._decide(call, attempts, executed)
            if results[i] is None:
                to_run.append(i)
        if to_run:
            ran = execute_batch(self.registry, [calls[i] for i in to_run], allow=self.config.allow_side_effects)
            for i, res in zip(to_run, ran, strict=True):
                results[i] = res
        was_blocked = [i not in to_run for i in range(len(calls))]
        return [r for r in results if r is not None], was_blocked

    # --------------------------------------------------------------------- run
    def run(self, task: str, history: Sequence[Message] = ()) -> GuardedRunResult:
        cfg = self.config
        messages: list[Message] = [system(self.system_prompt), *history, user(task)]
        result = GuardedRunResult(status="max_steps")
        result.messages = messages  # Pydantic copies list fields at construction; share the live list instead
        schemas = self.registry.schemas() or None
        attempts: Counter[str] = Counter()
        executed: Counter[str] = Counter()
        seen: set[str] = set()
        stale_steps = 0

        for index in range(cfg.max_steps):
            try:
                resp = self.client.chat(messages, tools=schemas)
            except LLMError as exc:
                result.status, result.error = "llm_error", f"{type(exc).__name__}: {exc}"
                return result
            result.usage.prompt_tokens += resp.usage.prompt_tokens
            result.usage.completion_tokens += resp.usage.completion_tokens
            step = Step(index=index, assistant=resp.message, usage=resp.usage)
            messages.append(resp.message)
            result.steps.append(step)
            self._emit({"type": "llm_response", "step": index, "tool_calls": [c.name for c in resp.tool_calls]})

            if not resp.tool_calls:
                result.status, result.final_answer = "completed", resp.content
                return result

            ids = [c.id for c in resp.tool_calls]
            if len(set(ids)) != len(ids):  # cannot answer ambiguous ids; history would be invalid
                result.status, result.error = "llm_error", f"protocol violation: duplicate tool call ids {ids}"
                return result

            outcomes, blocked_flags = self._run_calls(resp.tool_calls, attempts, executed)
            made_progress = False
            for call, outcome, was_blocked in zip(resp.tool_calls, outcomes, blocked_flags, strict=True):
                step.tool_results.append((call, outcome))
                messages.append(tool_result(call, outcome.content))
                result.blocked += was_blocked
                digest = _digest(outcome.content)
                if outcome.ok and digest not in seen:
                    made_progress = True
                    seen.add(digest)
                self._emit({"type": "tool_result", "step": index, "tool": call.name, "ok": outcome.ok,
                            "error_type": outcome.error_type, "blocked": was_blocked})
                if cfg.final_tool and call.name == cfg.final_tool and outcome.ok:
                    result.status, result.final_answer = "completed", str(call.arguments.get("answer", ""))
            result.executed_by_tool = dict(executed)
            if result.status == "completed":
                return result

            stale_steps = 0 if made_progress else stale_steps + 1
            reason: tuple[GuardedStatus, str] | None = None
            if result.blocked >= cfg.max_blocked:
                reason = ("loop_detected", f"{result.blocked} blocked repeat/cap attempts: the model is not changing course")
            elif stale_steps >= cfg.no_progress_steps:
                reason = ("no_progress", f"{stale_steps} consecutive steps produced no new successful result")
            elif cfg.max_total_tokens and result.usage.total_tokens >= cfg.max_total_tokens:
                reason = ("budget_exceeded", f"token budget {cfg.max_total_tokens} reached ({result.usage.total_tokens} used)")
            if reason:
                result.status, result.error = reason
                self._emit({"type": "guard_stop", "step": index, "status": reason[0], "detail": reason[1]})
                return self._wrap_up(result, messages)

        result.error = f"no final answer within {cfg.max_steps} steps"
        return self._wrap_up(result, messages)

    def _wrap_up(self, result: GuardedRunResult, messages: list[Message]) -> GuardedRunResult:
        """One tool-free call asking for what is known so far. Costs a request; off by default."""
        if not self.config.wrap_up:
            return result
        try:
            resp = self.client.chat([*messages, user(
                "Stop calling tools. In a few sentences, state what you established, cite ids, and say what remains unknown."
            )])
        except LLMError as exc:
            result.error = f"{result.error}; wrap-up failed: {type(exc).__name__}: {exc}"
            return result
        result.usage.prompt_tokens += resp.usage.prompt_tokens
        result.usage.completion_tokens += resp.usage.completion_tokens
        result.partial_answer = resp.content
        return result
