"""The baseline agent loop: model -> tool calls -> observations -> model ... -> answer.

This is deliberately small. Day 3 explains it, Days 4-7 and 13 extend it, and
the capstone replaces it with a full runtime. Everything an agent "is" fits here:

    while not done:
        response = llm(messages, tools)
        if no tool calls: done
        for call in tool calls: messages += execute(call)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from common.llm import LLMClient, LLMError, Message, ToolCall, Usage, system, tool_result, user
from common.tools import ToolRegistry, ToolResult

Status = Literal["completed", "max_steps", "budget_exceeded", "llm_error"]
EventHook = Callable[[dict[str, Any]], None]


class Step(BaseModel):
    index: int
    assistant: Message
    tool_results: list[tuple[ToolCall, ToolResult]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class RunResult(BaseModel):
    status: Status
    final_answer: str | None = None
    messages: list[Message] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.tool_results) for s in self.steps)


class AgentLoop:
    def __init__(
        self,
        client: LLMClient,
        registry: ToolRegistry,
        system_prompt: str,
        *,
        max_steps: int = 10,
        max_total_tokens: int | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_total_tokens = max_total_tokens
        self._emit = on_event or (lambda _e: None)

    def run(self, task: str, history: Sequence[Message] = ()) -> RunResult:
        messages: list[Message] = [system(self.system_prompt), *history, user(task)]
        result = RunResult(status="max_steps", messages=messages)
        schemas = self.registry.schemas() or None

        for index in range(self.max_steps):
            try:
                resp = self.client.chat(messages, tools=schemas)
            except LLMError as exc:
                result.status, result.error = "llm_error", f"{type(exc).__name__}: {exc}"
                self._emit({"type": "llm_error", "step": index, "error": result.error})
                return result

            result.usage.prompt_tokens += resp.usage.prompt_tokens
            result.usage.completion_tokens += resp.usage.completion_tokens
            step = Step(index=index, assistant=resp.message, usage=resp.usage)
            messages.append(resp.message)
            result.steps.append(step)
            self._emit({"type": "llm_response", "step": index, "tool_calls": [c.name for c in resp.tool_calls],
                        "tokens": resp.usage.total_tokens})

            if not resp.tool_calls:
                result.status, result.final_answer = "completed", resp.content
                return result

            for call in resp.tool_calls:
                outcome = self.registry.execute(call)
                step.tool_results.append((call, outcome))
                messages.append(tool_result(call, outcome.content))
                self._emit({"type": "tool_result", "step": index, "tool": call.name, "ok": outcome.ok,
                            "error_type": outcome.error_type, "duration_s": outcome.duration_s})

            if self.max_total_tokens and result.usage.total_tokens >= self.max_total_tokens:
                result.status = "budget_exceeded"
                result.error = f"token budget {self.max_total_tokens} reached ({result.usage.total_tokens} used)"
                return result

        result.error = f"no final answer within {self.max_steps} steps"
        return result
