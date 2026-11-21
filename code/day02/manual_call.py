"""One manual tool-calling round trip, with no loop, so the raw protocol is visible.

    request 1: messages + tools           ->  assistant message with tool_calls (no text)
    we execute each call ourselves
    request 2: messages + assistant msg + one tool message per call  ->  assistant text

`one_round` returns every payload it sent so a lesson (or a test) can print the exact wire shapes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from common.llm import LLMClient, Message, system, user
from common.tools import ToolRegistry, ToolResult
from day02.execution import execute_batch, results_to_messages

SYSTEM_PROMPT = (
    "You are the Ops Report Agent. Use the tools to look facts up; never invent incident details. "
    "If a tool returns an error, read it and correct your call or tell the user."
)


class ManualTrace(BaseModel):
    request_1: dict[str, Any]
    assistant_1: dict[str, Any]
    tool_messages: list[dict[str, Any]] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)
    request_2: dict[str, Any] | None = None
    final_text: str | None = None
    needs_another_round: bool = False  # the model asked for more tools: exactly why Day 3 needs a loop


def _payload(messages: list[Message], registry: ToolRegistry) -> dict[str, Any]:
    return {"messages": [m.to_openai() for m in messages], "tools": registry.schemas()}


def one_round(client: LLMClient, registry: ToolRegistry, question: str) -> ManualTrace:
    messages: list[Message] = [system(SYSTEM_PROMPT), user(question)]
    resp1 = client.chat(messages, tools=registry.schemas())
    trace = ManualTrace(request_1=_payload(messages, registry), assistant_1=resp1.message.to_openai())

    if not resp1.tool_calls:  # the model answered directly; nothing to execute
        trace.final_text = resp1.content
        return trace

    trace.results = execute_batch(registry, resp1.tool_calls)
    tool_messages = results_to_messages(resp1.tool_calls, trace.results)
    trace.tool_messages = [m.to_openai() for m in tool_messages]

    # The assistant message that *contained* the tool_calls must be replayed verbatim before the results.
    messages = [*messages, resp1.message, *tool_messages]
    trace.request_2 = _payload(messages, registry)
    resp2 = client.chat(messages, tools=registry.schemas())
    trace.final_text = resp2.content or None
    trace.needs_another_round = bool(resp2.tool_calls)
    return trace
