"""ReAct with a plain-text protocol, for models (or endpoints) without native tool calling.

Yao et al. (2022) interleave reasoning and acting as text:

    Thought: <reasoning>
    Action: <tool name>
    Action Input: <JSON object>
    Observation: <written by the HARNESS, never by the model>
    ... repeat ...
    Final Answer: <answer>

The model only ever produces text. Everything native tool calling gives you for free - a separate
channel for calls, JSON already parsed, call ids - we must rebuild: a parser, a repair path for
malformed output, and defence against the model *writing the Observation itself*.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from common.llm import LLMClient, LLMError, Message, ToolCall, assistant, system, user
from common.loop import RunResult, Step
from common.structured import extract_json
from common.tools import ToolRegistry

ReActStatus = Literal["completed", "max_steps", "budget_exceeded", "llm_error", "format_error"]

_OBSERVATION = re.compile(r"^\s*Observation\s*:", re.IGNORECASE | re.MULTILINE)
_FINAL = re.compile(r"^\s*Final Answer\s*:\s*(.*)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_ACTION = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_INPUT = re.compile(r"^\s*Action Input\s*:\s*(.*)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
_NEXT_SECTION = re.compile(r"^\s*(Thought|Action|Observation|Final Answer)\s*:", re.IGNORECASE | re.MULTILINE)


class ParsedStep(BaseModel):
    kind: Literal["action", "final", "invalid"]
    clean_text: str  # the model's text with any self-written Observation (and everything after) removed
    thought: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = {}
    arguments_error: str | None = None
    answer: str | None = None
    error: str | None = None


def parse_react(text: str) -> ParsedStep:
    """Parse one model turn. Never raises; malformed output becomes kind="invalid" with a reason."""
    cut = _OBSERVATION.search(text)
    clean = (text[: cut.start()] if cut else text).strip()  # the model must not author observations
    final, action = _FINAL.search(clean), _ACTION.search(clean)

    first = min((m.start() for m in (final, action) if m), default=len(clean))
    thought = re.sub(r"^\s*Thought\s*:\s*", "", clean[:first], flags=re.IGNORECASE).strip() or None

    if final and action:
        return ParsedStep(kind="invalid", clean_text=clean, thought=thought,
                          error="output contains both an Action and a Final Answer; produce exactly one")
    if final:
        answer = final.group(1).strip()
        if not answer:
            return ParsedStep(kind="invalid", clean_text=clean, thought=thought, error="Final Answer is empty")
        return ParsedStep(kind="final", clean_text=clean, thought=thought, answer=answer)
    if not action:
        return ParsedStep(kind="invalid", clean_text=clean, thought=thought,
                          error="no 'Action:' or 'Final Answer:' line found")

    tool = action.group(1).strip().strip("`'\"")
    args: dict[str, Any] = {}
    args_error: str | None = None
    after = clean[action.end():]
    inp = _INPUT.search(after)
    if inp:
        body = inp.group(1)
        nxt = _NEXT_SECTION.search(body)  # stop at a following Thought/Action line
        raw = (body[: nxt.start()] if nxt else body).strip()
        if raw:
            try:
                parsed = json.loads(extract_json(raw))
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    args_error = f"Action Input must be a JSON object, got {type(parsed).__name__}"
            except json.JSONDecodeError as exc:
                args_error = f"Action Input is not valid JSON: {exc.msg} (pos {exc.pos})"
    return ParsedStep(kind="action", clean_text=clean, thought=thought, tool=tool,
                      arguments=args, arguments_error=args_error)


def _arg_summary(registry: ToolRegistry, name: str) -> str:
    schema = registry.get(name).args_model.model_json_schema()  # type: ignore[union-attr]
    required = set(schema.get("required", []))
    parts = []
    for key, prop in schema.get("properties", {}).items():
        kind = "|".join(map(str, prop["enum"])) if "enum" in prop else prop.get("type", "any")
        parts.append(f'"{key}": {kind}{"" if key in required else " (optional)"}')
    return "{" + ", ".join(parts) + "}"


def build_react_prompt(registry: ToolRegistry, role: str = "You are the Ops Report Agent.") -> str:
    tools = "\n".join(f"- {n}: {registry.get(n).description}\n  Action Input: {_arg_summary(registry, n)}"  # type: ignore[union-attr]
                      for n in registry.names())
    return (
        f"{role} Answer the user's request using these tools:\n{tools}\n\n"
        "Use EXACTLY this format, one step at a time:\n"
        "Thought: your reasoning about what to do next\n"
        "Action: one tool name from the list\n"
        "Action Input: a single JSON object\n"
        "Then STOP and wait. The system replies with a line starting 'Observation:'. Never write an Observation yourself.\n"
        "When you have enough evidence, reply with:\n"
        "Thought: I can answer now\n"
        "Final Answer: your answer, citing incident and runbook ids"
    )


class ReActResult(RunResult):
    status: ReActStatus  # type: ignore[assignment]  # widened: adds "format_error"
    format_errors: int = 0


class ReActTextLoop:
    def __init__(self, client: LLMClient, registry: ToolRegistry, *, max_steps: int = 8,
                 max_format_errors: int = 2, max_total_tokens: int | None = None) -> None:
        self.client, self.registry = client, registry
        self.max_steps, self.max_format_errors, self.max_total_tokens = max_steps, max_format_errors, max_total_tokens

    def run(self, task: str, history: Sequence[Message] = ()) -> ReActResult:
        messages: list[Message] = [system(build_react_prompt(self.registry)), *history, user(task)]
        result = ReActResult(status="max_steps")
        result.messages = messages  # Pydantic copies list fields at construction; share the live list instead
        for index in range(self.max_steps):
            try:
                resp = self.client.chat(messages)  # no `tools=`: this path works without native tool calling
            except LLMError as exc:
                result.status, result.error = "llm_error", f"{type(exc).__name__}: {exc}"
                return result
            result.usage.prompt_tokens += resp.usage.prompt_tokens
            result.usage.completion_tokens += resp.usage.completion_tokens

            parsed = parse_react(resp.content)
            # History keeps only the cleaned text, so an invented Observation never becomes "evidence".
            turn = assistant(parsed.clean_text or "(empty reply)")
            step = Step(index=index, assistant=turn, usage=resp.usage)
            result.steps.append(step)
            messages.append(turn)

            if parsed.kind == "final":
                result.status, result.final_answer = "completed", parsed.answer
                return result
            if parsed.kind == "invalid":
                result.format_errors += 1
                if result.format_errors > self.max_format_errors:
                    result.status, result.error = "format_error", f"unparseable output: {parsed.error}"
                    return result
                messages.append(user(f"Observation: format error - {parsed.error}. Reply using the exact format."))
                continue

            call = ToolCall(id=f"react-{index}", name=parsed.tool or "", arguments=parsed.arguments,
                            arguments_error=parsed.arguments_error)
            outcome = self.registry.execute(call)
            step.tool_results.append((call, outcome))
            messages.append(user(f"Observation: {outcome.content}"))

            if self.max_total_tokens and result.usage.total_tokens >= self.max_total_tokens:
                result.status = "budget_exceeded"
                result.error = f"token budget {self.max_total_tokens} reached ({result.usage.total_tokens} used)"
                return result
        result.error = f"no final answer within {self.max_steps} steps"
        return result
