"""Tool registry: schema generation, argument validation, safe execution.

The model never executes anything. It *requests* a call; this module is the
trust boundary that decides whether and how that request runs (Days 2, 18, 19).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from common.llm import ToolCall

SideEffect = Literal["read", "write", "external"]
MAX_OUTPUT_CHARS = 8_000


class ToolResult(BaseModel):
    """What goes back to the model (content) plus metadata for us (everything else)."""

    ok: bool
    content: str
    error_type: str | None = None  # unknown_tool | bad_arguments | tool_error | timeout | denied
    duration_s: float = 0.0
    truncated: bool = False


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        fn: Callable[..., Any],
        *,
        side_effect: SideEffect = "read",
        idempotent: bool | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = args_model
        self.fn = fn
        self.side_effect = side_effect
        # Reads are idempotent by default; writes/external calls must opt in (Day 23).
        self.idempotent = (side_effect == "read") if idempotent is None else idempotent

    def openai_schema(self) -> dict[str, Any]:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, default=str, ensure_ascii=False)


class ToolRegistry:
    def __init__(self, max_output_chars: int = MAX_OUTPUT_CHARS) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars

    # -- registration -------------------------------------------------------
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        *,
        side_effect: SideEffect = "read",
        idempotent: bool | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form: the function receives the validated args model's fields as kwargs."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name, description, args_model, fn, side_effect=side_effect, idempotent=idempotent))
            return fn

        return deco

    # -- introspection ------------------------------------------------------
    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values() if only is None else [self._tools[n] for n in only if n in self._tools]
        return [t.openai_schema() for t in tools]

    # -- execution ----------------------------------------------------------
    def execute(self, call: ToolCall) -> ToolResult:
        """Validate then run. Never raises: failures become results the model can react to."""
        started = time.perf_counter()

        def done(ok: bool, content: str, error_type: str | None = None) -> ToolResult:
            truncated = len(content) > self.max_output_chars
            if truncated:
                content = content[: self.max_output_chars] + f"\n...[truncated {len(content) - self.max_output_chars} chars]"
            return ToolResult(ok=ok, content=content, error_type=error_type,
                              duration_s=time.perf_counter() - started, truncated=truncated)

        tool = self._tools.get(call.name)
        if tool is None:
            return done(False, f"Error: unknown tool '{call.name}'. Available tools: {', '.join(self.names()) or 'none'}.", "unknown_tool")
        if call.arguments_error:
            return done(False, f"Error: could not parse arguments for '{call.name}': {call.arguments_error}", "bad_arguments")
        try:
            args = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()[:6])
            return done(False, f"Error: invalid arguments for '{call.name}': {detail}", "bad_arguments")
        try:
            value = tool.fn(**args.model_dump())
        except Exception as exc:  # tool bugs and domain errors alike become observations
            return done(False, f"Error: {type(exc).__name__}: {exc}", "tool_error")
        return done(True, _to_text(value))
