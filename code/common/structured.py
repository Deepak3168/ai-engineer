"""Structured outputs: ask for JSON, validate with Pydantic, repair on failure.

Provider-portable on purpose. Native "JSON schema mode" differs between
OpenAI, NVIDIA NIM, vLLM, Ollama... so we rely on three things that work
everywhere: a schema in the prompt, tolerant extraction, and a validation
feedback loop. Native modes are an optimisation layered on top (Day 1).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from common.llm import LLMClient, Message, PermanentLLMError, Usage, assistant, system, user

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class StructuredOutputError(PermanentLLMError):
    """The model never produced output that validates, even after repairs."""

    def __init__(self, msg: str, attempts: list[str]):
        super().__init__(msg)
        self.attempts = attempts


def extract_json(text: str) -> str:
    """Pull the most likely JSON object out of free text.

    Handles: bare JSON, ```json fences, and prose before/after the object.
    """
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start == -1:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def schema_instructions(model_cls: type[BaseModel]) -> str:
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return (
        "Respond with a single JSON object and nothing else. "
        "It must validate against this JSON Schema:\n" + schema
    )


def chat_structured(
    client: LLMClient,
    messages: Sequence[Message],
    model_cls: type[T],
    *,
    max_repairs: int = 2,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[T, Usage]:
    """Return (validated_instance, total_usage). Raises StructuredOutputError on failure."""
    convo: list[Message] = [system(schema_instructions(model_cls)), *messages]
    attempts: list[str] = []
    total = Usage()
    for attempt in range(max_repairs + 1):
        resp = client.chat(convo, temperature=temperature, max_tokens=max_tokens)
        total.prompt_tokens += resp.usage.prompt_tokens
        total.completion_tokens += resp.usage.completion_tokens
        raw = resp.content
        attempts.append(raw)
        try:
            return model_cls.model_validate_json(extract_json(raw)), total
        except (ValidationError, ValueError) as exc:
            if attempt == max_repairs:
                break
            problem = _summarise_error(exc)
            convo += [
                assistant(raw),
                user(
                    "That output was invalid:\n"
                    f"{problem}\n"
                    "Return only the corrected JSON object."
                ),
            ]
    raise StructuredOutputError(
        f"no valid {model_cls.__name__} after {max_repairs + 1} attempts", attempts
    )


def _summarise_error(exc: Exception) -> str:
    """Compact, model-readable validation feedback (no stack traces)."""
    if isinstance(exc, ValidationError):
        return "\n".join(
            f"- field '{'.'.join(str(p) for p in e['loc']) or '<root>'}': {e['msg']}"
            for e in exc.errors()[:8]
        )
    return f"- {exc}"
