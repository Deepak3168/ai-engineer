"""A thin, provider-agnostic LLM layer for any OpenAI-compatible endpoint.

Design goals (see Day 1):
  * One internal message/response schema, so agents never touch vendor SDK types.
  * Credentials and model names come from the environment, never from code.
  * A deterministic `ScriptedLLM` so every lesson runs offline and is testable.
  * A small error taxonomy so Day 23 retry logic can tell "retry" from "give up".

Environment variables:
  LLM_API_KEY   (required for the real client)
  LLM_BASE_URL  (default https://api.openai.com/v1; NVIDIA: https://integrate.api.nvidia.com/v1)
  LLM_MODEL     (default gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
class ToolCall(BaseModel):
    """A model's request to run a tool. `arguments` is already parsed JSON."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Kept so a malformed-JSON failure can be shown to the model / the developer.
    raw_arguments: str | None = None
    arguments_error: str | None = None


class Message(BaseModel):
    role: Role
    content: str | None = None
    name: str | None = None  # tool name for role="tool"
    tool_call_id: str | None = None  # links a tool result to its ToolCall
    tool_calls: list[ToolCall] | None = None  # only on assistant messages

    def to_openai(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.role == "assistant":
            out["content"] = self.content
            if self.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments
                            if tc.raw_arguments is not None
                            else json.dumps(tc.arguments),
                        },
                    }
                    for tc in self.tool_calls
                ]
        else:
            out["content"] = self.content or ""
        if self.role == "tool":
            out["tool_call_id"] = self.tool_call_id
        if self.name and self.role != "assistant":
            out["name"] = self.name
        return out


def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


def tool_result(call: ToolCall, content: str) -> Message:
    return Message(role="tool", content=content, name=call.name, tool_call_id=call.id)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = "stop"  # stop | tool_calls | length | content_filter
    model: str = "unknown"
    latency_s: float = 0.0

    @property
    def content(self) -> str:
        return self.message.content or ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls or []


# --------------------------------------------------------------------------- #
# Errors: the retry policy in Day 23 branches on these types.
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Base class. Not retryable unless a subclass says so."""


class TransientLLMError(LLMError):
    """Timeouts, 5xx, connection resets. Safe to retry with backoff."""


class RateLimitedError(TransientLLMError):
    def __init__(self, msg: str = "rate limited", retry_after_s: float | None = None):
        super().__init__(msg)
        self.retry_after_s = retry_after_s


class PermanentLLMError(LLMError):
    """Bad request, auth failure, context too long. Retrying will not help."""


class ScriptExhaustedError(LLMError):
    """A ScriptedLLM was called more times than it had scripted responses."""


# --------------------------------------------------------------------------- #
# Client protocol + implementations
# --------------------------------------------------------------------------- #
@runtime_checkable
class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


def _parse_tool_args(raw: str | None) -> tuple[dict[str, Any], str | None]:
    """Parse tool-call arguments without ever raising.

    Models occasionally emit invalid JSON. We surface that as data
    (`arguments_error`) so the agent loop can feed it back to the model.
    """
    if not raw or not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments: {exc.msg} (pos {exc.pos})"
    if not isinstance(parsed, dict):
        return {}, f"arguments must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


class OpenAICompatClient:
    """Talks to any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        max_sdk_retries: int = 0,
        http_client: Any | None = None,
    ) -> None:
        """`http_client` (an httpx.Client) lets tests inject a mock transport - no network needed."""
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if not self.api_key:
            raise PermanentLLMError("LLM_API_KEY is not set (see .env.example)")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        # Retries are owned by *our* policy layer (Day 23), so the SDK's are off by default.
        from openai import OpenAI

        self._sdk = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout_s,
            max_retries=max_sdk_retries,
            http_client=http_client,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        started = time.perf_counter()
        try:
            resp = self._sdk.chat.completions.create(**kwargs)
        except openai.RateLimitError as exc:
            retry_after = None
            headers = getattr(getattr(exc, "response", None), "headers", None)
            if headers and headers.get("retry-after"):
                try:
                    retry_after = float(headers["retry-after"])
                except ValueError:
                    pass
            raise RateLimitedError(str(exc), retry_after) from exc
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
            raise TransientLLMError(str(exc)) from exc
        except openai.APIStatusError as exc:
            raise PermanentLLMError(f"{exc.status_code}: {exc}") from exc
        latency = time.perf_counter() - started

        choice = resp.choices[0]
        calls = []
        for tc in choice.message.tool_calls or []:
            args, err = _parse_tool_args(tc.function.arguments)
            calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    raw_arguments=tc.function.arguments,
                    arguments_error=err,
                )
            )
        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return LLMResponse(
            message=Message(role="assistant", content=choice.message.content, tool_calls=calls or None),
            usage=usage,
            finish_reason=choice.finish_reason or "stop",
            model=resp.model or self.model,
            latency_s=latency,
        )


class ScriptedLLM:
    """Deterministic stand-in for a model: replays a fixed script of responses.

    Each script item may be:
      * str                      -> assistant text reply
      * Message / LLMResponse    -> returned as-is
      * callable(messages) -> any of the above (lets a "model" react to input)
      * Exception instance       -> raised (simulate timeouts, rate limits)

    Every call is recorded in `.calls` so tests can assert on exactly what the
    agent sent. This is the seed of the simulation tooling in Day 22.
    """

    def __init__(self, script: Sequence[Any], model: str = "scripted-1") -> None:
        self._script = list(script)
        self._i = 0
        self.model = model
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def tool_call(name: str, call_id: str | None = None, **arguments: Any) -> Message:
        """Convenience: build an assistant message that requests one tool call."""
        cid = call_id or f"call_{name}_{int(time.time_ns()) % 10**8}"
        return Message(role="assistant", tool_calls=[ToolCall(id=cid, name=name, arguments=arguments)])

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": list(messages), "tools": tools, "response_format": response_format,
             "temperature": temperature, "max_tokens": max_tokens}
        )
        if self._i >= len(self._script):
            raise ScriptExhaustedError(
                f"ScriptedLLM got call #{self._i + 1} but only {len(self._script)} responses were scripted"
            )
        item = self._script[self._i]
        self._i += 1
        if callable(item):
            item = item(messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, str):
            item = Message(role="assistant", content=item)
        finish = "tool_calls" if item.tool_calls else "stop"
        prompt_tokens = sum(len(m.content or "") // 4 + 4 for m in messages)
        completion_tokens = len(item.content or "") // 4 + 1
        return LLMResponse(
            message=item,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason=finish,
            model=self.model,
            latency_s=0.0,
        )

    @property
    def remaining(self) -> int:
        return len(self._script) - self._i


def get_client(script: Sequence[Any] | None = None) -> LLMClient:
    """Real client if LLM_API_KEY is set, otherwise a ScriptedLLM built from `script`.

    Lets every demo run offline (deterministic) and against a live model with no code change.
    Set LESSON_OFFLINE=1 to force the scripted client even when a key is present.
    """
    if os.environ.get("LLM_API_KEY") and not os.environ.get("LESSON_OFFLINE"):
        return OpenAICompatClient()
    if script is None:
        raise PermanentLLMError("No LLM_API_KEY set and this demo has no offline script")
    return ScriptedLLM(script)
