"""Show the real HTTP exchange of a chat-completions call, with no network.

`OpenAICompatClient` accepts an injected httpx client. Plugging in `httpx.MockTransport`
lets us capture the exact JSON the SDK puts on the wire and answer with a canned reply.
This is the honest way to see "what is actually sent" without a live endpoint.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, Field

from common.llm import Message, OpenAICompatClient


class WireExchange(BaseModel):
    """One captured request/response pair. The API key is never stored."""

    url: str
    request_body: dict[str, Any]
    response_body: dict[str, Any]
    request_headers: dict[str, str] = Field(default_factory=dict)


def completion_body(
    content: str | None,
    *,
    model: str = "mock-model",
    finish_reason: str = "stop",
    prompt_tokens: int = 42,
    completion_tokens: int = 17,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A response body in the OpenAI chat-completions shape (hand-built, not recorded from a provider)."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def mock_client(handler: Callable[[httpx.Request], httpx.Response], model: str = "mock-model") -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key="test-key-not-real", base_url="http://mock.local/v1", model=model,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def capture_exchange(
    messages: Sequence[Message],
    reply: dict[str, Any],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> WireExchange:
    """Send `messages` through the real client against a mock server and record what crossed the wire."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["headers"] = {k: v for k, v in request.headers.items() if k.lower() in {"content-type", "accept"}}
        return httpx.Response(200, json=reply)

    mock_client(handler).chat(messages, temperature=temperature, max_tokens=max_tokens, tools=tools)
    return WireExchange(
        url=seen["url"], request_body=seen["body"], response_body=reply, request_headers=seen["headers"]
    )
