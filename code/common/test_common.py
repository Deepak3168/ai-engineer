"""Tests for the shared building blocks. Run: pytest code/common"""

from __future__ import annotations

import os

import numpy as np
import pytest
from pydantic import BaseModel, Field

from common.embeddings import HashEmbedder, cosine_matrix
from common.llm import (
    LLMResponse,
    Message,
    OpenAICompatClient,
    PermanentLLMError,
    RateLimitedError,
    ScriptedLLM,
    ScriptExhaustedError,
    ToolCall,
    _parse_tool_args,
    get_client,
    user,
)
from common.loop import AgentLoop
from common.structured import StructuredOutputError, chat_structured, extract_json
from common.tokens import count_message_tokens, count_tokens
from common.tools import ToolRegistry


# ---------------------------------------------------------------- llm.py ----
def test_scripted_llm_replays_in_order_and_records_calls():
    llm = ScriptedLLM(["one", "two"])
    assert llm.chat([user("a")]).content == "one"
    assert llm.chat([user("b")]).content == "two"
    assert len(llm.calls) == 2 and llm.remaining == 0
    with pytest.raises(ScriptExhaustedError):
        llm.chat([user("c")])


def test_scripted_llm_tool_call_sets_finish_reason():
    llm = ScriptedLLM([ScriptedLLM.tool_call("add", a=1, b=2)])
    resp = llm.chat([user("x")])
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].name == "add" and resp.tool_calls[0].arguments == {"a": 1, "b": 2}


def test_scripted_llm_raises_exception_items_and_supports_callables():
    llm = ScriptedLLM([RateLimitedError("slow down", 1.5), lambda msgs: f"saw {len(msgs)} msgs"])
    with pytest.raises(RateLimitedError) as ei:
        llm.chat([user("x")])
    assert ei.value.retry_after_s == 1.5
    assert llm.chat([user("x"), user("y")]).content == "saw 2 msgs"


@pytest.mark.parametrize(
    "raw,expected_args,has_error",
    [('{"a": 1}', {"a": 1}, False), ("", {}, False), ("{bad", {}, True), ("[1,2]", {}, True)],
)
def test_parse_tool_args_never_raises(raw, expected_args, has_error):
    args, err = _parse_tool_args(raw)
    assert args == expected_args and (err is not None) == has_error


def test_message_to_openai_roundtrip_shape():
    call = ToolCall(id="c1", name="f", arguments={"x": 1})
    m = Message(role="assistant", tool_calls=[call]).to_openai()
    assert m["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
    t = Message(role="tool", content="ok", name="f", tool_call_id="c1").to_openai()
    assert t["tool_call_id"] == "c1"


def test_real_client_requires_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(PermanentLLMError):
        OpenAICompatClient()


def test_get_client_offline_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert isinstance(get_client(["hi"]), ScriptedLLM)
    with pytest.raises(PermanentLLMError):
        get_client()
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LESSON_OFFLINE", "1")
    assert isinstance(get_client(["hi"]), ScriptedLLM)


# -------------------------------------------------------------- tools.py ----
class AddArgs(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def make_registry() -> ToolRegistry:
    reg = ToolRegistry(max_output_chars=50)

    @reg.tool("add", "Add two integers.", AddArgs)
    def add(a: int, b: int) -> int:
        return a + b

    @reg.tool("boom", "Always fails.", AddArgs)
    def boom(a: int, b: int) -> int:
        raise RuntimeError("kaput")

    @reg.tool("long", "Returns a long string.", AddArgs)
    def long(a: int, b: int) -> str:
        return "x" * 500

    return reg


def test_schema_generation_is_openai_shaped():
    schema = make_registry().schemas(only=["add"])[0]
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "add" and fn["parameters"]["required"] == ["a", "b"]
    assert "title" not in fn["parameters"]
    assert fn["parameters"]["properties"]["a"]["description"] == "first number"


def test_execute_success_and_failure_modes():
    reg = make_registry()
    ok = reg.execute(ToolCall(id="1", name="add", arguments={"a": 2, "b": 3}))
    assert ok.ok and ok.content == "5"
    assert reg.execute(ToolCall(id="2", name="nope", arguments={})).error_type == "unknown_tool"
    assert reg.execute(ToolCall(id="3", name="add", arguments={"a": "x"})).error_type == "bad_arguments"
    assert reg.execute(ToolCall(id="4", name="add", arguments={}, arguments_error="invalid JSON")).error_type == "bad_arguments"
    boom = reg.execute(ToolCall(id="5", name="boom", arguments={"a": 1, "b": 1}))
    assert not boom.ok and boom.error_type == "tool_error" and "kaput" in boom.content
    long = reg.execute(ToolCall(id="6", name="long", arguments={"a": 1, "b": 1}))
    assert long.truncated and "truncated" in long.content


def test_duplicate_registration_rejected():
    reg = make_registry()
    with pytest.raises(ValueError):
        reg.tool("add", "dup", AddArgs)(lambda a, b: a + b)


# --------------------------------------------------------------- loop.py ----
def test_loop_completes_after_tool_use():
    llm = ScriptedLLM([ScriptedLLM.tool_call("add", a=2, b=3), "The answer is 5."])
    result = AgentLoop(llm, make_registry(), "You are helpful.").run("2+3?")
    assert result.status == "completed" and result.final_answer == "The answer is 5."
    assert result.tool_call_count == 1
    # the tool observation must be fed back to the model on the second call
    second_call_msgs = llm.calls[1]["messages"]
    assert second_call_msgs[-1].role == "tool" and second_call_msgs[-1].content == "5"


def test_loop_recovers_from_bad_tool_call():
    llm = ScriptedLLM([ScriptedLLM.tool_call("add", a="x"), ScriptedLLM.tool_call("add", a=1, b=1), "2"])
    result = AgentLoop(llm, make_registry(), "sys").run("go")
    assert result.status == "completed" and result.tool_call_count == 2
    assert result.steps[0].tool_results[0][1].error_type == "bad_arguments"


def test_loop_stops_at_max_steps_and_on_budget():
    endless = [ScriptedLLM.tool_call("add", a=1, b=1, call_id=f"c{i}") for i in range(10)]
    r = AgentLoop(ScriptedLLM(endless), make_registry(), "sys", max_steps=3).run("go")
    assert r.status == "max_steps" and len(r.steps) == 3 and r.error
    r2 = AgentLoop(ScriptedLLM(endless), make_registry(), "sys", max_steps=9, max_total_tokens=1).run("go")
    assert r2.status == "budget_exceeded" and len(r2.steps) == 1


def test_loop_reports_llm_error_instead_of_raising():
    r = AgentLoop(ScriptedLLM([RateLimitedError("429")]), make_registry(), "sys").run("go")
    assert r.status == "llm_error" and "RateLimitedError" in (r.error or "")


def test_loop_emits_events():
    events: list[dict] = []
    llm = ScriptedLLM([ScriptedLLM.tool_call("add", a=1, b=2), "3"])
    AgentLoop(llm, make_registry(), "sys", on_event=events.append).run("go")
    assert [e["type"] for e in events] == ["llm_response", "tool_result", "llm_response"]


# ---------------------------------------------------------- structured.py ----
class Person(BaseModel):
    name: str
    age: int = Field(ge=0)


@pytest.mark.parametrize(
    "text",
    ['{"name": "A", "age": 3}', 'Sure!\n```json\n{"name": "A", "age": 3}\n```', 'Here: {"name": "A", "age": 3} done.'],
)
def test_extract_json_variants(text):
    assert Person.model_validate_json(extract_json(text)).name == "A"


def test_extract_json_ignores_braces_inside_strings():
    assert extract_json('x {"a": "}{", "b": 1} y') == '{"a": "}{", "b": 1}'


def test_chat_structured_repairs_then_succeeds():
    llm = ScriptedLLM(['{"name": "A", "age": -1}', '{"name": "A", "age": 4}'])
    person, usage = chat_structured(llm, [user("extract")], Person)
    assert person.age == 4 and usage.total_tokens > 0
    repair_prompt = llm.calls[1]["messages"][-1].content
    assert "age" in repair_prompt  # the validation error was fed back


def test_chat_structured_gives_up():
    llm = ScriptedLLM(["nope"] * 3)
    with pytest.raises(StructuredOutputError) as ei:
        chat_structured(llm, [user("extract")], Person, max_repairs=2)
    assert len(ei.value.attempts) == 3


# ------------------------------------------------- tokens.py / embeddings ----
def test_token_counts_are_positive_and_monotonic():
    assert count_tokens("") == 0
    assert 0 < count_tokens("hello") < count_tokens("hello " * 50)
    assert count_message_tokens([user("hello")]) > count_tokens("hello")


def test_hash_embedder_is_deterministic_normalised_and_lexically_sensible():
    emb = HashEmbedder(dim=256)
    docs = ["reset your password from the login page", "quarterly revenue grew twelve percent", "how to rotate API keys safely"]
    d = emb.embed(docs)
    assert d.shape == (3, 256) and np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(d, emb.embed(docs))
    q = emb.embed(["password reset"])
    assert int(np.argmax(cosine_matrix(q, d))) == 0


# ------------------------------- real client vs. a mock OpenAI-compatible server ----
def _client_with(handler):
    import httpx

    return OpenAICompatClient(
        api_key="test-key", base_url="http://mock.local/v1", model="m-1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _completion(message: dict, finish="stop"):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m-1",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def test_real_client_parses_text_tool_calls_and_usage():
    import httpx

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _j

        seen["body"] = _j.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=_completion({
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
                {"id": "call_2", "type": "function", "function": {"name": "add", "arguments": "{oops"}},
            ],
        }, finish="tool_calls"))

    client = _client_with(handler)
    reg = make_registry()
    resp = client.chat([user("hi")], tools=reg.schemas(), max_tokens=50)
    assert seen["auth"] == "Bearer test-key" and seen["body"]["model"] == "m-1"
    assert seen["body"]["tools"][0]["function"]["name"] in {"add", "boom", "long"}
    assert resp.finish_reason == "tool_calls" and resp.usage.total_tokens == 18
    good, bad = resp.tool_calls
    assert good.arguments == {"a": 1, "b": 2} and good.arguments_error is None
    assert bad.arguments == {} and "invalid JSON" in bad.arguments_error


@pytest.mark.parametrize(
    "status,headers,exc_type",
    [(429, {"retry-after": "2"}, RateLimitedError), (503, {}, "TransientLLMError"), (401, {}, PermanentLLMError), (400, {}, PermanentLLMError)],
)
def test_real_client_maps_http_errors_to_taxonomy(status, headers, exc_type):
    import httpx

    from common import llm as llm_mod

    exc_type = getattr(llm_mod, exc_type) if isinstance(exc_type, str) else exc_type
    client = _client_with(lambda req: httpx.Response(status, json={"error": {"message": "nope"}}, headers=headers))
    with pytest.raises(exc_type) as ei:
        client.chat([user("hi")])
    if status == 429:
        assert ei.value.retry_after_s == 2.0


def test_real_client_connection_failure_is_transient():
    import httpx

    from common.llm import TransientLLMError

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(TransientLLMError):
        _client_with(handler).chat([user("hi")])


# ------------------------------------------------------------- live (opt-in) ----
@pytest.mark.live_llm
@pytest.mark.skipif(not os.environ.get("LLM_API_KEY") or os.environ.get("LESSON_OFFLINE"), reason="needs LLM_API_KEY")
def test_live_llm_smoke():
    resp = get_client().chat([user("Reply with the single word: pong")], max_tokens=10)
    assert isinstance(resp, LLMResponse) and resp.content


# ------------------------------------------------------------ fixtures.py ----
def test_fixture_integrity():
    from common import fixtures as fx

    doc_ids = [d["id"] for d in fx.RUNBOOKS]
    assert len(doc_ids) == len(set(doc_ids)) and len(doc_ids) >= 15
    for inc in fx.INCIDENTS:
        assert set(inc["related_docs"]) <= set(doc_ids), inc["id"]
    for q in fx.RETRIEVAL_QUERIES:
        assert q["relevant"] and set(q["relevant"]) <= set(doc_ids), q["q"]
    assert fx.SECRET_CANARY in fx.POISONED_DOCUMENT["text"]


def test_metrics_are_deterministic_and_spike_only_for_incident_services():
    from common import fixtures as fx

    a = fx.metrics_for("checkout-api", "error_rate")
    assert a == fx.metrics_for("checkout-api", "error_rate")
    assert max(p["value"] for p in a) > 0.2
    assert max(p["value"] for p in fx.metrics_for("unknown-svc", "error_rate")) < 0.01
    with pytest.raises(ValueError):
        fx.metrics_for("checkout-api", "bogus")
    with pytest.raises(KeyError):
        fx.get_incident_record("INC-0")
