"""Day 1 tests: wire format, budget guard, structured extraction, repair, native mode, error mapping."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from common.llm import (
    LLMResponse,
    PermanentLLMError,
    RateLimitedError,
    ScriptedLLM,
    TransientLLMError,
    Usage,
    assistant,
    system,
    user,
)
from common.structured import StructuredOutputError
from common.tokens import count_tokens
from day01.budget import ContextBudgetError, guard_input, input_budget, truncate_to_tokens
from day01.triage import IncidentTriage, native_response_format, triage_alert, triage_native
from day01.wire import capture_exchange, completion_body, mock_client

ALERT = "checkout-api error_rate=31% since 09:12Z after deploy v2026.03.04-1"
VALID = {
    "severity": "SEV1", "service": "checkout-api", "summary": "Error rate 31% after deploy.",
    "suspected_cause": "bad deploy", "needs_page": True, "confidence": 0.8,
}


def valid_json(**overrides: object) -> str:
    return json.dumps({**VALID, **overrides})


# ----------------------------------------------------------------- wire format
def test_request_body_shape_and_statelessness():
    history = [system("sys"), user("q1")]
    first = capture_exchange(history, completion_body("a1"), max_tokens=50)
    assert first.url.endswith("/chat/completions")
    assert first.request_body["model"] == "mock-model"
    assert first.request_body["max_tokens"] == 50
    assert first.request_body["temperature"] == 0.0
    assert [m["role"] for m in first.request_body["messages"]] == ["system", "user"]

    history += [assistant("a1"), user("q2")]
    second = capture_exchange(history, completion_body("a2"))
    # The server keeps no state: turn 2 must carry all of turn 1.
    assert [m["content"] for m in second.request_body["messages"]] == ["sys", "q1", "a1", "q2"]


def test_api_key_is_not_recorded_in_captured_exchange():
    ex = capture_exchange([user("hi")], completion_body("yo"))
    assert "test-key-not-real" not in ex.model_dump_json()


def test_response_usage_and_finish_reason_are_parsed():
    reply = completion_body("cut off", finish_reason="length", prompt_tokens=10, completion_tokens=5)
    client = mock_client(lambda req: httpx.Response(200, json=reply))
    resp = client.chat([user("x")])
    assert resp.finish_reason == "length"
    assert resp.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (429, {"retry-after": "2"}, RateLimitedError),
        (500, {}, TransientLLMError),
        (400, {}, PermanentLLMError),
        (401, {}, PermanentLLMError),
    ],
)
def test_http_status_maps_to_error_taxonomy(status, headers, expected):
    client = mock_client(lambda req: httpx.Response(status, json={"error": {"message": "nope"}}, headers=headers))
    with pytest.raises(expected) as info:
        client.chat([user("x")])
    if status == 429:
        assert info.value.retry_after_s == 2.0
    else:
        assert not isinstance(info.value, RateLimitedError)
    # Transient errors are retryable, permanent ones are not.
    assert isinstance(info.value, TransientLLMError) == (status in (429, 500))


# -------------------------------------------------------------- budget guard
def test_input_budget_arithmetic_and_validation():
    assert input_budget(1000, 100, safety_margin=0.1) == 800
    with pytest.raises(ValueError):
        input_budget(100, 100)
    with pytest.raises(ValueError):
        input_budget(0, 10)
    with pytest.raises(ValueError):
        input_budget(1000, 10, safety_margin=1.0)


def test_guard_input_passes_small_and_rejects_large():
    assert guard_input([user("short")], context_window=1000, max_output_tokens=100) > 0
    with pytest.raises(ContextBudgetError) as info:
        guard_input([user("word " * 5000)], context_window=1000, max_output_tokens=100)
    assert info.value.needed > info.value.allowed
    assert isinstance(info.value, PermanentLLMError)  # never retried


def test_over_budget_alert_fails_before_any_model_call():
    client = ScriptedLLM([valid_json()])
    with pytest.raises(ContextBudgetError):
        triage_alert(client, "log line\n" * 10_000)
    assert client.calls == []  # the guard saved the round trip


def test_truncate_to_tokens_keeps_head_and_tail_within_budget():
    text = "HEAD " + "x " * 2000 + "TAIL"
    out = truncate_to_tokens(text, 50)
    assert count_tokens(out) <= 50
    assert out.startswith("HEAD") and out.endswith("TAIL") and "truncated" in out
    assert truncate_to_tokens("tiny", 50) == "tiny"
    with pytest.raises(ValueError):
        truncate_to_tokens("x", 0)


# ----------------------------------------------------------- structured output
def test_triage_happy_path_makes_one_call():
    client = ScriptedLLM([valid_json()])
    result = triage_alert(client, ALERT)
    assert result.triage.severity == "SEV1" and result.triage.needs_page is True
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 400
    # The JSON Schema travels in the system prompt; the alert is delimited as data.
    assert "additionalProperties" in client.calls[0]["messages"][0].content
    assert "<alert>" in client.calls[0]["messages"][-1].content


def test_fenced_json_with_prose_is_accepted():
    client = ScriptedLLM([f"Here you go:\n```json\n{valid_json()}\n```\nHope that helps!"])
    assert triage_alert(client, ALERT).triage.service == "checkout-api"
    assert len(client.calls) == 1


def test_repair_loop_feeds_validation_errors_back_then_succeeds():
    client = ScriptedLLM([valid_json(severity="SEV0"), valid_json(needs_page=False), valid_json()])
    result = triage_alert(client, ALERT)
    assert len(client.calls) == 3
    feedback_1 = client.calls[1]["messages"][-1].content
    feedback_2 = client.calls[2]["messages"][-1].content
    assert "severity" in feedback_1 and "SEV1" in feedback_1  # tells the model which field and what is allowed
    assert "needs_page" in feedback_2  # semantic rule surfaced, not just types
    # The invalid answer is part of the conversation the model repairs against.
    assert client.calls[1]["messages"][-2].content == valid_json(severity="SEV0")
    # Usage is summed over all attempts: repairs cost real money.
    assert result.usage.prompt_tokens > 2 * len(client.calls[0]["messages"])


def test_repairs_are_bounded_and_error_carries_attempts():
    client = ScriptedLLM(["nope", "{bad", "[]", valid_json()])
    with pytest.raises(StructuredOutputError) as info:
        triage_alert(client, ALERT, max_repairs=2)
    assert len(client.calls) == 3 and client.remaining == 1  # 4th scripted reply never consumed
    assert info.value.attempts == ["nope", "{bad", "[]"]
    assert isinstance(info.value, PermanentLLMError)


def test_zero_repairs_means_exactly_one_attempt():
    client = ScriptedLLM(["nope", valid_json()])
    with pytest.raises(StructuredOutputError):
        triage_alert(client, ALERT, max_repairs=0)
    assert len(client.calls) == 1


def test_model_rejects_extra_fields_and_bad_ranges():
    with pytest.raises(ValidationError):
        IncidentTriage.model_validate({**VALID, "extra": 1})
    with pytest.raises(ValidationError):
        IncidentTriage.model_validate({**VALID, "confidence": 1.5})
    with pytest.raises(ValidationError):
        IncidentTriage.model_validate({**VALID, "summary": "x" * 281})
    with pytest.raises(ValidationError):
        IncidentTriage.model_validate({**VALID, "severity": "SEV1", "needs_page": False})
    # Nullable but *required*: omitting suspected_cause is an error (strict-mode friendly).
    missing = {k: v for k, v in VALID.items() if k != "suspected_cause"}
    with pytest.raises(ValidationError):
        IncidentTriage.model_validate(missing)


def test_empty_alert_rejected_before_model_call():
    client = ScriptedLLM([valid_json()])
    with pytest.raises(ValueError):
        triage_alert(client, "   ")
    assert client.calls == []


def test_transient_error_from_client_propagates_unchanged():
    client = ScriptedLLM([TransientLLMError("timeout")])
    with pytest.raises(TransientLLMError):  # retry policy is the caller's job (Day 23), not the repair loop's
        triage_alert(client, ALERT)


# ------------------------------------------------------------------ native mode
def test_native_response_format_shape():
    rf = native_response_format(IncidentTriage)
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(IncidentTriage.model_fields)  # strict modes want all required


def test_native_mode_sends_response_format_and_validates():
    client = ScriptedLLM([valid_json()])
    result = triage_native(client, ALERT)
    assert result.strategy == "native" and result.triage.service == "checkout-api"
    assert client.calls[0]["response_format"]["json_schema"]["name"] == "incident_triage"
    assert len(client.calls) == 1


def test_native_mode_truncation_is_a_distinct_error_not_a_repair():
    cut = LLMResponse(message=assistant('{"severity": "SEV1", "serv'), finish_reason="length", usage=Usage())
    client = ScriptedLLM([cut, valid_json()])
    with pytest.raises(PermanentLLMError, match="truncated"):
        triage_native(client, ALERT)
    assert len(client.calls) == 1


def test_native_mode_still_enforces_semantic_rules():
    client = ScriptedLLM([valid_json(needs_page=False)])
    with pytest.raises(StructuredOutputError):  # schema-valid but violates the SEV1 rule; no repair in native path
        triage_native(client, ALERT)
