"""Day 2 tests: schemas, validation failures as observations, manual protocol, batch execution, idempotency."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import BaseModel

from common.llm import Message, ScriptedLLM, ToolCall
from common.tools import ToolRegistry
from day02.execution import execute_batch, results_to_messages
from day02.manual_call import one_round
from day02.ops_tools import TicketStore, build_registry, keyword_search, summarize_series


def call(name: str, call_id: str = "c1", **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


TICKET = {"title": "Add index on orders", "severity": "SEV2", "idempotency_key": "inc-1042-index"}


# ------------------------------------------------------------------ schemas
def test_schemas_are_strict_and_carry_descriptions() -> None:
    schemas = {s["function"]["name"]: s["function"] for s in build_registry().schemas()}
    assert set(schemas) == {"get_incident", "search_runbooks", "query_metrics", "create_ticket"}
    params = schemas["create_ticket"]["parameters"]
    assert params["additionalProperties"] is False
    assert params["properties"]["severity"]["enum"] == ["SEV1", "SEV2", "SEV3"]
    assert set(params["required"]) == {"title", "severity", "idempotency_key"}
    assert "title" not in params  # Pydantic's model title is stripped to save tokens
    assert schemas["create_ticket"]["description"].startswith("WRITE:")
    assert "incident_id" in schemas["get_incident"]["parameters"]["required"]


def test_side_effect_metadata() -> None:
    reg = build_registry()
    assert reg.get("get_incident").side_effect == "read"
    assert reg.get("create_ticket").side_effect == "write"
    assert reg.get("create_ticket").idempotent is True


# ------------------------------------------------- failures are observations
@pytest.mark.parametrize(
    ("tool_call", "error_type", "needle"),
    [
        (call("get_incident", incident_id="INC-1042", verbose=True), "bad_arguments", "verbose"),
        (call("get_incident", incident_id="1042"), "bad_arguments", "incident_id"),
        (call("create_ticket", **{**TICKET, "severity": "SEV9"}), "bad_arguments", "severity"),
        (call("query_metrics", service="checkout-api", metric="disk"), "bad_arguments", "metric"),
        (call("get_incident", incident_id="INC-9999"), "tool_error", "Known ids"),
        (call("query_metrics", service="checkout-api", metric="cpu_pct", minutes=500), "bad_arguments", "minutes"),
        (call("nope"), "unknown_tool", "Available tools"),
    ],
)
def test_bad_calls_return_actionable_errors(tool_call: ToolCall, error_type: str, needle: str) -> None:
    res = build_registry().execute(tool_call)
    assert res.ok is False
    assert res.error_type == error_type
    assert needle in res.content


def test_malformed_json_arguments_are_reported_not_raised() -> None:
    bad = ToolCall(id="c1", name="get_incident", raw_arguments='{"incident_id": ', arguments_error="invalid JSON arguments")
    res = build_registry().execute(bad)
    assert (res.ok, res.error_type) == (False, "bad_arguments")
    assert "invalid JSON" in res.content


def test_tool_bug_does_not_escape_registry() -> None:
    class Empty(BaseModel):
        pass

    reg = ToolRegistry()

    @reg.tool("boom", "always fails", Empty)
    def boom() -> str:
        raise RuntimeError("db down")

    res = reg.execute(call("boom"))
    assert res.error_type == "tool_error"
    assert "RuntimeError: db down" in res.content


def test_duplicate_tool_name_is_rejected() -> None:
    reg = build_registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register(reg.get("get_incident"))


# ------------------------------------------------------------ tool behaviour
def test_get_incident_returns_json_record() -> None:
    res = build_registry().execute(call("get_incident", incident_id="INC-1042"))
    assert res.ok
    assert json.loads(res.content)["service"] == "checkout-api"


def test_keyword_search_ranks_by_overlap_and_limits() -> None:
    hits = keyword_search("queue backlog consumers restart", 2)
    assert hits[0]["id"] == "rb-queue-backlog"
    assert len(hits) <= 2
    assert keyword_search("zzzz qqqq", 3) == []


def test_keyword_search_is_lexical_and_misses_paraphrase() -> None:
    # Documents the limitation Days 8-10 fix: zero shared words means zero hits, even for the right document.
    assert keyword_search("memoisation layer wiped everything slowed", 5) == []
    assert keyword_search("the of and", 5) == []  # stopwords alone never match


def test_metrics_result_is_a_compact_summary() -> None:
    res = build_registry().execute(call("query_metrics", service="checkout-api", metric="error_rate"))
    data = json.loads(res.content)
    assert data["points"] == 60 and data["max"] == pytest.approx(0.31)
    assert len(data["samples"]) <= 6
    assert len(res.content) < 400


def test_summarize_series_peak_and_mean() -> None:
    s = summarize_series([{"minute": 0.0, "value": 1.0}, {"minute": 1.0, "value": 5.0}, {"minute": 2.0, "value": 3.0}])
    assert (s["peak_minute"], s["max"], s["min"], s["mean"]) == (1, 5.0, 1.0, 3.0)


def test_oversized_output_is_truncated_and_flagged() -> None:
    reg = build_registry(max_output_chars=100)
    res = reg.execute(call("get_incident", incident_id="INC-1042"))
    assert res.ok and res.truncated
    assert "[truncated" in res.content
    assert len(res.content) < 100 + 40


# --------------------------------------------------------- manual round trip
def _two_call_message() -> Message:
    return Message(role="assistant", tool_calls=[
        call("get_incident", "call_a", incident_id="INC-1042"),
        call("search_runbooks", "call_b", query="deploy rollback"),
    ])


def test_one_round_replays_assistant_message_then_one_tool_message_per_call() -> None:
    llm = ScriptedLLM([_two_call_message(), "Done."])
    trace = one_round(llm, build_registry(), "What happened in INC-1042?")
    assert len(llm.calls) == 2
    assert llm.calls[0]["tools"] and llm.calls[1]["tools"]  # tools are re-sent on every request
    roles = [m.role for m in llm.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    tool_msgs = [m for m in llm.calls[1]["messages"] if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["call_a", "call_b"]
    assert llm.calls[1]["messages"][2].tool_calls[0].id == "call_a"
    assert trace.final_text == "Done." and trace.needs_another_round is False
    assert trace.assistant_1["content"] is None
    assert json.loads(trace.assistant_1["tool_calls"][0]["function"]["arguments"]) == {"incident_id": "INC-1042"}


def test_one_round_without_tool_calls_makes_a_single_request() -> None:
    llm = ScriptedLLM(["I need an incident id."])
    trace = one_round(llm, build_registry(), "What happened?")
    assert len(llm.calls) == 1
    assert trace.final_text == "I need an incident id." and trace.request_2 is None


def test_one_round_flags_when_model_wants_another_round() -> None:
    llm = ScriptedLLM([_two_call_message(), ScriptedLLM.tool_call("query_metrics", "call_c", service="checkout-api", metric="cpu_pct")])
    trace = one_round(llm, build_registry(), "Investigate INC-1042")
    assert trace.needs_another_round is True
    assert trace.final_text is None


def test_one_round_feeds_tool_errors_back_to_the_model() -> None:
    bad = ScriptedLLM.tool_call("get_incident", "call_x", incident_id="INC-0000")
    seen: list[str] = []

    def second(messages: list[Message]) -> str:
        seen.append(messages[-1].content or "")
        return "That incident does not exist."

    trace = one_round(ScriptedLLM([bad, second]), build_registry(), "Look up INC-0000")
    assert "Known ids" in seen[0]
    assert trace.results[0].error_type == "tool_error"


# ------------------------------------------------------------ batch execution
def test_batch_preserves_order_and_one_result_per_call() -> None:
    reg = build_registry()
    calls = [call("get_incident", "a", incident_id="INC-1051"), call("nope", "b"), call("get_incident", "c", incident_id="INC-1042")]
    results = execute_batch(reg, calls)
    assert [r.error_type for r in results] == [None, "unknown_tool", None]
    assert "INC-1051" in results[0].content and "INC-1042" in results[2].content
    msgs = results_to_messages(calls, results)
    assert [m.tool_call_id for m in msgs] == ["a", "b", "c"]


def test_read_calls_really_run_concurrently() -> None:
    barrier = threading.Barrier(3, timeout=2)

    class Args(BaseModel):
        n: int

    reg = ToolRegistry()

    @reg.tool("slow_read", "waits for its siblings", Args)
    def slow_read(n: int) -> str:
        barrier.wait()  # raises BrokenBarrierError (-> tool_error) unless 3 calls are in flight together
        return f"ok {n}"

    results = execute_batch(reg, [call("slow_read", f"c{i}", n=i) for i in range(3)], max_workers=3)
    assert [r.content for r in results] == ["ok 0", "ok 1", "ok 2"]


def test_write_calls_run_sequentially_in_model_order() -> None:
    order: list[str] = []
    active = 0
    overlap = False
    lock = threading.Lock()

    class Args(BaseModel):
        tag: str

    reg = ToolRegistry()

    @reg.tool("w", "write", Args, side_effect="write")
    def w(tag: str) -> str:
        nonlocal active, overlap
        with lock:
            active += 1
            overlap = overlap or active > 1
        order.append(tag)
        with lock:
            active -= 1
        return tag

    execute_batch(reg, [call("w", f"c{i}", tag=t) for i, t in enumerate("xyz")], allow=("write",))
    assert order == ["x", "y", "z"] and overlap is False


def test_policy_denies_writes_without_executing_them() -> None:
    store = TicketStore()
    res = execute_batch(build_registry(store), [call("create_ticket", **TICKET)])[0]
    assert (res.ok, res.error_type) == (False, "denied")
    assert "Do not retry" in res.content
    assert store.executions == 0 and store.tickets == {}


def test_duplicate_call_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate tool_call ids"):
        execute_batch(build_registry(), [call("get_incident", "same", incident_id="INC-1042")] * 2)


def test_results_to_messages_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="exactly one result"):
        results_to_messages([call("get_incident", incident_id="INC-1042")], [])


# ---------------------------------------------------------------- idempotency
def test_same_idempotency_key_creates_one_ticket() -> None:
    store = TicketStore()
    reg = build_registry(store)
    first = json.loads(reg.execute(call("create_ticket", "a", **TICKET)).content)
    second = json.loads(reg.execute(call("create_ticket", "b", **TICKET)).content)
    assert first == {"ticket_id": "TCK-001", "created": True}
    assert second == {"ticket_id": "TCK-001", "created": False}
    assert store.executions == 1


def test_different_keys_create_different_tickets() -> None:
    store = TicketStore()
    reg = build_registry(store)
    reg.execute(call("create_ticket", **TICKET))
    reg.execute(call("create_ticket", **{**TICKET, "idempotency_key": "another-key-1"}))
    assert store.executions == 2 and len(store.tickets) == 2


def test_idempotency_holds_under_concurrent_retries() -> None:
    store = TicketStore()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: store.create("Add index on orders", "SEV2", "same-key-123"), range(32)))
    assert store.executions == 1 and len(store.tickets) == 1
