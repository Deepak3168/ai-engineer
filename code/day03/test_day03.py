"""Day 3 tests: ReAct text parsing/loop, and every guard in GuardedLoop (plus the baseline's blind spot)."""

from __future__ import annotations

import pytest
from day02.ops_tools import TicketStore, build_registry
from pydantic import BaseModel

from common.llm import Message, PermanentLLMError, ScriptedLLM, ToolCall, TransientLLMError
from common.loop import AgentLoop
from common.tools import ToolRegistry
from day03.final_answer import FINAL_TOOL, add_final_tool
from day03.guarded_loop import GuardConfig, GuardedLoop
from day03.react_text import ReActTextLoop, build_react_prompt, parse_react

SYS = "You are the Ops Report Agent."
LONG_ANSWER = "INC-1042 was caused by an unindexed query after the release; follow rb-deploy-rollback."


def tc(name: str, call_id: str, **args: object) -> Message:
    return ScriptedLLM.tool_call(name, call_id, **args)


def history_is_valid(messages: list[Message]) -> bool:
    """Every assistant tool_call id has exactly one tool message, and no tool message is orphaned."""
    wanted = [c.id for m in messages if m.role == "assistant" for c in (m.tool_calls or [])]
    got = [m.tool_call_id for m in messages if m.role == "tool"]
    return sorted(wanted) == sorted(got)


# ============================================================ ReAct parsing
def test_parse_action_with_json_input() -> None:
    p = parse_react('Thought: need the record\nAction: get_incident\nAction Input: {"incident_id": "INC-1042"}')
    assert (p.kind, p.tool, p.arguments, p.thought) == ("action", "get_incident", {"incident_id": "INC-1042"}, "need the record")


def test_parse_final_answer_multiline() -> None:
    p = parse_react("Thought: done\nFinal Answer: Line one.\nLine two.")
    assert p.kind == "final" and p.answer == "Line one.\nLine two."


def test_model_written_observation_is_cut_off() -> None:
    p = parse_react('Action: get_incident\nAction Input: {"incident_id": "INC-1042"}\nObservation: it was DNS\nFinal Answer: DNS')
    assert p.kind == "action"  # the fabricated Final Answer after the fake Observation is discarded too
    assert "DNS" not in p.clean_text


def test_action_and_final_together_is_invalid() -> None:
    p = parse_react('Action: get_incident\nAction Input: {}\nFinal Answer: done already')
    assert p.kind == "invalid" and "both" in (p.error or "")


@pytest.mark.parametrize("text", ["I think the answer is 42.", "", "Thought: hmm"])
def test_no_action_or_final_is_invalid(text: str) -> None:
    assert parse_react(text).kind == "invalid"


def test_empty_final_answer_is_invalid() -> None:
    assert parse_react("Final Answer:   ").kind == "invalid"


def test_fenced_and_prose_wrapped_json_input() -> None:
    p = parse_react('Action: query_metrics\nAction Input: ```json\n{"service": "checkout-api", "metric": "cpu_pct"}\n```')
    assert p.arguments == {"service": "checkout-api", "metric": "cpu_pct"} and p.arguments_error is None


def test_bad_json_input_sets_arguments_error_not_exception() -> None:
    p = parse_react('Action: get_incident\nAction Input: {"incident_id": ')
    assert p.kind == "action" and p.arguments == {} and "not valid JSON" in (p.arguments_error or "")


def test_non_object_input_is_reported() -> None:
    p = parse_react('Action: get_incident\nAction Input: ["INC-1042"]')
    assert "must be a JSON object" in (p.arguments_error or "")


def test_tool_name_is_unquoted_and_input_stops_at_next_thought() -> None:
    p = parse_react('Action: `get_incident`\nAction Input: {"incident_id": "INC-1042"}\nThought: next')
    assert p.tool == "get_incident" and p.arguments == {"incident_id": "INC-1042"}


def test_prompt_lists_every_tool_with_argument_shapes() -> None:
    prompt = build_react_prompt(build_registry())
    for name in ("get_incident", "search_runbooks", "query_metrics", "create_ticket"):
        assert f"- {name}:" in prompt
    assert '"metric": error_rate|p95_latency_ms|cpu_pct' in prompt
    assert '"limit": integer (optional)' in prompt


# ============================================================ ReAct loop
def _react_happy() -> list[str]:
    return [
        'Thought: record\nAction: get_incident\nAction Input: {"incident_id": "INC-1042"}\nObservation: DNS outage',
        'Thought: runbook\nAction: search_runbooks\nAction Input: {"query": "deploy rollback"}',
        f"Thought: done\nFinal Answer: {LONG_ANSWER}",
    ]


def test_react_loop_completes_and_observations_come_from_the_harness() -> None:
    llm = ScriptedLLM(_react_happy())
    res = ReActTextLoop(llm, build_registry()).run("Investigate INC-1042")
    assert res.status == "completed" and res.final_answer == LONG_ANSWER
    assert len(llm.calls) == 3 and all(c["tools"] is None for c in llm.calls)  # no native tools used
    obs = [m.content for m in res.messages if m.role == "user" and (m.content or "").startswith("Observation:")]
    assert len(obs) == 2 and "Checkout API 5xx spike" in obs[0]
    assert not any("DNS outage" in (m.content or "") for m in res.messages)


def test_react_loop_recovers_from_one_format_error() -> None:
    script = ["Let me think.", *_react_happy()]
    res = ReActTextLoop(ScriptedLLM(script), build_registry()).run("t")
    assert res.status == "completed" and res.format_errors == 1
    assert "format error" in (res.messages[3].content or "")


def test_react_loop_gives_up_after_too_many_format_errors() -> None:
    llm = ScriptedLLM(["hmm"] * 5)
    res = ReActTextLoop(llm, build_registry(), max_format_errors=2).run("t")
    assert res.status == "format_error" and res.format_errors == 3 and len(llm.calls) == 3


def test_react_unknown_tool_and_bad_args_become_observations() -> None:
    script = ['Action: nope\nAction Input: {}', 'Action: get_incident\nAction Input: {"incident_id": ', "Final Answer: cannot proceed"]
    res = ReActTextLoop(ScriptedLLM(script), build_registry()).run("t")
    obs = [m.content for m in res.messages if (m.content or "").startswith("Observation:")]
    assert "unknown tool 'nope'" in obs[0] and "could not parse arguments" in obs[1]
    assert res.status == "completed"


def test_react_max_steps_and_budget_and_llm_error() -> None:
    action = 'Action: search_runbooks\nAction Input: {"query": "queue"}'
    assert ReActTextLoop(ScriptedLLM([action] * 3), build_registry(), max_steps=3).run("t").status == "max_steps"
    budget = ReActTextLoop(ScriptedLLM([action] * 3), build_registry(), max_total_tokens=1).run("t")
    assert budget.status == "budget_exceeded" and len(budget.steps) == 1
    err = ReActTextLoop(ScriptedLLM([TransientLLMError("boom")]), build_registry()).run("t")
    assert err.status == "llm_error" and "boom" in (err.error or "")


# ================================================== baseline loop's blind spot
def test_baseline_loop_happily_repeats_a_stuck_call_until_max_steps() -> None:
    stuck = [tc("search_runbooks", f"s{i}", query="latency") for i in range(6)]
    res = AgentLoop(ScriptedLLM(stuck), build_registry(), SYS, max_steps=6).run("t")
    assert res.status == "max_steps" and res.tool_call_count == 6


# ============================================================= guarded loop
class Counting(BaseModel):
    q: str


def counting_registry() -> tuple[ToolRegistry, list[str]]:
    reg, calls = ToolRegistry(), []

    @reg.tool("lookup", "look something up", Counting)
    def lookup(q: str) -> str:
        calls.append(q)
        return f"result for {q}"

    return reg, calls


def test_guarded_matches_baseline_on_the_happy_path() -> None:
    script = [tc("get_incident", "a", incident_id="INC-1042"), LONG_ANSWER]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS).run("t")
    assert (res.status, res.final_answer, len(res.steps), res.blocked) == ("completed", LONG_ANSWER, 2, 0)
    assert history_is_valid(res.messages) and res.messages[-1].content == LONG_ANSWER


def test_identical_call_is_blocked_not_executed() -> None:
    reg, executions = counting_registry()
    script = [tc("lookup", f"c{i}", q="same") for i in range(3)] + [LONG_ANSWER]
    res = GuardedLoop(ScriptedLLM(script), reg, SYS, GuardConfig(max_identical_calls=2, max_blocked=5, no_progress_steps=9)).run("t")
    assert executions == ["same", "same"]  # third attempt never reached the tool
    assert res.status == "completed" and res.blocked == 1
    blocked = [r for s in res.steps for _, r in s.tool_results if r.error_type == "repeat_blocked"]
    assert len(blocked) == 1 and "exactly these arguments" in blocked[0].content


def test_argument_order_does_not_defeat_repeat_detection() -> None:
    llm = ScriptedLLM([
        tc("query_metrics", "a", service="checkout-api", metric="cpu_pct"),
        tc("query_metrics", "b", metric="cpu_pct", service="checkout-api"),
        tc("query_metrics", "c", metric="cpu_pct", service="checkout-api"),
        LONG_ANSWER,
    ])
    res = GuardedLoop(llm, build_registry(), SYS, GuardConfig(no_progress_steps=9, max_blocked=9)).run("t")
    assert res.blocked == 1 and res.executed_by_tool == {"query_metrics": 2}


def test_repeated_blocking_aborts_with_loop_detected() -> None:
    reg, executions = counting_registry()
    stuck = [tc("lookup", f"c{i}", q="same") for i in range(10)]
    res = GuardedLoop(ScriptedLLM(stuck), reg, SYS, GuardConfig(max_steps=10, max_blocked=3, no_progress_steps=9)).run("t")
    assert res.status == "loop_detected" and res.blocked == 3 and len(executions) == 2
    assert len(res.steps) == 5 < 10
    assert history_is_valid(res.messages)


def test_per_tool_cap_blocks_even_with_different_arguments() -> None:
    reg, executions = counting_registry()
    script = [tc("lookup", f"c{i}", q=f"q{i}") for i in range(3)] + [LONG_ANSWER]
    res = GuardedLoop(ScriptedLLM(script), reg, SYS, GuardConfig(max_calls_per_tool={"lookup": 2}, max_blocked=9)).run("t")
    assert executions == ["q0", "q1"] and res.blocked == 1 and res.status == "completed"
    capped = [r for s in res.steps for _, r in s.tool_results if r.error_type == "tool_cap"]
    assert "(2) is used up" in capped[0].content


def test_no_progress_detects_varied_calls_that_learn_nothing() -> None:
    script = [tc("search_runbooks", f"w{i}", query=f"zzqx{i} nothing") for i in range(8)]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS, GuardConfig(max_steps=8, no_progress_steps=3)).run("t")
    assert res.status == "no_progress" and len(res.steps) == 4  # 1 new "[]" result, then 3 stale steps
    assert res.blocked == 0  # nothing was repeated, so only this guard could catch it


def test_new_information_resets_the_no_progress_counter() -> None:
    script = [
        tc("search_runbooks", "a", query="zzqx"), tc("search_runbooks", "b", query="zzqy"),
        tc("get_incident", "c", incident_id="INC-1042"),  # new info resets
        tc("search_runbooks", "d", query="zzqz"), tc("search_runbooks", "e", query="zzqw"),
        LONG_ANSWER,
    ]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS, GuardConfig(no_progress_steps=3)).run("t")
    assert res.status == "completed"


def test_errors_are_not_progress() -> None:
    script = [tc("get_incident", f"e{i}", incident_id=f"INC-000{i}") for i in range(6)]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS, GuardConfig(no_progress_steps=3)).run("t")
    assert res.status == "no_progress" and len(res.steps) == 3


def test_wrap_up_returns_partial_answer_without_tools() -> None:
    script = [tc("search_runbooks", f"w{i}", query=f"zzqx{i}") for i in range(4)] + ["Nothing matched; incident unknown."]
    llm = ScriptedLLM(script)
    res = GuardedLoop(llm, build_registry(), SYS, GuardConfig(no_progress_steps=3, wrap_up=True)).run("t")
    assert res.status == "no_progress" and res.partial_answer == "Nothing matched; incident unknown."
    assert llm.calls[-1]["tools"] is None and (llm.calls[-1]["messages"][-1].content or "").startswith("Stop calling tools")
    assert res.final_answer is None  # a partial answer is not a final answer


def test_wrap_up_failure_is_recorded_not_raised() -> None:
    script = [tc("search_runbooks", f"w{i}", query=f"zzqx{i}") for i in range(4)] + [TransientLLMError("down")]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS, GuardConfig(no_progress_steps=3, wrap_up=True)).run("t")
    assert res.status == "no_progress" and "wrap-up failed" in (res.error or "") and res.partial_answer is None


def test_max_steps_exit_keeps_history_valid() -> None:
    script = [tc("get_incident", f"g{i}", incident_id="INC-1042" if i % 2 else "INC-1051") for i in range(3)]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS, GuardConfig(max_steps=3, no_progress_steps=9)).run("t")
    assert res.status == "max_steps" and "no final answer within 3 steps" in (res.error or "")
    assert history_is_valid(res.messages)


def test_token_budget_stops_the_run() -> None:
    script = [tc("get_incident", "a", incident_id="INC-1042"), LONG_ANSWER]
    res = GuardedLoop(ScriptedLLM(script), build_registry(), SYS, GuardConfig(max_total_tokens=10)).run("t")
    assert res.status == "budget_exceeded" and len(res.steps) == 1


def test_llm_error_is_reported_with_type() -> None:
    res = GuardedLoop(ScriptedLLM([PermanentLLMError("bad key")]), build_registry(), SYS).run("t")
    assert res.status == "llm_error" and res.error == "PermanentLLMError: bad key"


def test_duplicate_call_ids_abort_without_executing() -> None:
    msg = Message(role="assistant", tool_calls=[
        ToolCall(id="dup", name="get_incident", arguments={"incident_id": "INC-1042"}),
        ToolCall(id="dup", name="get_incident", arguments={"incident_id": "INC-1051"}),
    ])
    res = GuardedLoop(ScriptedLLM([msg]), build_registry(), SYS).run("t")
    assert res.status == "llm_error" and "duplicate tool call ids" in (res.error or "") and res.executed_by_tool == {}


def test_blocked_and_executed_calls_in_one_parallel_turn_keep_order() -> None:
    reg, executions = counting_registry()
    two = Message(role="assistant", tool_calls=[
        ToolCall(id="x1", name="lookup", arguments={"q": "a"}),
        ToolCall(id="x2", name="lookup", arguments={"q": "b"}),
    ])
    cfg = GuardConfig(max_calls_per_tool={"lookup": 1}, max_blocked=9, no_progress_steps=9)
    res = GuardedLoop(ScriptedLLM([two, LONG_ANSWER]), reg, SYS, cfg).run("t")
    tool_msgs = [m for m in res.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["x1", "x2"]
    assert tool_msgs[0].content == "result for a" and tool_msgs[1].content.startswith("Error: call budget")
    assert executions == ["a"]


def test_writes_are_denied_by_default_policy_and_allowed_when_granted() -> None:
    args = {"title": "Add index on orders", "severity": "SEV2", "idempotency_key": "inc-1042-index"}
    store = TicketStore()
    denied = GuardedLoop(ScriptedLLM([tc("create_ticket", "t1", **args), LONG_ANSWER]), build_registry(store), SYS).run("t")
    assert store.executions == 0
    assert denied.steps[0].tool_results[0][1].error_type == "denied"
    GuardedLoop(ScriptedLLM([tc("create_ticket", "t1", **args), LONG_ANSWER]), build_registry(store), SYS,
                GuardConfig(allow_side_effects=("read", "write"))).run("t")
    assert store.executions == 1


def test_explicit_final_tool_ends_the_run_with_validated_answer() -> None:
    reg = add_final_tool(build_registry())
    submit = tc(FINAL_TOOL, "f1", answer=LONG_ANSWER, evidence=["INC-1042"], confidence="high")
    llm = ScriptedLLM([submit, "never reached"])
    res = GuardedLoop(llm, reg, SYS, GuardConfig(final_tool=FINAL_TOOL)).run("t")
    assert res.status == "completed" and res.final_answer == LONG_ANSWER and len(llm.calls) == 1
    assert history_is_valid(res.messages)


def test_invalid_final_submission_does_not_end_the_run() -> None:
    reg = add_final_tool(build_registry())
    bad = tc(FINAL_TOOL, "f1", answer=LONG_ANSWER, evidence=[], confidence="high")
    good = tc(FINAL_TOOL, "f2", answer=LONG_ANSWER, evidence=["INC-1042"], confidence="high")
    res = GuardedLoop(ScriptedLLM([bad, good]), reg, SYS, GuardConfig(final_tool=FINAL_TOOL)).run("t")
    assert res.status == "completed" and len(res.steps) == 2
    assert res.steps[0].tool_results[0][1].error_type == "bad_arguments"


def test_events_report_blocked_calls_and_guard_stop() -> None:
    events: list[dict] = []
    reg, _ = counting_registry()
    stuck = [tc("lookup", f"c{i}", q="same") for i in range(6)]
    GuardedLoop(ScriptedLLM(stuck), reg, SYS, GuardConfig(max_blocked=1, no_progress_steps=9), on_event=events.append).run("t")
    assert any(e["type"] == "tool_result" and e["blocked"] for e in events)
    assert events[-1]["type"] == "guard_stop" and events[-1]["status"] == "loop_detected"


def test_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        GuardConfig(max_steps=0)
