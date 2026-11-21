"""Day 4 tests: plan validation, layering, execution semantics, planner repair, replanning, verification."""

from __future__ import annotations

import json
from typing import Any

import pytest
from day02.ops_tools import TicketStore, build_registry

from common.llm import ScriptedLLM
from day04.executor import PlanExecutor, PlanReferenceError, resolve_args
from day04.plan import Plan, PlanError, find_refs, parse_plan, topological_layers, transitive_dependents, validate_plan
from day04.planner import PlanValidationError, make_plan
from day04.replan import run_with_replanning
from day04.verify import plan_quality, verify_citations

GOAL = "Draft a postmortem for INC-1042"


def step(sid: str, tool: str, args: dict[str, Any], deps: list[str] | None = None) -> dict[str, Any]:
    return {"id": sid, "tool": tool, "args": args, "depends_on": deps or [], "purpose": f"purpose of {sid}"}


def good_plan(incident: str = "INC-1042") -> dict[str, Any]:
    return {"goal": GOAL, "steps": [
        step("s1", "get_incident", {"incident_id": incident}),
        step("s2", "query_metrics", {"service": "${s1.service}", "metric": "error_rate"}, ["s1"]),
        step("s3", "query_metrics", {"service": "${s1.service}", "metric": "p95_latency_ms"}, ["s1"]),
        step("s4", "search_runbooks", {"query": "deploy rollback"}),
    ]}


def plan_of(data: dict[str, Any]) -> Plan:
    result = parse_plan(data)
    assert isinstance(result, Plan), result
    return result


# ------------------------------------------------------------------ structure --
def codes(data: dict[str, Any]) -> list[str]:
    result = parse_plan(data)
    return [] if isinstance(result, Plan) else [i.message for i in result]


def test_valid_plan_parses_and_exposes_ids() -> None:
    assert plan_of(good_plan()).ids == ["s1", "s2", "s3", "s4"]


def test_duplicate_ids_self_dependency_and_cycles_are_rejected() -> None:
    dup = good_plan()
    dup["steps"][3]["id"] = "s1"
    assert any("[duplicate_id]" in m for m in codes(dup))

    selfdep = good_plan()
    selfdep["steps"][3]["depends_on"] = ["s4"]
    assert any("[self_dependency]" in m for m in codes(selfdep))

    cyc = good_plan()
    cyc["steps"][1]["depends_on"], cyc["steps"][2]["depends_on"] = ["s1", "s3"], ["s1", "s2"]
    msgs = codes(cyc)
    assert any("[cycle]" in m and "->" in m for m in msgs)


def test_reference_must_be_declared_as_dependency() -> None:
    bad = good_plan()
    bad["steps"][1]["depends_on"] = []
    assert any("[undeclared_ref]" in m for m in codes(bad))


def test_schema_limits_and_unknown_fields_rejected() -> None:
    extra = good_plan()
    extra["steps"][0]["surprise"] = 1
    assert codes(extra)
    assert codes({"goal": GOAL, "steps": []})
    too_many = {"goal": GOAL, "steps": [step(f"s{i}", "get_incident", {"incident_id": "INC-1042"}) for i in range(13)]}
    assert codes(too_many)
    bad_id = good_plan()
    bad_id["steps"][0]["id"] = "S 1"
    assert codes(bad_id)


def test_find_refs_walks_nested_structures_but_only_matches_whole_strings() -> None:
    assert find_refs({"a": ["${s1.x.y}", {"b": "${s2}"}], "c": "prefix ${s3.z}"}) == [("s1", "x.y"), ("s2", "")]


# ------------------------------------------------------------ context checks --
def test_validate_plan_catches_unknown_tool_bad_args_and_missing_dependency() -> None:
    reg = build_registry(TicketStore())
    data = good_plan()
    data["steps"][3]["tool"] = "runbook_search"
    data["steps"].append(step("s5", "get_incident", {"incident_id": 5}, ["s9"]))
    found = {i.code for i in validate_plan(plan_of(data), reg)}
    assert {"unknown_tool", "missing_dependency", "bad_args"} <= found


def test_validate_plan_allows_dependency_on_known_completed_steps_and_blocks_id_reuse() -> None:
    reg = build_registry(TicketStore())
    follow_up = plan_of({"goal": GOAL, "steps": [step("n1", "query_metrics", {"service": "${s1.service}", "metric": "cpu_pct"}, ["s1"])]})
    assert validate_plan(follow_up, reg, known={"s1"}) == []
    assert [i.code for i in validate_plan(follow_up, reg)] == ["missing_dependency"]
    reuse = plan_of({"goal": GOAL, "steps": [step("s1", "search_runbooks", {"query": "x y"})]})
    assert [i.code for i in validate_plan(reuse, reg, reserved={"s1"})] == ["duplicate_id"]


def test_side_effect_policy_rejects_write_tools_by_default() -> None:
    reg = build_registry(TicketStore())
    data = {"goal": GOAL, "steps": [step("s1", "create_ticket", {"title": "Add index", "severity": "SEV2", "idempotency_key": "inc-1042-idx"})]}
    assert [i.code for i in validate_plan(plan_of(data), reg)] == ["tool_not_allowed"]
    assert validate_plan(plan_of(data), reg, allow=("read", "write")) == []


# --------------------------------------------------------------------- layers --
def test_topological_layers_group_independent_steps() -> None:
    layers = topological_layers(plan_of(good_plan()))
    assert [[s.id for s in layer] for layer in layers] == [["s1", "s4"], ["s2", "s3"]]


def test_layers_with_known_and_unresolved_dependencies() -> None:
    follow_up = plan_of({"goal": GOAL, "steps": [step("n1", "get_incident", {"incident_id": "INC-1042"}, ["s1"])]})
    assert [[s.id for s in layer] for layer in topological_layers(follow_up, known={"s1"})] == [["n1"]]
    with pytest.raises(PlanError, match="unresolved"):
        topological_layers(follow_up)


def test_transitive_dependents() -> None:
    chain = plan_of({"goal": GOAL, "steps": [
        step("a", "get_incident", {"incident_id": "INC-1042"}),
        step("b", "get_incident", {"incident_id": "INC-1051"}, ["a"]),
        step("c", "get_incident", {"incident_id": "INC-1063"}, ["b"]),
        step("d", "search_runbooks", {"query": "x y"}),
    ]})
    assert transitive_dependents(chain, "a") == {"b", "c"}
    assert transitive_dependents(chain, "d") == set()


# ------------------------------------------------------------------ execution --
def test_executor_runs_layers_and_resolves_references() -> None:
    run = PlanExecutor(build_registry(TicketStore())).run(plan_of(good_plan()))
    assert run.status == "succeeded"
    assert run.layers == [["s1", "s4"], ["s2", "s3"]]
    assert json.loads(run.outcomes["s2"].content)  # query_metrics got service resolved from s1's output
    assert run.outcomes["s2"].layer == 1 and run.outcomes["s4"].layer == 0


def test_failure_skips_only_dependents_and_independent_work_continues() -> None:
    run = PlanExecutor(build_registry(TicketStore())).run(plan_of(good_plan("INC-9999")))
    assert run.status == "partial"
    assert run.outcomes["s1"].status == "failed"
    assert run.ids_with("skipped") == ["s2", "s3"]
    assert "dependency 's1' failed" in (run.outcomes["s2"].reason or "")
    assert run.outcomes["s4"].status == "succeeded"
    assert set(run.outputs()) == {"s4"}


def test_invalid_plan_executes_nothing() -> None:
    store = TicketStore()
    data = {"goal": GOAL, "steps": [
        step("s1", "search_runbooks", {"query": "deploy rollback"}),
        step("s2", "create_ticket", {"title": "Add index", "severity": "SEV2", "idempotency_key": "inc-1042-idx"}, ["s1"]),
    ]}
    with pytest.raises(PlanValidationError) as ei:
        PlanExecutor(build_registry(store)).run(plan_of(data))
    assert ei.value.issues[0].code == "tool_not_allowed"
    assert store.executions == 0
    run = PlanExecutor(build_registry(store), allow=("read", "write")).run(plan_of(data))
    assert run.status == "succeeded" and store.executions == 1


def test_bad_reference_is_a_step_failure_not_an_exception() -> None:
    data = {"goal": GOAL, "steps": [
        step("s1", "get_incident", {"incident_id": "INC-1042"}),
        step("s2", "query_metrics", {"service": "${s1.no_such_field}", "metric": "error_rate"}, ["s1"]),
    ]}
    run = PlanExecutor(build_registry(TicketStore())).run(plan_of(data))
    assert run.outcomes["s2"].status == "failed" and run.outcomes["s2"].error_type == "bad_reference"
    assert "no_such_field" in (run.outcomes["s2"].reason or "")


def test_resolve_args_errors() -> None:
    with pytest.raises(PlanReferenceError, match="no output"):
        resolve_args("${s1.x}", {})
    with pytest.raises(PlanReferenceError, match="not valid JSON"):
        resolve_args("${s1.x}", {"s1": "plain text, maybe truncated"})
    assert resolve_args({"k": ["${s1.a.0}", 3]}, {"s1": '{"a": ["v"]}'}) == {"k": ["v", 3]}


# -------------------------------------------------------------------- planner --
def test_planner_repairs_schema_then_semantic_errors() -> None:
    reg = build_registry(TicketStore())
    cyclic = good_plan()
    cyclic["steps"][1]["depends_on"], cyclic["steps"][2]["depends_on"] = ["s1", "s3"], ["s1", "s2"]
    wrong_tool = good_plan()
    wrong_tool["steps"][3]["tool"] = "runbook_search"
    llm = ScriptedLLM([json.dumps(cyclic), json.dumps(wrong_tool), json.dumps(good_plan())])
    result = make_plan(llm, GOAL, reg)
    assert result.plan.ids == ["s1", "s2", "s3", "s4"]
    assert len(llm.calls) == 3 and result.semantic_repairs == 1
    assert "cycle" in llm.calls[1]["messages"][-1].content
    assert "unknown_tool" in llm.calls[2]["messages"][-1].content


def test_planner_gives_up_with_issues_and_hides_disallowed_tools_from_the_prompt() -> None:
    reg = build_registry(TicketStore())
    wrong = good_plan()
    wrong["steps"][3]["tool"] = "runbook_search"
    llm = ScriptedLLM([json.dumps(wrong)] * 3)
    with pytest.raises(PlanValidationError) as ei:
        make_plan(llm, GOAL, reg, max_plan_repairs=2)
    assert ei.value.issues[0].code == "unknown_tool"
    prompt = llm.calls[0]["messages"][1].content  # [0] is the schema message added by chat_structured
    assert "create_ticket" not in prompt and "get_incident" in prompt


# ------------------------------------------------------------------- replanning --
def test_replanning_reuses_finished_work_and_reserves_old_ids() -> None:
    reg = build_registry(TicketStore())
    first = plan_of(good_plan("INC-9999"))
    revised = {"goal": GOAL, "steps": [
        step("r1", "get_incident", {"incident_id": "INC-1042"}),
        step("r2", "query_metrics", {"service": "${r1.service}", "metric": "error_rate"}, ["r1"]),
    ]}
    llm = ScriptedLLM([json.dumps(revised)])
    result = run_with_replanning(llm, reg, GOAL, first)
    assert result.succeeded and result.replans == 1
    assert {"s4", "r1", "r2"} <= set(result.completed) and "s1" not in result.completed
    request = llm.calls[0]["messages"][-1].content
    assert "REPLAN" in request and "step s1 failed" in request and "s4" in request


def test_replan_budget_is_bounded_and_policy_denials_are_not_replanned() -> None:
    reg = build_registry(TicketStore())
    still_bad = {"goal": GOAL, "steps": [step("r1", "get_incident", {"incident_id": "INC-0000"})]}
    llm = ScriptedLLM([json.dumps(still_bad)])
    result = run_with_replanning(llm, reg, GOAL, plan_of(good_plan("INC-9999")), max_replans=1)
    assert not result.succeeded and result.replans == 1 and "budget" in result.stopped_because

    never = ScriptedLLM([])
    denied = {"goal": GOAL, "steps": [step("s1", "search_runbooks", {"query": "x y"})]}
    ok = run_with_replanning(never, reg, GOAL, plan_of(denied))
    assert ok.succeeded and ok.replans == 0 and never.calls == []


# ---------------------------------------------------------------------- verify --
def test_plan_quality_flags_false_dependencies_and_duplicates() -> None:
    data = {"goal": GOAL, "steps": [
        step("a", "search_runbooks", {"query": "deploy rollback"}),
        step("b", "search_runbooks", {"query": "deploy rollback"}, ["a"]),  # duplicate call AND ordering-only edge
        step("c", "get_incident", {"incident_id": "INC-1042"}),
    ]}
    q = plan_quality(plan_of(data), build_registry(TicketStore()))
    assert q.depth == 2 and q.width == 2
    assert q.duplicate_calls == [("a", "b")] and q.ordering_only_edges == [("a", "b")]
    good = plan_quality(plan_of(good_plan()))
    assert good.ordering_only_edges == [] and (good.depth, good.width) == (2, 2)


def test_citation_check_flags_ids_absent_from_tool_output() -> None:
    outputs = {"s1": '{"id": "INC-1042", "related_docs": ["rb-deploy-rollback"]}'}
    report = verify_citations("See INC-1042, rb-deploy-rollback and INC-1099.", outputs)
    assert report.unknown == ["INC-1099"] and not report.grounded
    assert verify_citations("See INC-1042.", outputs).grounded
