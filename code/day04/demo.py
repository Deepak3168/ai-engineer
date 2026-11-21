"""Day 4 demo: plan "draft a postmortem for INC-1042", validate it, run it, break it, replan it. Offline.

    PYTHONPATH=code LESSON_OFFLINE=1 python -m day04.demo

The planner and writer "models" are scripts, so this shows the machinery around a plan - validation,
layering, parallel execution, failure propagation, replanning, citation checking - not how good a real
model's plans are. With LLM_API_KEY set (and LESSON_OFFLINE unset) sections 1, 2 and 5 call a real model.
"""

from __future__ import annotations

import json

from day02.ops_tools import TicketStore, build_registry

from common.llm import get_client
from day04.executor import PlanExecutor
from day04.plan import Plan, topological_layers
from day04.planner import PlanValidationError, make_plan
from day04.replan import run_with_replanning
from day04.verify import plan_quality, synthesize

GOAL = "Draft a postmortem for INC-1042"


def step(sid: str, tool: str, args: dict, deps: list[str], purpose: str) -> dict:
    return {"id": sid, "tool": tool, "args": args, "depends_on": deps, "purpose": purpose}


def plan_json(incident: str = "INC-1042") -> str:
    return json.dumps({"goal": GOAL, "steps": [
        step("s1", "get_incident", {"incident_id": incident}, [], "Load the incident record"),
        step("s2", "query_metrics", {"service": "${s1.service}", "metric": "error_rate"}, ["s1"], "Quantify error-rate impact"),
        step("s3", "query_metrics", {"service": "${s1.service}", "metric": "p95_latency_ms"}, ["s1"], "Quantify latency impact"),
        step("s4", "search_runbooks", {"query": "deploy rollback"}, [], "Find the rollback procedure"),
        step("s5", "search_runbooks", {"query": "postmortem template"}, [], "Find the postmortem structure"),
    ]})


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def show_run(run) -> None:
    for sid, o in run.outcomes.items():
        note = f"  <- {o.reason[:70]}" if o.reason else ""
        print(f"  layer {o.layer} {sid}: {o.status}{note}")


def main() -> None:
    store = TicketStore()
    reg = build_registry(store)

    banner("1. Planning: two kinds of repair before a plan is accepted")
    cyclic = json.loads(plan_json())
    cyclic["steps"][1]["depends_on"], cyclic["steps"][2]["depends_on"] = ["s1", "s3"], ["s1", "s2"]
    wrong_tool = json.loads(plan_json())
    wrong_tool["steps"][3]["tool"] = "runbook_search"
    wrong_tool["steps"][4]["depends_on"] = ["s9"]
    client = get_client([json.dumps(cyclic), json.dumps(wrong_tool), plan_json()])
    planned = make_plan(client, GOAL, reg)
    plan = planned.plan
    if hasattr(client, "calls"):
        print("model calls:", len(client.calls), "| semantic repairs:", planned.semantic_repairs)
        print("feedback after call 1 (schema/DAG):", client.calls[1]["messages"][-1].content.splitlines()[1][:140])
        print("feedback after call 2 (semantic)  :", " | ".join(client.calls[2]["messages"][-1].content.splitlines()[1:3])[:210])
    layers = topological_layers(plan)
    print("layers:", [[s.id for s in layer] for layer in layers])
    print("quality:", plan_quality(plan, reg).model_dump())

    banner("2. Executing: parallel inside a layer, references resolved from earlier outputs")
    run = PlanExecutor(reg).run(plan)
    show_run(run)
    print("s2 args were resolved to service =", json.loads(run.outcomes["s1"].content)["service"], "| run status:", run.status)
    writer = get_client([
        "Postmortem INC-1042 (SEV1, checkout-api). Impact: error rate peaked at 0.31 (metrics s2). Root cause: unindexed "
        "query after release. Actions: follow rb-deploy-rollback and rb-db-connection-pool. Related: INC-1099."])
    draft = synthesize(writer, GOAL, run.outputs())
    print("citations:", draft.citations.cited, "| not found in any tool output:", draft.citations.unknown)

    banner("3. Failure propagation: INC-9999 does not exist")
    bad_plan = Plan.model_validate_json(plan_json("INC-9999"))
    bad_run = PlanExecutor(reg).run(bad_plan)
    show_run(bad_run)
    print("run status:", bad_run.status)

    banner("4. Replanning from the failure, reusing finished work")
    fix = json.dumps({"goal": GOAL, "steps": [
        step("r1_inc", "get_incident", {"incident_id": "INC-1042"}, [], "Load the correct incident record"),
        step("r1_err", "query_metrics", {"service": "${r1_inc.service}", "metric": "error_rate"}, ["r1_inc"], "Error-rate impact"),
    ]})
    rc = get_client([fix])
    out = run_with_replanning(rc, reg, GOAL, bad_plan)
    print("replans:", out.replans, "| final run:")
    show_run(out.runs[-1])
    print("reused from run 1 without re-executing:", sorted(set(out.completed) & set(out.runs[0].ids_with('succeeded'))))
    if hasattr(rc, "calls"):
        print("replanner was told:", rc.calls[0]["messages"][-1].content.split("Problems from the previous attempt:")[1].strip().splitlines()[0][:120])

    banner("5. Policy is checked before anything runs")
    risky = Plan.model_validate_json(json.dumps({"goal": GOAL, "steps": [
        step("s1", "get_incident", {"incident_id": "INC-1042"}, [], "Load the incident record"),
        step("s2", "create_ticket", {"title": "Add index on orders", "severity": "SEV2", "idempotency_key": "inc-1042-index"},
             ["s1"], "File the follow-up ticket"),
    ]}))
    try:
        PlanExecutor(reg).run(risky)
    except PlanValidationError as exc:
        print("rejected:", exc.issues[0])
    print("tickets created:", len(store.tickets), "| read-only steps executed: 0")
    allowed = PlanExecutor(reg, allow=("read", "write")).run(risky)
    print("with write allowed:", allowed.status, "| tickets created:", len(store.tickets))


if __name__ == "__main__":
    main()
