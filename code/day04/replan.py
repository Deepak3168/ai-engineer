"""Replanning: when a plan partly fails, ask for a *revised* plan for the remaining work.

What carries over: outputs of steps that already succeeded (the new plan may depend on them by id).
What does not: failed and skipped steps are superseded, and their ids are reserved so the trace stays unambiguous.
Replanning is bounded (`max_replans`) and never attempted for policy denials, which no new plan can fix.
"""

from __future__ import annotations

from collections.abc import Collection

from pydantic import BaseModel, Field

from common.llm import LLMClient
from common.tools import SideEffect, ToolRegistry
from day04.executor import PlanExecutor, PlanRun
from day04.plan import Plan
from day04.planner import make_plan


class ReplanResult(BaseModel):
    plans: list[Plan] = Field(default_factory=list)
    runs: list[PlanRun] = Field(default_factory=list)
    completed: dict[str, str] = Field(default_factory=dict)
    replans: int = 0
    stopped_because: str = "succeeded"

    @property
    def succeeded(self) -> bool:
        return bool(self.runs) and self.runs[-1].status == "succeeded"


def describe_failures(run: PlanRun) -> list[str]:
    lines = []
    for sid in run.ids_with("failed"):
        o = run.outcomes[sid]
        lines.append(f"step {sid} failed ({o.error_type}): {(o.reason or '')[:240]}")
    lines += [f"step {sid} was skipped: {run.outcomes[sid].reason}" for sid in run.ids_with("skipped")]
    return lines


def run_with_replanning(
    client: LLMClient,
    registry: ToolRegistry,
    goal: str,
    plan: Plan,
    *,
    allow: Collection[SideEffect] = ("read",),
    max_replans: int = 1,
) -> ReplanResult:
    executor = PlanExecutor(registry, allow=allow)
    result = ReplanResult()
    used_ids: set[str] = set()
    current = plan
    while True:
        run = executor.run(current, completed=result.completed)
        result.plans.append(current)
        result.runs.append(run)
        used_ids |= set(current.ids)
        result.completed.update(run.outputs())
        if run.status == "succeeded":
            return result
        if any(o.error_type == "denied" for o in run.outcomes.values()):
            result.stopped_because = "policy denial: replanning cannot fix it"
            return result
        if result.replans >= max_replans:
            result.stopped_because = f"replan budget ({max_replans}) exhausted"
            return result
        result.replans += 1
        current = make_plan(client, goal, registry, allow=allow, known_outputs=result.completed,
                            reserved=used_ids, failures=describe_failures(run)).plan
