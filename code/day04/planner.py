"""Turn a goal into a validated `Plan` with a model, using two nested repair loops.

  inner  (`chat_structured`)   schema + intrinsic DAG errors  -> fed back by common.structured
  outer  (`make_plan`)         unknown tools, bad args, missing deps, policy -> fed back here

A plan that still fails after the repair budget raises `PlanValidationError`: executing a plan nobody
validated is how a typo in a tool name becomes a half-finished incident response.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from pydantic import BaseModel, Field

from common.llm import LLMClient, PermanentLLMError, Usage, assistant, system, user
from common.structured import chat_structured
from common.tools import SideEffect, ToolRegistry
from day04.plan import MAX_STEPS, Plan, PlanIssue, validate_plan


class PlanValidationError(PermanentLLMError):
    def __init__(self, issues: list[PlanIssue]) -> None:
        super().__init__("invalid plan: " + "; ".join(str(i) for i in issues))
        self.issues = issues


class PlanResult(BaseModel):
    plan: Plan
    usage: Usage = Field(default_factory=Usage)
    semantic_repairs: int = 0


def planner_prompt(registry: ToolRegistry, allow: Collection[SideEffect]) -> str:
    tools = "\n".join(
        f"- {t.name} [{t.side_effect}]: {t.description}"
        for t in (registry.get(n) for n in registry.names())
        if t is not None and t.side_effect in allow
    )
    return (
        "You are the planning module of an ops assistant. Produce a plan as JSON; do not execute anything.\n"
        f"Available tools (use ONLY these):\n{tools}\n"
        f"Rules: at most {MAX_STEPS} steps; ids like s1, s2; list in depends_on every step whose output you need or that must "
        "happen first; steps with no dependency between them run in parallel, so do not add dependencies you do not need. "
        "To use an earlier step's output write the argument as the exact string '${step_id.field}' (JSON object outputs only) "
        "and list that step in depends_on. Do not plan side-effecting tools unless the goal asks for them."
    )


def make_plan(
    client: LLMClient,
    goal: str,
    registry: ToolRegistry,
    *,
    allow: Collection[SideEffect] = ("read",),
    known_outputs: Mapping[str, str] | None = None,
    reserved: Collection[str] = (),
    failures: list[str] | None = None,
    max_plan_repairs: int = 2,
    max_schema_repairs: int = 2,
) -> PlanResult:
    known = dict(known_outputs or {})
    request = f"Goal: {goal}"
    if known or failures:
        request += "\nThis is a REPLAN. Do not repeat finished work.\n"
        if known:
            request += "Finished steps you may depend on (id: output preview):\n" + "\n".join(
                f"- {i}: {out[:200]}" for i, out in known.items()) + "\n"
        if failures:
            request += "Problems from the previous attempt:\n" + "\n".join(f"- {f}" for f in failures) + "\n"
        request += "Use new step ids that do not collide with: " + ", ".join(sorted({*known, *reserved})) + "."
    messages = [system(planner_prompt(registry, allow)), user(request)]

    total, repairs = Usage(), 0
    for attempt in range(max_plan_repairs + 1):
        plan, usage = chat_structured(client, messages, Plan, max_repairs=max_schema_repairs)
        total.prompt_tokens += usage.prompt_tokens
        total.completion_tokens += usage.completion_tokens
        issues = validate_plan(plan, registry, known=known.keys(), reserved=reserved, allow=allow)
        if not issues:
            return PlanResult(plan=plan, usage=total, semantic_repairs=repairs)
        if attempt == max_plan_repairs:
            raise PlanValidationError(issues)
        repairs += 1
        messages += [
            assistant(plan.model_dump_json()),
            user("That plan cannot run:\n" + "\n".join(f"- {i}" for i in issues) + "\nReturn the corrected plan."),
        ]
    raise AssertionError("unreachable")  # pragma: no cover
