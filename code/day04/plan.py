"""A plan is data: a validated DAG of tool steps, not a paragraph of intent.

Validation is split on purpose:
  * intrinsic checks live in `Plan` itself (duplicate ids, self-dependency, cycles, undeclared references),
    so `common.structured.chat_structured` feeds them back to the model through its normal repair loop;
  * checks that need the outside world (missing dependencies given already-completed work, unknown tools,
    bad arguments, side-effect policy) live in `validate_plan`, which the planner calls after parsing.

Data flows between steps through references: an argument value that is exactly the string "${s1.service}"
is replaced at run time by the `service` field of step `s1`'s JSON output. Every referenced step must also be
listed in `depends_on`: the dependency graph is the *only* thing that orders execution.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from common.tools import SideEffect, ToolRegistry

MAX_STEPS = 12
REF_RE = re.compile(r"^\$\{([a-z][a-z0-9_]*)((?:\.[A-Za-z0-9_]+)*)\}$")


class PlanError(ValueError):
    """The plan cannot be ordered (cycle or unresolved dependency)."""


class PlanIssue(BaseModel):
    code: str  # duplicate_id | self_dependency | cycle | missing_dependency | undeclared_ref | unknown_tool | bad_args | tool_not_allowed
    step_id: str | None = None
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$", description="Unique step id, e.g. s1 or fetch_incident")
    tool: str = Field(description="Name of one registered tool")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments; may contain '${step_id.field}' references")
    depends_on: list[str] = Field(default_factory=list, description="Ids of steps that must succeed first")
    purpose: str = Field(min_length=3, max_length=200, description="Why this step exists, in one sentence")


def find_refs(value: Any) -> list[tuple[str, str]]:
    """All (step_id, dotted_path) references anywhere inside an args structure."""
    if isinstance(value, str):
        m = REF_RE.match(value)
        return [(m.group(1), m.group(2).lstrip("."))] if m else []
    if isinstance(value, dict):
        return [r for v in value.values() for r in find_refs(v)]
    if isinstance(value, list):
        return [r for v in value for r in find_refs(v)]
    return []


def step_refs(step: PlanStep) -> set[str]:
    return {sid for sid, _ in find_refs(step.args)}


def _find_cycle(steps: Iterable[PlanStep]) -> list[str] | None:
    graph = {s.id: [d for d in s.depends_on] for s in steps}
    state: dict[str, int] = {}  # 1 = visiting, 2 = done
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in graph:
                continue  # missing deps are reported elsewhere
            if state.get(dep) == 1:
                return [*stack[stack.index(dep):], dep]
            if state.get(dep) is None and (cyc := visit(dep)):
                return cyc
        stack.pop()
        state[node] = 2
        return None

    for node in graph:
        if state.get(node) is None and (cyc := visit(node)):
            return cyc
    return None


def intrinsic_issues(steps: list[PlanStep]) -> list[PlanIssue]:
    issues: list[PlanIssue] = []
    seen: set[str] = set()
    for s in steps:
        if s.id in seen:
            issues.append(PlanIssue(code="duplicate_id", step_id=s.id, message=f"step id '{s.id}' is used more than once"))
        seen.add(s.id)
        if s.id in s.depends_on:
            issues.append(PlanIssue(code="self_dependency", step_id=s.id, message=f"step '{s.id}' depends on itself"))
        for ref in sorted(step_refs(s) - set(s.depends_on)):
            issues.append(PlanIssue(code="undeclared_ref", step_id=s.id,
                                    message=f"step '{s.id}' references '${{{ref}...}}' but does not list '{ref}' in depends_on"))
    if not any(i.code == "duplicate_id" for i in issues) and (cycle := _find_cycle(steps)):
        issues.append(PlanIssue(code="cycle", step_id=cycle[0], message="dependency cycle: " + " -> ".join(cycle)))
    return issues


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=3, max_length=300)
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_STEPS)

    @model_validator(mode="after")
    def _structure_is_sound(self) -> Plan:
        issues = intrinsic_issues(self.steps)
        if issues:
            raise ValueError("; ".join(str(i) for i in issues))
        return self

    def step(self, step_id: str) -> PlanStep:
        return next(s for s in self.steps if s.id == step_id)

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.steps]


def parse_plan(data: dict[str, Any]) -> Plan | list[PlanIssue]:
    """Validate raw data; return the Plan or the list of intrinsic issues (never raises on bad plans)."""
    try:
        return Plan.model_validate(data)
    except ValidationError as exc:
        return [PlanIssue(code="schema", message=e["msg"].removeprefix("Value error, ")) for e in exc.errors()]


def validate_plan(
    plan: Plan,
    registry: ToolRegistry | None = None,
    *,
    known: Collection[str] = (),
    reserved: Collection[str] = (),
    allow: Collection[SideEffect] = ("read",),
) -> list[PlanIssue]:
    """Everything that needs context beyond the plan itself.

    known     ids of steps that already succeeded (their outputs are available, so depending on them is fine)
    reserved  ids used by earlier plans; a replan must not reuse them
    allow     side-effect classes this run may execute; a disallowed tool is rejected *before anything runs*
    """
    issues = intrinsic_issues(plan.steps)
    for s in plan.steps:
        if s.id in reserved or s.id in known:
            issues.append(PlanIssue(code="duplicate_id", step_id=s.id, message=f"step id '{s.id}' was used by an earlier plan"))
        for dep in s.depends_on:
            if dep not in plan.ids and dep not in known:
                issues.append(PlanIssue(code="missing_dependency", step_id=s.id,
                                        message=f"step '{s.id}' depends on '{dep}', which does not exist"))
        if registry is None:
            continue
        tool = registry.get(s.tool)
        if tool is None:
            issues.append(PlanIssue(code="unknown_tool", step_id=s.id,
                                    message=f"step '{s.id}' uses unknown tool '{s.tool}'; available: {', '.join(registry.names())}"))
            continue
        if tool.side_effect not in allow:
            issues.append(PlanIssue(code="tool_not_allowed", step_id=s.id,
                                    message=f"step '{s.id}' uses '{s.tool}' (side effect '{tool.side_effect}') which this run may not execute"))
        if not find_refs(s.args):  # with references the arguments are only complete at run time
            try:
                tool.args_model.model_validate(s.args)
            except ValidationError as exc:
                detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:4])
                issues.append(PlanIssue(code="bad_args", step_id=s.id, message=f"step '{s.id}' has invalid args for '{s.tool}': {detail}"))
    return issues


def topological_layers(plan: Plan, known: Collection[str] = ()) -> list[list[PlanStep]]:
    """Kahn's algorithm, grouped into layers: every step in a layer depends only on earlier layers.

    Steps in one layer are mutually independent, so they may run in parallel. Order inside a layer is the
    plan's declaration order, which keeps runs deterministic.
    """
    remaining = {s.id: {d for d in s.depends_on if d not in known} for s in plan.steps}
    unknown = {d for deps in remaining.values() for d in deps} - set(remaining)
    if unknown:
        raise PlanError(f"unresolved dependencies: {sorted(unknown)}")
    layers: list[list[PlanStep]] = []
    done: set[str] = set()
    while remaining:
        ready = [s for s in plan.steps if s.id in remaining and remaining[s.id] <= done]
        if not ready:
            raise PlanError(f"dependency cycle among {sorted(remaining)}")
        layers.append(ready)
        for s in ready:
            done.add(s.id)
            del remaining[s.id]
    return layers


def transitive_dependents(plan: Plan, step_id: str) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for s in plan.steps:
        for d in s.depends_on:
            children[d].add(s.id)
    out: set[str] = set()
    todo = [step_id]
    while todo:
        for child in children[todo.pop()]:
            if child not in out:
                out.add(child)
                todo.append(child)
    return out
