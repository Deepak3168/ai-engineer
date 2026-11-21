"""Run a validated plan layer by layer: parallel inside a layer, failures propagate to dependents only.

Semantics (the part worth reading closely):
  * a step runs only if every dependency *succeeded*; otherwise it is `skipped` and says which dependency broke it;
  * independent branches keep running after a failure - one bad lookup does not discard unrelated work;
  * references like "${s1.service}" are resolved from the JSON output of the step they name, and a bad
    reference is a step failure (`bad_reference`), not an exception;
  * the plan is validated first: an invalid plan executes nothing.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any, Literal

from day02.execution import execute_batch
from pydantic import BaseModel, Field

from common.llm import ToolCall
from common.tools import SideEffect, ToolRegistry
from day04.plan import REF_RE, Plan, PlanStep, topological_layers, validate_plan
from day04.planner import PlanValidationError

StepStatus = Literal["succeeded", "failed", "skipped"]


class StepOutcome(BaseModel):
    step_id: str
    status: StepStatus
    content: str = ""
    error_type: str | None = None
    reason: str | None = None
    layer: int = 0


class PlanRun(BaseModel):
    outcomes: dict[str, StepOutcome] = Field(default_factory=dict)
    layers: list[list[str]] = Field(default_factory=list)

    @property
    def status(self) -> Literal["succeeded", "partial", "failed"]:
        statuses = [o.status for o in self.outcomes.values()]
        if all(s == "succeeded" for s in statuses):
            return "succeeded"
        return "partial" if any(s == "succeeded" for s in statuses) else "failed"

    def ids_with(self, status: StepStatus) -> list[str]:
        return [i for i, o in self.outcomes.items() if o.status == status]

    def outputs(self) -> dict[str, str]:
        return {i: o.content for i, o in self.outcomes.items() if o.status == "succeeded"}


class PlanReferenceError(ValueError):
    """A ${step.path} reference could not be resolved."""


def _lookup(data: Any, path: str, ref: str) -> Any:
    for part in [p for p in path.split(".") if p]:
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdigit() and int(part) < len(data):
            data = data[int(part)]
        else:
            raise PlanReferenceError(f"{ref}: no field '{part}' in the output")
    return data


def resolve_args(value: Any, outputs: Mapping[str, str]) -> Any:
    """Replace whole-string references with values from earlier outputs (which must be JSON)."""
    if isinstance(value, str) and (m := REF_RE.match(value)):
        step_id, path = m.group(1), m.group(2).lstrip(".")
        if step_id not in outputs:
            raise PlanReferenceError(f"{value}: step '{step_id}' has no output")
        try:
            parsed = json.loads(outputs[step_id])
        except json.JSONDecodeError:
            raise PlanReferenceError(f"{value}: output of '{step_id}' is not valid JSON (truncated or plain text?)") from None
        return _lookup(parsed, path, value)
    if isinstance(value, dict):
        return {k: resolve_args(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_args(v, outputs) for v in value]
    return value


class PlanExecutor:
    def __init__(self, registry: ToolRegistry, *, allow: Collection[SideEffect] = ("read",), max_workers: int = 4) -> None:
        self.registry, self.allow, self.max_workers = registry, tuple(allow), max_workers

    def run(self, plan: Plan, completed: Mapping[str, str] | None = None) -> PlanRun:
        """`completed` maps ids of already-succeeded steps (from an earlier plan) to their outputs."""
        done = dict(completed or {})
        issues = validate_plan(plan, self.registry, known=done.keys(), allow=self.allow)
        if issues:
            raise PlanValidationError(issues)

        layers = topological_layers(plan, known=done.keys())
        run = PlanRun(layers=[[s.id for s in layer] for layer in layers])
        outputs = dict(done)
        for depth, layer in enumerate(layers):
            runnable: list[tuple[PlanStep, dict[str, Any]]] = []
            for step in layer:
                broken = next((d for d in step.depends_on if d in run.outcomes and run.outcomes[d].status != "succeeded"), None)
                if broken:
                    run.outcomes[step.id] = StepOutcome(
                        step_id=step.id, status="skipped", layer=depth,
                        reason=f"dependency '{broken}' {run.outcomes[broken].status}")
                    continue
                try:
                    runnable.append((step, resolve_args(step.args, outputs)))
                except PlanReferenceError as exc:
                    run.outcomes[step.id] = StepOutcome(step_id=step.id, status="failed", layer=depth,
                                                        error_type="bad_reference", reason=str(exc))
            calls = [ToolCall(id=s.id, name=s.tool, arguments=args) for s, args in runnable]
            for (step, _), result in zip(runnable, execute_batch(self.registry, calls, allow=self.allow,
                                                                  max_workers=self.max_workers), strict=True):
                run.outcomes[step.id] = StepOutcome(
                    step_id=step.id, status="succeeded" if result.ok else "failed", content=result.content,
                    error_type=result.error_type, reason=None if result.ok else result.content, layer=depth)
                if result.ok:
                    outputs[step.id] = result.content
        return run
