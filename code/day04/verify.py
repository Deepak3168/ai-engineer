"""Judging plans and their results without a model: static quality metrics and citation checking.

These are cheap, deterministic checks. They cannot tell you a plan is *good*, only that it is not
obviously wasteful (false dependencies serialise work) or that a written report cites only evidence
the run actually retrieved.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping

from pydantic import BaseModel, Field

from common.llm import LLMClient, Usage, system, user
from common.tools import ToolRegistry
from day04.plan import Plan, step_refs, topological_layers

ID_RE = re.compile(r"\bINC-\d{4}\b|\brb-[a-z0-9]+(?:-[a-z0-9]+)*\b")


class PlanQuality(BaseModel):
    steps: int
    depth: int  # number of layers = length of the critical path in steps
    width: int  # largest layer = best-case parallelism
    side_effect_steps: list[str] = Field(default_factory=list)
    duplicate_calls: list[tuple[str, str]] = Field(default_factory=list)
    ordering_only_edges: list[tuple[str, str]] = Field(default_factory=list)


def plan_quality(plan: Plan, registry: ToolRegistry | None = None, known: Collection[str] = ()) -> PlanQuality:
    layers = topological_layers(plan, known)
    seen: dict[str, str] = {}
    dupes: list[tuple[str, str]] = []
    for s in plan.steps:
        key = f"{s.tool}:{sorted(s.args.items(), key=lambda kv: kv[0])}"
        if key in seen:
            dupes.append((seen[key], s.id))
        seen.setdefault(key, s.id)
    # An edge is "ordering only" when the dependent never reads the dependency's output. Sometimes legitimate
    # (do X after Y), but each one removes parallelism, so a reviewer should look at them.
    ordering = [(d, s.id) for s in plan.steps for d in s.depends_on if d not in step_refs(s) and d not in known]
    writes = [s.id for s in plan.steps if registry and (t := registry.get(s.tool)) and t.side_effect != "read"]
    return PlanQuality(steps=len(plan.steps), depth=len(layers), width=max(len(layer) for layer in layers),
                       side_effect_steps=writes, duplicate_calls=dupes, ordering_only_edges=ordering)


class CitationReport(BaseModel):
    cited: list[str]
    unknown: list[str]  # cited ids that appear in no tool output of this run: fabricated or stale

    @property
    def grounded(self) -> bool:
        return not self.unknown


def verify_citations(text: str, outputs: Mapping[str, str]) -> CitationReport:
    evidence = {m for out in outputs.values() for m in ID_RE.findall(out)}
    cited = sorted(set(ID_RE.findall(text)))
    return CitationReport(cited=cited, unknown=[c for c in cited if c not in evidence])


class Synthesis(BaseModel):
    text: str
    citations: CitationReport
    usage: Usage = Field(default_factory=Usage)


def synthesize(client: LLMClient, goal: str, outputs: Mapping[str, str], *, per_step_chars: int = 1500) -> Synthesis:
    """One model call that writes the deliverable from step outputs, then a mechanical citation check."""
    evidence = "\n\n".join(f"[{sid}]\n{out[:per_step_chars]}" for sid, out in outputs.items())
    resp = client.chat([
        system("Write the requested document using ONLY the evidence below. Cite incident and runbook ids "
               "exactly as they appear. If evidence is missing for a section, say so instead of guessing."),
        user(f"Goal: {goal}\n\nEvidence from tool steps:\n{evidence}"),
    ])
    return Synthesis(text=resp.content, citations=verify_citations(resp.content, outputs), usage=resp.usage)
