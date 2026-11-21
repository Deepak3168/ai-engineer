"""Single source of truth for the 28-day curriculum, and README table generator.

    python scripts/curriculum.py            # rewrite the table inside README.md
    python scripts/curriculum.py --check    # exit 1 if a lesson file is missing or README is stale
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRACKS: list[tuple[str, list[tuple[int, str, str]]]] = [
    ("Foundations", [
        (1, "llm-apis-and-structured-outputs", "LLM APIs, messages, tokens, and structured outputs"),
        (2, "tool-calling-and-function-execution", "Function calling and tool execution"),
        (3, "agent-loops-and-react", "Agent loops, ReAct, and autonomous execution"),
        (4, "planning-and-task-decomposition", "Planning, task decomposition, and dependency management"),
        (5, "agent-state-and-checkpoints", "Agent state, checkpoints, and execution lifecycle"),
        (6, "short-term-and-long-term-memory", "Short-term memory and long-term memory"),
        (7, "context-engineering-and-token-budgets", "Context engineering, token budgets, and conversation management"),
    ]),
    ("Knowledge and tools", [
        (8, "embeddings-and-semantic-search", "Embeddings and semantic search"),
        (9, "rag-architecture-and-retrieval-pipelines", "RAG architecture and retrieval pipelines"),
        (10, "reranking-hybrid-search-and-retrieval-evaluation", "Reranking, hybrid search, and retrieval evaluation"),
        (11, "mcp-architecture-servers-and-clients", "MCP architecture, servers, clients, and tool discovery"),
        (12, "production-ready-mcp-tools", "Building production-ready MCP tools"),
    ]),
    ("Agent runtime and orchestration", [
        (13, "framework-independent-agent-runtime", "Designing a framework-independent agent runtime"),
        (14, "workflow-orchestration-and-state-graphs", "Workflow orchestration and state graphs"),
        (15, "multi-agent-architecture-and-communication", "Multi-agent architecture and communication patterns"),
        (16, "delegation-parallel-execution-and-shared-workspaces", "Delegation, parallel execution, and shared workspaces"),
        (17, "human-in-the-loop-approvals", "Human-in-the-loop approvals and intervention"),
    ]),
    ("Production engineering", [
        (18, "agent-security-permissions-and-prompt-injection", "Agent security, permissions, and prompt injection"),
        (19, "guardrails-validation-and-policy-enforcement", "Guardrails, validation, and policy enforcement"),
        (20, "evaluation-methodology-and-benchmark-design", "Evaluation methodology and benchmark design"),
        (21, "tracing-observability-and-debugging", "Tracing, observability, and debugging"),
        (22, "testing-agents-and-deterministic-simulations", "Testing agents and deterministic simulations"),
        (23, "reliability-retries-timeouts-and-idempotency", "Reliability, retries, timeouts, and idempotency"),
        (24, "deployment-scaling-and-infrastructure", "Deployment, scaling, and infrastructure considerations"),
    ]),
    ("Capstone", [
        (25, "capstone-architecture-and-requirements", "Capstone architecture and requirements"),
        (26, "capstone-agent-runtime-and-tools", "Implementing the agent runtime and tools"),
        (27, "capstone-memory-retrieval-and-orchestration", "Implementing memory, retrieval, and orchestration"),
        (28, "capstone-deployment-and-review", "Testing, hardening, deployment, and final documentation"),
    ]),
]
BEGIN, END = "<!-- curriculum:begin -->", "<!-- curriculum:end -->"


def filename(day: int, slug: str) -> str:
    return f"day-{day:02d}-{slug}.md"


def render() -> str:
    out: list[str] = []
    done = total = 0
    for track, items in TRACKS:
        out += [f"### {track}", "", "| Day | Lesson | Status |", "|---:|---|:---:|"]
        for day, slug, title in items:
            total += 1
            exists = (ROOT / "lessons" / filename(day, slug)).is_file()
            done += exists
            link = f"[{title}](lessons/{filename(day, slug)})" if exists else title
            out.append(f"| {day} | {link} | {'published' if exists else 'planned'} |")
        out.append("")
    out.insert(0, f"**{done} of {total} lessons published.**\n")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        print("README.md has no curriculum markers", file=sys.stderr)
        return 1
    new = pattern.sub(lambda _m: f"{BEGIN}\n{render()}{END}", text)
    if "--check" in sys.argv:
        return 0 if new == text else 1
    readme.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
