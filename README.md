# AI Agent Engineering: a 28-day, first-principles handbook

A hands-on course that takes an experienced Python backend developer from "an LLM can request a function call" to designing, evaluating, securing and deploying a production-style agent system. Every concept is built from scratch in framework-independent Python first. Frameworks (LangGraph, the MCP SDK) appear only after you understand the mechanism they abstract.

All lesson code is real: it is linted, unit-tested, and executed offline by a deterministic scripted model, so you can run everything without an API key. The same code switches to a live model by setting environment variables.

## Who this is for

Developers who already ship backend software and want to understand agents from the inside rather than memorise a framework's API. The course assumes you do **not** need Python, HTTP or SQL explained, and spends its time on the agent-specific parts: tool execution, state, memory, retrieval, orchestration, security, evaluation and operations.

## Prerequisites

- Python 3.12+ (developed and tested on 3.13), comfortable with type hints, `asyncio` basics and `pytest`.
- SQL and PostgreSQL; REST APIs and FastAPI or Django; Redis basics; Docker basics.
- Optional: an API key for any OpenAI-compatible endpoint. Not required to run anything.

## Curriculum

Each lesson is roughly 30-45 minutes and follows the same 15-section structure: objectives, why it matters, mental model, architecture, core concepts, implementation, complete code, design decisions, failure scenarios, production considerations, exercises, a challenge, knowledge check, references, and a self-assessment.

The table below is generated from the files in `lessons/` by `scripts/curriculum.py`, so a lesson is linked only once it exists.

<!-- curriculum:begin -->
**4 of 28 lessons published.**

### Foundations

| Day | Lesson | Status |
|---:|---|:---:|
| 1 | [LLM APIs, messages, tokens, and structured outputs](lessons/day-01-llm-apis-and-structured-outputs.md) | published |
| 2 | [Function calling and tool execution](lessons/day-02-tool-calling-and-function-execution.md) | published |
| 3 | [Agent loops, ReAct, and autonomous execution](lessons/day-03-agent-loops-and-react.md) | published |
| 4 | [Planning, task decomposition, and dependency management](lessons/day-04-planning-and-task-decomposition.md) | published |
| 5 | Agent state, checkpoints, and execution lifecycle | planned |
| 6 | Short-term memory and long-term memory | planned |
| 7 | Context engineering, token budgets, and conversation management | planned |

### Knowledge and tools

| Day | Lesson | Status |
|---:|---|:---:|
| 8 | Embeddings and semantic search | planned |
| 9 | RAG architecture and retrieval pipelines | planned |
| 10 | Reranking, hybrid search, and retrieval evaluation | planned |
| 11 | MCP architecture, servers, clients, and tool discovery | planned |
| 12 | Building production-ready MCP tools | planned |

### Agent runtime and orchestration

| Day | Lesson | Status |
|---:|---|:---:|
| 13 | Designing a framework-independent agent runtime | planned |
| 14 | Workflow orchestration and state graphs | planned |
| 15 | Multi-agent architecture and communication patterns | planned |
| 16 | Delegation, parallel execution, and shared workspaces | planned |
| 17 | Human-in-the-loop approvals and intervention | planned |

### Production engineering

| Day | Lesson | Status |
|---:|---|:---:|
| 18 | Agent security, permissions, and prompt injection | planned |
| 19 | Guardrails, validation, and policy enforcement | planned |
| 20 | Evaluation methodology and benchmark design | planned |
| 21 | Tracing, observability, and debugging | planned |
| 22 | Testing agents and deterministic simulations | planned |
| 23 | Reliability, retries, timeouts, and idempotency | planned |
| 24 | Deployment, scaling, and infrastructure considerations | planned |

### Capstone

| Day | Lesson | Status |
|---:|---|:---:|
| 25 | Capstone architecture and requirements | planned |
| 26 | Implementing the agent runtime and tools | planned |
| 27 | Implementing memory, retrieval, and orchestration | planned |
| 28 | Testing, hardening, deployment, and final documentation | planned |
<!-- curriculum:end -->

## Learning outcomes

By the end you should be able to:

- Explain the exact request/response protocol behind tool calling and implement the validate-execute-observe contract yourself.
- Build an agent loop with budgets, stall detection and recoverable failures, and decide when *not* to use autonomy.
- Persist agent state so a crashed run resumes without repeating side effects.
- Design memory and context budgets deliberately instead of appending messages until the window fills.
- Build and evaluate a retrieval pipeline (chunking, pgvector, BM25, hybrid fusion, reranking) with proper metrics.
- Build and test MCP servers and clients, and expose their tools to your own runtime.
- Design a framework-independent runtime, and map it onto state graphs and multi-agent patterns.
- Apply least privilege, taint tracking and approval gates against prompt injection, and state honestly what residual risk remains.
- Evaluate agents with confidence intervals, trace them, test them deterministically, make them reliable, and deploy them.

## Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.12+ | Type hints and Pydantic v2 throughout |
| LLM access | Any OpenAI-compatible endpoint | Configured by `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`; NVIDIA-hosted endpoints work the same way |
| Validation | Pydantic | Tool arguments, plans, structured outputs, state |
| Service layer | FastAPI | Days 12, 24 and the capstone |
| Database | PostgreSQL 16 + pgvector | State, vectors, full-text, approvals |
| Cache and queue | Redis | Session memory, idempotency, job queue |
| Protocol | MCP (Python SDK 2.x) | Note: 2.x renamed `FastMCP` to `MCPServer` |
| Orchestration | LangGraph 1.x | Introduced only after a from-scratch state graph |
| Packaging | Docker, docker compose | `docker-compose.yml` provides Postgres+pgvector and Redis |

No secrets are stored in the repository. Credentials come from the environment (see `.env.example`).

## How to use this repository

```bash
git clone https://github.com/Deepak3168/ai-engineer.git
cd ai-engineer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add a key to use a live model
docker compose up -d            # Postgres+pgvector and Redis (needed from Day 5 on)
export DATABASE_URL=postgresql://agent:agent@localhost:5432/agents
export REDIS_URL=redis://localhost:6379/0
```

Run a lesson's demo and tests (from the repository root):

```bash
PYTHONPATH=code LESSON_OFFLINE=1 python -m day03.demo
pytest code/day03
pytest                           # everything; infra tests skip if Postgres/Redis are unreachable
```

`LESSON_OFFLINE=1` forces the deterministic scripted model even when an API key is set. Without a key, demos use it automatically.

Suggested rhythm: read the lesson, run the demo, read the embedded code, then do the exercises and the challenge before looking at the knowledge-check answers.

### Repository layout

```text
ai-engineer/
├── README.md
├── requirements.txt
├── docker-compose.yml          Postgres+pgvector and Redis
├── .env.example
├── lessons/                    day-NN-<topic>.md, the course text
├── code/
│   ├── common/                 shared building blocks reused by every lesson (tested)
│   └── dayNN/                  runnable code, demo and tests for each lesson
├── projects/capstone/          the Ops Report Agent (Days 25-28)
├── resources/references.md     authoritative reading list
└── scripts/
    ├── sync_code.py            keeps code listings in lessons identical to the real files
    └── curriculum.py           generates the curriculum table above
```

Code listings inside lessons are not hand-copied. `scripts/sync_code.py` embeds the real source files, so what you read is what the tests run.

## Running scenario and capstone

Every lesson uses one consistent scenario: an **Ops Report Agent** that helps an on-call engineer investigate incidents and draft postmortems from runbooks, incident records and metrics. The shared synthetic dataset lives in `code/common/fixtures.py`.

The capstone (Days 25-28, in `projects/capstone/`) assembles the whole series into one service: planner, researcher, writer and reviewer roles; hybrid retrieval over pgvector; tools exposed through MCP; a human approval gate before anything is published; checkpointed, resumable runs; tracing; and an evaluation suite.

## Honest limits of the offline setup

- The `ScriptedLLM` replays fixed responses. It tests your plumbing (parsing, validation, loops, state, error handling), not model quality.
- `HashEmbedder` is lexical (hashed words and trigrams). It makes retrieval code runnable and deterministic, but it does not capture meaning. Lessons say so wherever it affects a conclusion.
- Nothing here reports performance numbers from a live model. Where a lesson prints numbers, they come from the offline code you can re-run.

## Progress tracker

- [ ] Day 1: LLM APIs and structured outputs
- [ ] Day 2: Tool calling
- [ ] Day 3: Agent loops and ReAct
- [ ] Day 4: Planning and decomposition
- [ ] Day 5: State and checkpoints
- [ ] Day 6: Memory
- [ ] Day 7: Context engineering
- [ ] Day 8: Embeddings and semantic search
- [ ] Day 9: RAG
- [ ] Day 10: Reranking, hybrid search, evaluation
- [ ] Day 11: MCP architecture
- [ ] Day 12: Production MCP tools
- [ ] Day 13: Agent runtime
- [ ] Day 14: State graphs
- [ ] Day 15: Multi-agent
- [ ] Day 16: Delegation and shared workspaces
- [ ] Day 17: Human-in-the-loop
- [ ] Day 18: Security and prompt injection
- [ ] Day 19: Guardrails
- [ ] Day 20: Evaluation
- [ ] Day 21: Tracing and observability
- [ ] Day 22: Testing agents
- [ ] Day 23: Reliability
- [ ] Day 24: Deployment
- [ ] Day 25: Capstone architecture
- [ ] Day 26: Capstone runtime and tools
- [ ] Day 27: Capstone memory, retrieval, orchestration
- [ ] Day 28: Capstone hardening and deployment

See [resources/references.md](resources/references.md) for the reading list.
