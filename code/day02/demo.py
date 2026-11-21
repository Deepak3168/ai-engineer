"""Day 2 demo: the tool-calling contract, offline.

    PYTHONPATH=code LESSON_OFFLINE=1 python -m day02.demo

Section 2 uses `get_client(script)`: with LLM_API_KEY set (and LESSON_OFFLINE unset) a real model
answers instead of the script, so its tool choices and wording will differ from the lesson.
Sections 1, 3, 4 and 5 call no model at all.
"""

from __future__ import annotations

import json

from common.fixtures import metrics_for
from common.llm import Message, ToolCall, get_client
from day02.execution import execute_batch
from day02.manual_call import one_round
from day02.ops_tools import TicketStore, build_registry


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def scripted_model() -> list[object]:
    first = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="call_a1", name="get_incident", arguments={"incident_id": "INC-1042"}),
            ToolCall(id="call_a2", name="search_runbooks", arguments={"query": "deploy rollback error rate", "limit": 2}),
        ],
    )
    final = (
        "INC-1042 (SEV1, checkout-api): error rate hit 31% minutes after release v2026.03.04-1; "
        "root cause was an unindexed query that saturated the connection pool. "
        "Relevant runbooks: rb-deploy-rollback, rb-db-connection-pool."
    )
    return [first, final]


def main() -> None:
    store = TicketStore()
    reg = build_registry(store)

    banner("1. What the model sees: the schema generated for create_ticket")
    print(json.dumps(reg.schemas(only=["create_ticket"])[0], indent=2))

    banner("2. One manual round trip (no loop)")
    client = get_client(scripted_model())
    trace = one_round(client, reg, "What happened in INC-1042 and which runbook applies?")
    print("assistant_1 :", json.dumps(trace.assistant_1)[:230], "...")
    for msg, res in zip(trace.tool_messages, trace.results, strict=True):
        print(f"tool message: id={msg['tool_call_id']} name={msg['name']} ok={res.ok} chars={len(msg['content'])}")
    print("request_2 roles:", [m["role"] for m in (trace.request_2 or {})["messages"]])
    print("final_text  :", trace.final_text)
    print("needs_another_round:", trace.needs_another_round)

    banner("3. Failures become observations, never exceptions")
    cases = [
        ToolCall(id="c1", name="get_incident_by_id", arguments={"incident_id": "INC-1042"}),
        ToolCall(id="c2", name="get_incident", arguments={}, raw_arguments='{"incident_id": "INC-1042"',
                 arguments_error="invalid JSON arguments: Expecting ',' delimiter (pos 27)"),
        ToolCall(id="c3", name="create_ticket", arguments={"title": "x", "severity": "SEV9", "idempotency_key": "k"}),
        ToolCall(id="c4", name="get_incident", arguments={"incident_id": "INC-1042", "verbose": True}),
        ToolCall(id="c5", name="get_incident", arguments={"incident_id": "INC-9999"}),
    ]
    for res in execute_batch(reg, cases, allow=("read", "write")):
        print(f"[{res.error_type}] {res.content[:200]}")

    banner("4. Result formatting: summary vs raw series")
    raw = json.dumps(metrics_for("checkout-api", "error_rate", 0, 60))
    summary = reg.execute(ToolCall(id="m1", name="query_metrics",
                                   arguments={"service": "checkout-api", "metric": "error_rate"}))
    print(f"raw series: {len(raw)} chars | summarised tool result: {len(summary.content)} chars")
    print(summary.content)
    tiny = build_registry(max_output_chars=120)
    clipped = tiny.execute(ToolCall(id="m2", name="get_incident", arguments={"incident_id": "INC-1042"}))
    print(f"clipped to 120 chars -> truncated={clipped.truncated}; tail: {clipped.content[-45:]!r}")

    banner("5. Side-effect policy and idempotent writes")
    write = ToolCall(id="w1", name="create_ticket",
                     arguments={"title": "Add index on orders", "severity": "SEV2", "idempotency_key": "inc-1042-index"})
    denied = execute_batch(reg, [write])[0]
    print(f"read-only run   : [{denied.error_type}] tickets={len(store.tickets)}")
    for call_id in ("w2", "w3"):
        res = execute_batch(reg, [write.model_copy(update={"id": call_id})], allow=("read", "write"))[0]
        print(f"write, call {call_id}  : {res.content}")
    print(f"tickets stored  : {len(store.tickets)}, real executions: {store.executions}")


if __name__ == "__main__":
    main()
