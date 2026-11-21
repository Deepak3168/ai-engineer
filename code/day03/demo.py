"""Day 3 demo: investigate INC-1042 three ways, then watch the guards stop two stuck agents. Offline.

    PYTHONPATH=code LESSON_OFFLINE=1 python -m day03.demo

The "models" are scripts, so this shows the *mechanics* of loops and guards - not how a real model
behaves. Token numbers come from ScriptedLLM's rough estimate (characters / 4), not a real tokenizer.
With LLM_API_KEY set (and LESSON_OFFLINE unset) sections 1-2 call a real model instead of the script.
"""

from __future__ import annotations

from day02.ops_tools import build_registry

from common.llm import Message, ScriptedLLM, ToolCall, get_client
from common.loop import AgentLoop
from day03.final_answer import FINAL_TOOL, add_final_tool
from day03.guarded_loop import GuardConfig, GuardedLoop
from day03.react_text import ReActTextLoop

TASK = "Investigate INC-1042: what happened, what does the data show, and which runbook applies?"
SYSTEM = "You are the Ops Report Agent. Use tools for facts; stop and answer once you have enough evidence."
ANSWER = (
    "INC-1042 (SEV1, checkout-api): error rate rose to 31% after release v2026.03.04-1 (peak minute 25 in the "
    "metrics window). Root cause per the record: an unindexed query saturated the DB connection pool. "
    "Follow rb-deploy-rollback, then rb-db-connection-pool."
)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def native_script() -> list[object]:
    return [
        Message(role="assistant", tool_calls=[
            ToolCall(id="n1", name="get_incident", arguments={"incident_id": "INC-1042"}),
            ToolCall(id="n2", name="query_metrics", arguments={"service": "checkout-api", "metric": "error_rate"}),
        ]),
        ScriptedLLM.tool_call("search_runbooks", "n3", query="deploy rollback connection pool"),
        ANSWER,
    ]


def react_script() -> list[object]:
    return [
        # The model invents its own Observation: the parser must discard it.
        'Thought: I need the incident record first.\nAction: get_incident\nAction Input: {"incident_id": "INC-1042"}\n'
        "Observation: INC-1042 was a DNS outage.",
        "I will look at metrics now.",  # no Action line: a format error, repaired by feedback
        'Thought: check the error rate.\nAction: query_metrics\n'
        'Action Input: ```json\n{"service": "checkout-api", "metric": "error_rate"}\n```',
        'Thought: find the procedure.\nAction: search_runbooks\nAction Input: {"query": "deploy rollback"}',
        f"Thought: I can answer now.\nFinal Answer: {ANSWER}",
    ]


def show_events(event: dict) -> None:
    if event["type"] == "llm_response":
        print(f"  step {event['step']}: model requested {event['tool_calls'] or 'a final answer'}")
    elif event["type"] == "tool_result":
        extra = " (BLOCKED)" if event.get("blocked") else ""
        print(f"    {event['tool']:<16} ok={event['ok']} {event.get('error_type') or ''}{extra}")


def main() -> None:
    reg = build_registry()

    banner("1. Native tool-calling loop (common.loop.AgentLoop)")
    res = AgentLoop(get_client(native_script()), reg, SYSTEM, max_steps=6, on_event=show_events).run(TASK)
    print(f"status={res.status} steps={len(res.steps)} tool_calls={res.tool_call_count} tokens~{res.usage.total_tokens}")
    print("answer:", res.final_answer)

    banner("2. Text ReAct loop (no native tool calling)")
    react = ReActTextLoop(get_client(react_script()), reg).run(TASK)
    for m in react.messages[1:]:
        print(f"  [{m.role:<9}] {(m.content or '')[:96]!r}")
    print(f"status={react.status} steps={len(react.steps)} format_errors={react.format_errors}")
    leaked = any("DNS outage" in (m.content or "") for m in react.messages)
    print("model-invented Observation present in history:", leaked)

    banner("3. A stuck model: same search forever")
    stuck = [ScriptedLLM.tool_call("search_runbooks", f"s{i}", query="latency spike") for i in range(8)]
    base = AgentLoop(ScriptedLLM(stuck), build_registry(), SYSTEM, max_steps=8).run(TASK)
    print(f"AgentLoop   : status={base.status:<14} steps={len(base.steps)} tool executions={base.tool_call_count} "
          f"tokens~{base.usage.total_tokens}")
    guarded = GuardedLoop(ScriptedLLM(stuck), build_registry(), SYSTEM, GuardConfig(max_steps=8, max_blocked=2),
                          on_event=show_events).run(TASK)
    print(f"GuardedLoop : status={guarded.status:<14} steps={len(guarded.steps)} "
          f"tool executions={sum(guarded.executed_by_tool.values())} blocked={guarded.blocked} tokens~{guarded.usage.total_tokens}")
    print("detail:", guarded.error)

    banner("4. A wandering model: different queries, nothing new learned")
    wander = [ScriptedLLM.tool_call("search_runbooks", f"w{i}", query=f"zzqx{i} unknownterm") for i in range(4)]
    wander.append("So far no runbook matched my keyword searches; the incident id and service are still unconfirmed.")
    res = GuardedLoop(ScriptedLLM(wander), build_registry(), SYSTEM,
                      GuardConfig(max_steps=8, no_progress_steps=3, wrap_up=True)).run(TASK)
    print(f"status={res.status} steps={len(res.steps)} executed={res.executed_by_tool}")
    print("detail:", res.error)
    print("partial answer from wrap-up call:", res.partial_answer)

    banner("5. Per-tool cap, then an explicit final answer")
    capped_reg = add_final_tool(build_registry())
    script = [
        ScriptedLLM.tool_call("search_runbooks", "k1", query="deploy rollback"),
        ScriptedLLM.tool_call("search_runbooks", "k2", query="connection pool"),
        ScriptedLLM.tool_call("search_runbooks", "k3", query="queue backlog"),
        ScriptedLLM.tool_call(FINAL_TOOL, "k4", answer=ANSWER, evidence=["INC-1042", "rb-deploy-rollback"], confidence="medium"),
    ]
    cfg = GuardConfig(max_calls_per_tool={"search_runbooks": 2}, final_tool=FINAL_TOOL)
    res = GuardedLoop(ScriptedLLM(script), capped_reg, SYSTEM, cfg, on_event=show_events).run(TASK)
    third = next(r for s in res.steps for c, r in s.tool_results if c.id == "k3")
    print(f"status={res.status} blocked={res.blocked} executed={res.executed_by_tool}")
    print("model saw for the 3rd search:", third.content)


if __name__ == "__main__":
    main()
