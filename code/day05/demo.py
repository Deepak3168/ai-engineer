"""Day 5 demo: crash a run mid-flight, resume it from disk, and prove no write was repeated. Offline.

    PYTHONPATH=code LESSON_OFFLINE=1 python -m day05.demo

The "crash" is an exception that is not an LLMError, so the runner (correctly) does not catch it. It stands
in for SIGKILL, an OOM kill or a deploy. Only what reached the checkpoint store survives, as in production.
"""

from __future__ import annotations

import tempfile

from common.llm import Message, ScriptedLLM, ToolCall
from day05.migrate import migrate
from day05.ops_tools import TicketLedger, build_registry
from day05.runner import DurableAgent
from day05.state import RunStatus
from day05.store import CheckpointConflict, FileCheckpointStore


class SimulatedCrash(Exception):
    """Not an LLMError: the runner must let it propagate, exactly like a killed process."""


def agent(llm: ScriptedLLM, registry, store) -> DurableAgent:
    return DurableAgent(llm, registry, store, "You are the Ops Report Agent.")


def ticket(call_id: str, name: str = "create_ticket", **extra: str) -> Message:
    return ScriptedLLM.tool_call(name, call_id=call_id, title="Add index on orders", severity="SEV2", **extra)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def scenario_resume_after_crash(store: FileCheckpointStore) -> None:
    banner("1. Crash after a write finished, resume in a 'new process'")
    ledger = TicketLedger()
    registry = build_registry(ledger)
    first = ScriptedLLM([ScriptedLLM.tool_call("get_incident", "c1", incident_id="INC-1042"), ticket("c2"),
                         SimulatedCrash("worker killed")])
    try:
        agent(first, registry, store).start("Investigate INC-1042 and file a follow-up ticket", "run-1")
    except SimulatedCrash as exc:
        print(f"process died: {exc}")
    on_disk = store.load("run-1")
    print(f"on disk: status={on_disk.status.value} seq={on_disk.seq} steps={on_disk.steps} tickets_created={len(ledger.created)}")
    print("tool ledger:", {c: (r.name, r.status) for c, r in on_disk.tool_records.items()})

    second = ScriptedLLM(["Filed TCK-001 for INC-1042."])
    state = agent(second, registry, store).resume("run-1")
    print(f"resumed: status={state.status.value} resume_count={state.resume_count} tickets_created={len(ledger.created)}")
    sent = [m.content for m in second.calls[0]["messages"] if m.role == "tool"]
    print("model saw recorded tool results, not re-executions:", [c[:40] for c in sent])


def scenario_unknown_outcome(store: FileCheckpointStore) -> None:
    banner("2. Crash INSIDE a non-idempotent tool: the honest answer is 'unknown'")
    ledger = TicketLedger()
    registry = build_registry(ledger)
    real = registry.execute

    def crash_after_effect(call: ToolCall):
        real(call)  # the ticket IS created...
        raise SimulatedCrash("...then the worker dies before the result is checkpointed")

    registry.execute = crash_after_effect  # type: ignore[method-assign]
    try:
        agent(ScriptedLLM([ticket("c1")]), registry, store).start("File a ticket", "run-2")
    except SimulatedCrash:
        pass
    registry.execute = real  # type: ignore[method-assign]
    print(f"tickets actually created before resume: {len(ledger.created)}")

    def next_reply(messages) -> str:
        note = next(m.content for m in messages if m.role == "tool")
        return "I could not confirm the ticket; verifying before retrying." if "UNKNOWN" in (note or "") else "?"

    state = agent(ScriptedLLM([next_reply]), registry, store).resume("run-2")
    record = state.tool_records["c1"]
    print(f"record: ok={record.ok} error_type={record.error_type}")
    print(f"tickets after resume (no blind retry): {len(ledger.created)} | final: {state.final_answer}")


def scenario_two_workers(store: FileCheckpointStore) -> None:
    banner("3. Two workers resume the same run: the second write is rejected")
    run = store.load("run-1")
    stale = run.model_copy(deep=True)
    run.seq += 1
    store.save(run)
    try:
        store.save(stale.model_copy(update={"seq": run.seq}))
    except CheckpointConflict as exc:
        print("rejected:", exc)


def scenario_migration() -> None:
    banner("4. An old (v1) checkpoint written by last month's code")
    old = {"schema_version": 1, "run_id": "legacy", "task": "t", "status": "done", "messages": [],
           "tool_outputs": {"c1": "TCK-007"}}
    new = migrate(old)
    print("status:", new["status"], "| schema_version:", new["schema_version"], "| tool_records:", list(new["tool_records"]))
    assert RunStatus(new["status"]) == RunStatus.COMPLETED


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = FileCheckpointStore(tmp)
        scenario_resume_after_crash(store)
        scenario_unknown_outcome(store)
        scenario_two_workers(store)
        scenario_migration()


if __name__ == "__main__":
    main()
