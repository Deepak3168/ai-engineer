"""Behavioural tests for DurableAgent, including real crash-and-resume scenarios."""

from __future__ import annotations

import pytest

from common.llm import LLMResponse, Message, ScriptedLLM, ToolCall, TransientLLMError
from common.tools import ToolRegistry
from day05.ops_tools import TicketLedger, build_registry
from day05.runner import DurableAgent, RunNotFound
from day05.state import IllegalTransition, RunStatus
from day05.store import FileCheckpointStore


class SimulatedCrash(Exception):
    """Stands in for SIGKILL / OOM / deploy: not an LLMError, so the runner must not catch it."""


def agent(llm, registry, store, **kw) -> DurableAgent:
    return DurableAgent(llm, registry, store, "You are the Ops Report Agent.", **kw)


def ticket(call_id: str, name: str = "create_ticket", **extra) -> Message:
    args = {"title": "Add index on orders", "severity": "SEV2", **extra}
    return ScriptedLLM.tool_call(name, call_id=call_id, **args)


def test_happy_path_checkpoints_every_step(tmp_path):
    ledger = TicketLedger()
    events: list[dict] = []
    llm = ScriptedLLM([ScriptedLLM.tool_call("get_incident", "c1", incident_id="INC-1042"), ticket("c2"), "Done: TCK-001"])
    state = agent(llm, build_registry(ledger), FileCheckpointStore(tmp_path), on_event=events.append).start("go", "run-a")
    assert state.status == RunStatus.COMPLETED and state.final_answer == "Done: TCK-001"
    assert len(ledger.created) == 1
    seqs = [e["seq"] for e in events if e["type"] == "checkpoint"]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    # created, running, llm1, tool1, llm2, intent(write), tool2, llm3, completed
    assert state.seq == 9
    assert FileCheckpointStore(tmp_path).load("run-a").status == RunStatus.COMPLETED


def test_crash_between_steps_then_resume_does_not_repeat_write(tmp_path):
    """The core Day 5 test: kill the run after a write completed, resume in a 'new process'."""
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)
    crashing = ScriptedLLM([ticket("c1"), SimulatedCrash("worker killed")])
    with pytest.raises(SimulatedCrash):
        agent(crashing, registry, store).start("file a ticket", "run-b")
    assert len(ledger.created) == 1

    on_disk = store.load("run-b")
    assert on_disk.status == RunStatus.RUNNING  # a crash is invisible to the state machine
    assert on_disk.tool_records["c1"].status == "finished"

    # "New process": fresh agent, fresh LLM client, same store. Only the checkpoint survives.
    fresh = ScriptedLLM(["Ticket TCK-001 filed."])
    state = agent(fresh, registry, store).resume("run-b")
    assert state.status == RunStatus.COMPLETED
    assert len(ledger.created) == 1, "completed write must not run again"
    assert state.resume_count == 1
    sent = fresh.calls[0]["messages"]
    assert any(m.role == "tool" and m.content == "TCK-001" for m in sent), "model sees the recorded result"


def test_crash_after_llm_response_before_tools_runs_pending_tools_once(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)

    class CrashBeforeTools(ToolRegistry):
        pass

    # Crash inside the registry on the first call, before any tool executed.
    real = registry.execute
    state_flag = {"armed": True}

    def flaky_execute(call: ToolCall):
        if state_flag["armed"]:
            raise SimulatedCrash("killed before tool ran")
        return real(call)

    registry.execute = flaky_execute  # type: ignore[method-assign]
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([ticket("c1")]), registry, store).start("t", "run-c")
    assert ledger.created == []
    # The intent record was durably written before the crash, so the outcome is declared unknown.
    assert store.load("run-c").tool_records["c1"].status == "started"
    state_flag["armed"] = False
    state = agent(ScriptedLLM(["ok"]), registry, store).resume("run-c")
    assert state.status == RunStatus.COMPLETED
    assert ledger.created == [], "non-idempotent call with unknown outcome is not blindly re-run"
    assert state.tool_records["c1"].error_type == "outcome_unknown"


def test_crash_mid_tool_idempotent_tool_is_retried_safely(tmp_path):
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)
    real = registry.execute

    def crash_after_effect(call: ToolCall):
        real(call)  # the side effect happens...
        raise SimulatedCrash("...then the worker dies before checkpointing")

    registry.execute = crash_after_effect  # type: ignore[method-assign]
    key = "key-inc-1042-ticket"
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([ticket("c1", "create_ticket_keyed", idempotency_key=key)]), registry, store).start("t", "run-d")
    assert len(ledger.created) == 1  # effect happened, checkpoint did not
    registry.execute = real  # type: ignore[method-assign]
    state = agent(ScriptedLLM(["done"]), registry, store).resume("run-d")
    assert state.status == RunStatus.COMPLETED
    assert len(ledger.created) == 1, "same idempotency key -> deduplicated on re-execution"
    assert state.tool_records["c1"].ok is True


def test_replay_uses_recorded_result_when_record_is_finished(tmp_path):
    """A finished record whose message was not appended is replayed from the ledger, not re-executed."""
    store = FileCheckpointStore(tmp_path)
    ledger = TicketLedger()
    registry = build_registry(ledger)
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([ticket("c1"), SimulatedCrash("x")]), registry, store).start("t", "run-e")
    s = store.load("run-e")
    s.messages.pop()  # simulate a checkpoint taken after the record but before the message
    s.seq += 1
    store.save(s)
    events: list[dict] = []
    state = agent(ScriptedLLM(["ok"]), registry, store, on_event=events.append).resume("run-e")
    assert [e["type"] for e in events if e["type"].startswith("tool_")] == ["tool_replayed"]
    assert len(ledger.created) == 1 and state.status == RunStatus.COMPLETED


def test_llm_error_fails_durably_and_resume_retries(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    state = agent(ScriptedLLM([TransientLLMError("upstream 503")]), registry, store).start("t", "run-f")
    assert state.status == RunStatus.FAILED and "upstream 503" in state.error
    state = agent(ScriptedLLM(["recovered"]), registry, store).resume("run-f")
    assert state.status == RunStatus.COMPLETED and state.error is None


def test_step_budget_survives_resume(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    loop = [ScriptedLLM.tool_call("get_incident", f"c{i}", incident_id="INC-1042") for i in range(5)]
    state = agent(ScriptedLLM(loop), registry, store, max_steps=2).start("t", "run-g")
    assert state.status == RunStatus.FAILED and state.steps == 2
    assert "no final answer" in state.error


def test_terminal_runs_are_not_rerun_and_cancel_is_final(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    llm = ScriptedLLM(["hi"])
    a = agent(llm, registry, store)
    done = a.start("t", "run-h")
    again = a.resume("run-h")
    assert again.seq == done.seq and llm.remaining == 0
    with pytest.raises(IllegalTransition):
        a.cancel("run-h")
    with pytest.raises(ValueError):
        a.start("t", "run-h")
    with pytest.raises(RunNotFound):
        a.resume("nope")


def test_usage_accumulates_across_a_crash(tmp_path):
    store = FileCheckpointStore(tmp_path)
    registry = build_registry(TicketLedger())
    resp = LLMResponse(message=ScriptedLLM.tool_call("get_incident", "c1", incident_id="INC-1042"),
                       usage={"prompt_tokens": 100, "completion_tokens": 10})
    with pytest.raises(SimulatedCrash):
        agent(ScriptedLLM([resp, SimulatedCrash("x")]), registry, store).start("t", "run-i")
    final = LLMResponse(message=Message(role="assistant", content="ok"), usage={"prompt_tokens": 120, "completion_tokens": 5})
    state = agent(ScriptedLLM([final]), registry, store).resume("run-i")
    assert state.usage.total_tokens == 235 and state.steps == 2
