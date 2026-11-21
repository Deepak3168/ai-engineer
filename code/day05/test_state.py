from __future__ import annotations

import pytest

from common.llm import Message, ToolCall, assistant, system, tool_result, user
from day05.migrate import UnsupportedSchema, migrate
from day05.state import IllegalTransition, RunState, RunStatus
from day05.store import parse_state


def test_happy_path_transitions():
    s = RunState(run_id="r1", task="t")
    for nxt in (RunStatus.RUNNING, RunStatus.COMPLETED):
        s.transition(nxt)
    assert s.is_terminal


@pytest.mark.parametrize(
    "start,bad",
    [
        (RunStatus.CREATED, RunStatus.COMPLETED),
        (RunStatus.COMPLETED, RunStatus.RUNNING),
        (RunStatus.CANCELLED, RunStatus.RUNNING),
        (RunStatus.RUNNING, RunStatus.CREATED),
    ],
)
def test_illegal_transitions_raise_and_do_not_mutate(start, bad):
    s = RunState(run_id="r1", task="t", status=start)
    with pytest.raises(IllegalTransition):
        s.transition(bad)
    assert s.status == start


def test_failed_can_be_retried():
    s = RunState(run_id="r1", task="t", status=RunStatus.FAILED)
    s.transition(RunStatus.RUNNING)
    assert s.status == RunStatus.RUNNING


def test_state_json_roundtrip_is_lossless():
    call = ToolCall(id="c1", name="get_incident", arguments={"incident_id": "INC-1042"})
    s = RunState(run_id="r1", task="t", messages=[system("s"), user("u"), Message(role="assistant", tool_calls=[call])])
    again = RunState.model_validate_json(s.model_dump_json())
    assert again == s


def test_pending_calls_only_lists_unanswered():
    a, b = ToolCall(id="a", name="x"), ToolCall(id="b", name="y")
    s = RunState(run_id="r", task="t", messages=[system("s"), user("u"), Message(role="assistant", tool_calls=[a, b])])
    assert [c.id for c in s.pending_calls()] == ["a", "b"]
    s.messages.append(tool_result(a, "ok"))
    assert [c.id for c in s.pending_calls()] == ["b"]
    s.messages.append(tool_result(b, "ok"))
    assert s.pending_calls() == []
    s.messages.append(assistant("done"))
    assert s.pending_calls() == []


def v1_doc() -> dict:
    return {"schema_version": 1, "run_id": "old", "task": "t", "status": "done", "seq": 4,
            "messages": [{"role": "user", "content": "hi"}], "steps": 1,
            "tool_outputs": {"c1": "TCK-001"}}


def test_v1_checkpoint_migrates_to_current_schema():
    state = parse_state(v1_doc())
    assert state.schema_version == 2
    assert state.status == RunStatus.COMPLETED
    rec = state.tool_records["c1"]
    assert rec.status == "finished" and rec.content == "TCK-001"


def test_future_or_missing_version_is_refused():
    with pytest.raises(UnsupportedSchema):
        migrate({"schema_version": 99})
    with pytest.raises(UnsupportedSchema):
        migrate({"run_id": "x"})
