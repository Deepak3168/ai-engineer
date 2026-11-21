from __future__ import annotations

import pytest

from common.infra import postgres_available, temp_schema
from common.llm import ScriptedLLM
from day05.ops_tools import TicketLedger, build_registry
from day05.pg_store import PostgresCheckpointStore
from day05.runner import DurableAgent
from day05.state import RunState, RunStatus
from day05.store import CheckpointConflict

pytestmark = [
    pytest.mark.infra,
    pytest.mark.skipif(not postgres_available(), reason="Postgres not reachable (set DATABASE_URL)"),
]


@pytest.fixture()
def store():
    with temp_schema() as (url, schema):
        s = PostgresCheckpointStore(url, schema)
        s.ensure_schema()
        yield s


class Crash(Exception):
    pass


def test_roundtrip_history_and_time_travel(store):
    s = RunState(run_id="pg-1", task="t", seq=1)
    store.save(s)
    s.seq = 2
    s.transition(RunStatus.RUNNING)
    store.save(s)
    assert store.load("pg-1").status == RunStatus.RUNNING
    assert store.load("pg-1", seq=1).status == RunStatus.CREATED
    assert store.history("pg-1") == [(1, "created"), (2, "running")]
    assert store.load("none") is None and store.list_runs() == ["pg-1"]


def test_two_writers_same_seq_exactly_one_wins(store):
    a = RunState(run_id="pg-2", task="t", seq=1)
    store.save(a)
    w1 = RunState(run_id="pg-2", task="worker-1", seq=2)
    w2 = RunState(run_id="pg-2", task="worker-2", seq=2)
    store.save(w1)
    with pytest.raises(CheckpointConflict):
        store.save(w2)
    assert store.load("pg-2").task == "worker-1"


def test_prune_keeps_newest(store):
    for i in range(1, 8):
        store.save(RunState(run_id="pg-3", task="t", seq=i))
    assert store.prune("pg-3", keep_last=3) == 4
    assert [s for s, _ in store.history("pg-3")] == [5, 6, 7]
    with pytest.raises(ValueError):
        store.prune("pg-3", keep_last=0)


def test_bad_schema_name_rejected():
    with pytest.raises(ValueError):
        PostgresCheckpointStore("postgresql://x", 'x"; DROP SCHEMA public; --')


def test_crash_and_resume_through_postgres(store):
    ledger = TicketLedger()
    registry = build_registry(ledger)
    first = ScriptedLLM([ScriptedLLM.tool_call("create_ticket", "c1", title="Add index", severity="SEV2"), Crash("kill")])
    with pytest.raises(Crash):
        DurableAgent(first, registry, store, "sys").start("t", "pg-run")
    state = DurableAgent(ScriptedLLM(["TCK-001 filed"]), registry, store, "sys").resume("pg-run")
    assert state.status == RunStatus.COMPLETED and len(ledger.created) == 1
    assert store.history("pg-run")[-1][1] == "completed"
