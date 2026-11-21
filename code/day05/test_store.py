from __future__ import annotations

import json
import os

import pytest

from day05.state import RunState, RunStatus
from day05.store import CheckpointConflict, CorruptCheckpoint, FileCheckpointStore, validate_run_id


def make(seq: int, run_id: str = "r1") -> RunState:
    return RunState(run_id=run_id, task="t", seq=seq)


def test_save_and_load_roundtrip(tmp_path):
    store = FileCheckpointStore(tmp_path)
    s = make(1)
    s.transition(RunStatus.RUNNING)
    store.save(s)
    assert store.load("r1") == s
    assert store.load("missing") is None
    assert store.list_runs() == ["r1"]


def test_stale_or_duplicate_seq_is_a_conflict(tmp_path):
    store = FileCheckpointStore(tmp_path)
    store.save(make(2))
    with pytest.raises(CheckpointConflict):
        store.save(make(2))
    with pytest.raises(CheckpointConflict):
        store.save(make(1))
    store.save(make(3))
    assert store.load("r1").seq == 3


def test_failed_write_leaves_previous_checkpoint_intact(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    store.save(make(1))

    def boom(*_a, **_k):
        raise OSError("disk full during rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.save(make(2))
    monkeypatch.undo()
    assert store.load("r1").seq == 1
    assert [p.name for p in tmp_path.iterdir()] == ["r1.json"]  # temp file cleaned up


def test_corrupt_file_raises_not_silently_ignored(tmp_path):
    store = FileCheckpointStore(tmp_path)
    (tmp_path / "bad.json").write_text('{"schema_version": 2, "run_id": "bad"')  # torn write
    with pytest.raises(CorruptCheckpoint):
        store.load("bad")
    (tmp_path / "bad2.json").write_text(json.dumps({"schema_version": 2, "run_id": "bad2"}))  # missing fields
    with pytest.raises(CorruptCheckpoint):
        store.load("bad2")


@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "", "..", "x" * 200, "a b"])
def test_run_id_validation_blocks_path_traversal(bad):
    with pytest.raises(ValueError):
        validate_run_id(bad)
