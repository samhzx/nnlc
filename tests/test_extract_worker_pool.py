import csv
import threading
from pathlib import Path

import pytest

from nnlc_tools import extract_lateral_data as extractor


class FakeWorker:
    def __init__(self, state):
        self.state = state
        self.files_processed = 0
        self.closed = False

    def process(self, request):
        if self.closed:
            raise RuntimeError("fake worker is closed")
        path = request["rlog_path"]
        with self.state["lock"]:
            self.state["calls"][path] = self.state["calls"].get(path, 0) + 1
            call_count = self.state["calls"][path]
        if path in self.state["delays"]:
            self.state["delays"][path].wait()
        if path in self.state["crash_paths"]:
            Path(request["output_path"]).write_text("partial\n", encoding="utf-8")
            raise extractor.RlogWorkerCrash(11, "simulated native crash")

        columns = extractor.COLUMNS + (
            [spec[2] for spec in extractor._temporal_column_specs()]
            if request.get("temporal") else []
        ) + ["route_id"]
        row = [0] * len(columns)
        row[0] = float(self.state["row_values"][path])
        row[-1] = request["route_id"]
        with open(request["output_path"], "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            writer.writerow(row)
        self.files_processed += 1
        return {
            "status": "ok",
            "rows_written": 1,
            "rows_seen": 1,
            "rows_filtered": 0,
        }

    def close(self):
        self.closed = True
        self.state["closed"] += 1

    def interrupt(self):
        self.closed = True


class FakeWorkerFactory:
    def __init__(self, state):
        self.state = state

    def __call__(self):
        self.state["created"] += 1
        return FakeWorker(self.state)


def new_state(row_values):
    return {
        "row_values": row_values,
        "delays": {},
        "crash_paths": set(),
        "calls": {},
        "created": 0,
        "closed": 0,
        "lock": threading.Lock(),
    }


def test_resolve_rlog_workers(monkeypatch):
    monkeypatch.setattr(extractor.os, "cpu_count", lambda: 16)
    assert extractor.resolve_rlog_workers("auto", 20) == extractor.MAX_RLOG_WORKERS
    assert extractor.resolve_rlog_workers("auto", 2) == 2
    assert extractor.resolve_rlog_workers("3", 20) == 3
    assert extractor.resolve_rlog_workers(20, 3) == 3
    with pytest.raises(ValueError):
        extractor.resolve_rlog_workers("0", 2)


def test_worker_slot_restarts_after_limit(monkeypatch):
    state = new_state({"a": 1, "b": 2})
    factory = FakeWorkerFactory(state)
    slot = extractor._RlogWorkerSlot(factory)
    monkeypatch.setattr(extractor, "MAX_RLOGS_PER_WORKER", 1)
    output_dir = Path("/tmp")

    first = output_dir / "nnlc-test-worker-1.csv"
    second = output_dir / "nnlc-test-worker-2.csv"
    try:
        slot.process({
            "rlog_path": "a",
            "output_path": str(first),
            "temporal": False,
            "route_id": "a",
        })
        slot.process({
            "rlog_path": "b",
            "output_path": str(second),
            "temporal": False,
            "route_id": "b",
        })
        assert state["created"] == 2
        assert state["closed"] == 1
    finally:
        slot.close()
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)

