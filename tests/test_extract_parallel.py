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
        delay = self.state["delays"].get(path, 0)
        if delay:
            import time
            time.sleep(delay)
        if path in self.state["crash_once"] and call_count == 1:
            raise extractor.RlogWorkerCrash(11, "simulated crash")
        if path in self.state["always_crash"]:
            raise extractor.RlogWorkerCrash(12, "simulated unreadable rlog")

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


def make_state(row_values):
    return {
        "row_values": row_values,
        "delays": {},
        "crash_once": set(),
        "always_crash": set(),
        "calls": {},
        "created": 0,
        "closed": 0,
        "lock": threading.Lock(),
    }


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_parallel_merge_preserves_input_order_and_cleans_segments(tmp_path):
    paths = ["route-a--2/rlog.zst", "route-b--1/rlog.zst", "route-c--0/rlog.zst"]
    state = make_state(dict(zip(paths, (10, 20, 30))))
    state["delays"] = {paths[0]: 0.15, paths[1]: 0.01, paths[2]: 0.05}
    output = tmp_path / "output.csv"

    stats = extractor.extract_rlogs_to_csv(
        paths,
        output,
        temporal=True,
        show_progress=False,
        worker_factory=FakeWorkerFactory(state),
        rlog_workers=3,
    )

    rows = read_rows(output)
    assert [float(row["timestamp"]) for row in rows] == [10, 20, 30]
    assert [row["route_id"] for row in rows] == ["route-a", "route-b", "route-c"]
    assert stats["rows_written"] == 3
    assert not list(tmp_path.glob(".nnlc_rlog_*.csv"))
    assert state["created"] == 3
    assert state["closed"] == 3


def test_skip_corrupt_retries_once_then_continues(tmp_path):
    paths = ["good/rlog.zst", "bad/rlog.zst", "later/rlog.zst"]
    state = make_state({paths[0]: 1, paths[2]: 3})
    state["always_crash"].add(paths[1])
    output = tmp_path / "output.csv"

    stats = extractor.extract_rlogs_to_csv(
        paths,
        output,
        show_progress=False,
        worker_factory=FakeWorkerFactory(state),
        rlog_workers=2,
        skip_corrupt=True,
    )

    assert [float(row["timestamp"]) for row in read_rows(output)] == [1, 3]
    assert state["calls"][paths[1]] == 2
    assert stats["skipped_rlogs"][0][0] == paths[1]
    assert not list(tmp_path.glob(".nnlc_rlog_*.csv"))


def test_strict_failure_closes_workers_and_cleans_segments(tmp_path):
    paths = ["bad/rlog.zst", "good/rlog.zst"]
    state = make_state({paths[1]: 2})
    state["always_crash"].add(paths[0])
    output = tmp_path / "output.csv"

    with pytest.raises(RuntimeError, match="bad/rlog.zst"):
        extractor.extract_rlogs_to_csv(
            paths,
            output,
            show_progress=False,
            worker_factory=FakeWorkerFactory(state),
            rlog_workers=2,
        )

    assert state["closed"] >= 1
    assert not list(tmp_path.glob(".nnlc_rlog_*.csv"))

