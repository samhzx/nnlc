import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from nnlc_tools import extract_lateral_data as extractor
from nnlc_tools.route_utils import extract_route_id


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
        columns = extractor.csv_output_columns(request.get("temporal"))
        row = [0] * len(columns)
        row[0] = float(self.state["row_values"][path])
        row[-1] = request["route_id"]
        with open(request["output_path"], "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            writer.writerow(row)
        cancel_event = self.state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()
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
        "calls": {},
        "created": 0,
        "closed": 0,
        "lock": threading.Lock(),
        "cancel_event": None,
    }


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def touch_rlog(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return os.fspath(path)


def write_day_csv(path, timestamp, route_id, temporal=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = extractor.csv_output_columns(temporal)
    row = [0] * len(columns)
    row[0] = float(timestamp)
    row[-1] = route_id
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerow(row)
    return path


def extract_with_fake(data_dir, output, row_values, **kwargs):
    state = make_state(row_values)
    stats = extractor.extract_directory_to_csv(
        data_dir,
        output,
        temporal=True,
        show_progress=False,
        worker_factory=FakeWorkerFactory(state),
        rlog_workers=1,
        **kwargs,
    )
    return stats, state


def test_non_date_tree_extracts_once_without_writing_data_dir(tmp_path):
    data = tmp_path / "reallog"
    rlog = touch_rlog(data / "route-a--0" / "rlog.zst")
    output = tmp_path / "out" / "lateral_data.csv"

    stats, state = extract_with_fake(data, output, {rlog: 11})

    assert stats["rows_written"] == 1
    assert stats["reused_days"] == 0
    assert stats["extracted_days"] == 0
    assert state["calls"][rlog] == 1
    assert Path(output).is_file()
    assert list(data.rglob("lateral_data.csv")) == []
    assert [float(row["timestamp"]) for row in read_rows(output)] == [11]


def test_nested_yyyymmdd_without_direct_child_is_not_date_mode(tmp_path):
    data = tmp_path / "reallog"
    rlog = touch_rlog(data / "nested" / "20261016" / "route-a--0" / "rlog.zst")
    output = tmp_path / "out.csv"

    assert extractor.uses_date_cached_extraction(data) is False
    stats, _state = extract_with_fake(data, output, {rlog: 4})

    assert stats["extracted_days"] == 0
    assert not (data / "nested" / "20261016" / "lateral_data.csv").exists()
    assert output.is_file()


def test_two_days_skip_existing_and_preserve_find_rlogs_order(tmp_path):
    data = tmp_path / "reallog"
    day1_rlog = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    day2_rlog = touch_rlog(data / "20261017" / "route-b--0" / "rlog.zst")
    rlogs = extractor.find_rlogs(data)
    assert rlogs == [day1_rlog, day2_rlog]

    write_day_csv(
        data / "20261016" / "lateral_data.csv",
        10,
        extract_route_id(day1_rlog),
    )
    output = tmp_path / "out" / "lateral_data.csv"
    stats, state = extract_with_fake(
        data,
        output,
        {day1_rlog: 10, day2_rlog: 20},
        rlog_files=list(reversed(rlogs)),
    )

    assert stats["reused_days"] == 1
    assert stats["extracted_days"] == 1
    assert day1_rlog not in state["calls"]
    assert state["calls"][day2_rlog] == 1
    assert (data / "20261017" / "lateral_data.csv").is_file()

    expected = tmp_path / "expected.csv"
    extractor.extract_rlogs_to_csv(
        rlogs,
        expected,
        temporal=True,
        show_progress=False,
        worker_factory=FakeWorkerFactory(make_state({day1_rlog: 10, day2_rlog: 20})),
        rlog_workers=1,
    )
    assert read_rows(output) == read_rows(expected)


def test_existing_day_csv_ignores_new_rlogs(tmp_path):
    data = tmp_path / "reallog"
    first = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    output = tmp_path / "out.csv"
    stats, _state = extract_with_fake(data, output, {first: 1})
    assert stats["extracted_days"] == 1
    original_rows = read_rows(output)

    extra = touch_rlog(data / "20261016" / "route-b--0" / "rlog.zst")
    stats, state = extract_with_fake(data, output, {first: 1, extra: 2})

    assert stats["reused_days"] == 1
    assert stats["extracted_days"] == 0
    assert state["calls"] == {}
    assert read_rows(output) == original_rows
    assert [float(row["timestamp"]) for row in original_rows] == [1]


def test_single_date_dir_as_reallog_root_reuses_day_csv(tmp_path):
    data = tmp_path / "20261016"
    rlog = touch_rlog(data / "route-a--0" / "rlog.zst")
    output = tmp_path / "combined.csv"

    stats, _state = extract_with_fake(data, output, {rlog: 9})
    day_csv = data / "lateral_data.csv"
    assert stats["extracted_days"] == 1
    assert day_csv.is_file()
    assert read_rows(output) == read_rows(day_csv)

    stats, state = extract_with_fake(data, output, {rlog: 9})
    assert stats["reused_days"] == 1
    assert stats["extracted_days"] == 0
    assert state["calls"] == {}
    assert read_rows(output) == read_rows(day_csv)


def test_wrong_header_day_csv_is_reextracted(tmp_path):
    data = tmp_path / "reallog"
    rlog = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    day_csv = data / "20261016" / "lateral_data.csv"
    day_csv.parent.mkdir(parents=True, exist_ok=True)
    day_csv.write_text("foo,bar\n1,2\n", encoding="utf-8")
    output = tmp_path / "out.csv"

    stats, state = extract_with_fake(data, output, {rlog: 6})

    assert stats["extracted_days"] == 1
    assert state["calls"][rlog] == 1
    with open(day_csv, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == extractor.csv_output_columns(True)
    assert [float(row["timestamp"]) for row in read_rows(output)] == [6]


def test_temporal_mode_mismatch_invalidates_day_csv(tmp_path):
    data = tmp_path / "reallog"
    rlog = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    day_csv = data / "20261016" / "lateral_data.csv"
    write_day_csv(day_csv, 1, "route-a", temporal=False)
    output = tmp_path / "out.csv"

    stats, state = extract_with_fake(data, output, {rlog: 8})

    assert stats["reused_days"] == 0
    assert stats["extracted_days"] == 1
    assert state["calls"][rlog] == 1
    assert [float(row["timestamp"]) for row in read_rows(output)] == [8]


@pytest.mark.parametrize(
    "body",
    [b"not,a,valid,row\n", b"\xff,invalid,utf8\n"],
    ids=["wrong-column-count", "invalid-utf8"],
)
def test_unreadable_day_csv_body_is_reextracted(tmp_path, body):
    data = tmp_path / "reallog"
    rlog = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    day_csv = data / "20261016" / "lateral_data.csv"
    day_csv.write_bytes(
        extractor._csv_header_bytes(extractor.csv_output_columns(True)) + body,
    )
    output = tmp_path / "out.csv"

    stats, state = extract_with_fake(data, output, {rlog: 7})

    assert stats["extracted_days"] == 1
    assert stats["reused_days"] == 0
    assert state["calls"][rlog] == 1
    assert [float(row["timestamp"]) for row in read_rows(day_csv)] == [7]
    assert [float(row["timestamp"]) for row in read_rows(output)] == [7]


def test_merge_separates_day_csvs_without_trailing_newline(tmp_path):
    header = extractor._csv_header_bytes(extractor.csv_output_columns(True))
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "combined.csv"
    first.write_bytes(header + b"first-row")
    second.write_bytes(header + b"second-row\n")

    rows_written = extractor._merge_csv_files([first, second], output, header)

    with open(output, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows_written == 2
    assert rows[1] == ["first-row"]
    assert rows[2] == ["second-row"]


def test_failed_merge_preserves_existing_combined_csv_and_cleans_temp(tmp_path):
    header = extractor._csv_header_bytes(extractor.csv_output_columns(True))
    valid = tmp_path / "valid.csv"
    invalid = tmp_path / "invalid.csv"
    output = tmp_path / "combined.csv"
    valid.write_bytes(header + b"valid-row\n")
    invalid.write_bytes(b"wrong,header\ninvalid-row\n")
    output.write_bytes(b"previous-complete-output\n")

    with pytest.raises(RuntimeError, match="Cached CSV has an invalid header"):
        extractor._merge_csv_files([valid, invalid], output, header)

    assert output.read_bytes() == b"previous-complete-output\n"
    assert not list(tmp_path.glob(".combined.csv.*.tmp"))


def test_cancel_during_worker_keeps_existing_outputs_and_cleans_temps(tmp_path):
    data = tmp_path / "reallog"
    first = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    touch_rlog(data / "20261016" / "route-a--1" / "rlog.zst")
    output = tmp_path / "out.csv"
    output.write_text("previous-complete-output\n", encoding="utf-8")
    cancel = threading.Event()
    state = make_state({first: 7})
    state["cancel_event"] = cancel

    with pytest.raises(RuntimeError, match="提取已取消"):
        extractor.extract_directory_to_csv(
            data,
            output,
            temporal=True,
            show_progress=False,
            worker_factory=FakeWorkerFactory(state),
            rlog_workers=1,
            cancel_event=cancel,
        )

    day_dir = data / "20261016"
    assert not (day_dir / "lateral_data.csv").exists()
    assert output.read_text(encoding="utf-8") == "previous-complete-output\n"
    assert not list(day_dir.glob(".lateral_data.csv.*.tmp"))
    assert not list(tmp_path.glob(".out.csv.*.tmp"))


def test_process_termination_never_exposes_partial_final_csv(tmp_path):
    data = tmp_path / "reallog"
    touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    output = tmp_path / "out.csv"
    output.write_text("previous-complete-output\n", encoding="utf-8")
    marker = tmp_path / "worker-started"
    script = r"""
import sys
import time
from nnlc_tools import extract_lateral_data as extractor

marker = sys.argv[1]

class BlockingWorker:
    files_processed = 0
    def process(self, request):
        open(marker, "w").close()
        with open(request["output_path"], "wb") as handle:
            handle.write(b"partial worker output")
        while True:
            time.sleep(0.05)
    def close(self):
        pass
    def interrupt(self):
        pass

extractor._RlogWorker = BlockingWorker
extractor.extract_directory_to_csv(
    sys.argv[2], sys.argv[3], temporal=True, show_progress=False,
    rlog_workers=1,
)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(marker), str(data), str(output)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), "extractor worker did not reach the blocking point"
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    assert not (data / "20261016" / "lateral_data.csv").exists()
    assert output.read_text(encoding="utf-8") == "previous-complete-output\n"


def test_cli_main_routes_csv_extraction_through_directory_helper(
    tmp_path, monkeypatch,
):
    data = tmp_path / "20261016"
    data.mkdir()
    output = tmp_path / "out.csv"
    captured = {}

    def fake_extract(input_dir, output_path, **kwargs):
        captured.update(input_dir=input_dir, output_path=output_path, **kwargs)
        write_day_csv(output_path, 12, "route-a", temporal=kwargs["temporal"])
        return {
            "rows_written": 1,
            "rows_seen": 1,
            "rows_filtered": 0,
            "skipped_rlogs": [],
            "exact_sort_fallbacks": 0,
            "max_lateness_seconds": 0.0,
            "reused_days": 0,
            "extracted_days": 1,
            "skipped_outside_date_dirs": [],
        }

    monkeypatch.setattr(extractor, "find_rlogs", lambda _path: ["dummy/rlog.zst"])
    monkeypatch.setattr(extractor, "extract_directory_to_csv", fake_extract)
    monkeypatch.setattr(
        "sys.argv",
        [
            "nnlc-extract", str(data), "-o", str(output), "--temporal",
            "--rlog-workers", "2",
        ],
    )

    extractor.main()

    assert captured["input_dir"] == str(data)
    assert captured["output_path"] == str(output)
    assert captured["temporal"] is True
    assert captured["rlog_workers"] == "2"
    assert captured["rlog_files"] == ["dummy/rlog.zst"]
    assert read_rows(output)[0]["timestamp"] == "12.0"


def test_training_step_invokes_shared_extractor_cli(tmp_path, monkeypatch):
    import nnlc_auto_train

    data = tmp_path / "reallog"
    data.mkdir()
    output_dir = tmp_path / "training-output"
    captured = {}

    def fake_run_command(command, description, **kwargs):
        captured["command"] = command
        captured["description"] = description
        output = Path(command[command.index("-o") + 1])
        write_day_csv(output, 5, "route-a")

    monkeypatch.setattr(nnlc_auto_train, "run_command", fake_run_command)

    output = nnlc_auto_train.step_extract(
        str(data), str(output_dir), "python", rlog_workers=3,
    )

    command = captured["command"]
    assert command[:3] == ["python", "-m", "nnlc_tools.extract_lateral_data"]
    assert command[command.index("-o") + 1] == str(output)
    assert "--temporal" in command
    assert command[command.index("--rlog-workers") + 1] == "3"
    assert Path(output).is_file()


def test_date_mode_skips_rlogs_outside_date_directories(tmp_path, capsys):
    data = tmp_path / "reallog"
    dated = touch_rlog(data / "20261016" / "route-a--0" / "rlog.zst")
    extra = touch_rlog(data / "misc" / "route-b--0" / "rlog.zst")
    output = tmp_path / "out.csv"

    stats, state = extract_with_fake(
        data, output, {dated: 3, extra: 4},
    )

    assert extra in stats["skipped_outside_date_dirs"]
    assert extra not in state["calls"]
    assert [float(row["timestamp"]) for row in read_rows(output)] == [3]
    assert not (data / "misc" / "lateral_data.csv").exists()
    assert "Skipping rlog outside YYYYMMDD directory" in capsys.readouterr().out
