from pathlib import Path
import threading

import pandas as pd
import pytest

from nnlc_tools.analyze_interventions import prune_override_rows
from nnlc_tools.prune_routes import prepare_training_csv, prune_routes
from nnlc_tools.score_routes import (
    keep_routes_from_scores,
    recommend_min_score,
    score_csv_stream,
)


COLUMNS = [
    "timestamp", "route_id", "v_ego", "steering_pressed", "saturated",
    "active", "standstill", "lane_change_state", "desired_lateral_accel",
]


def _row(timestamp, route_id, *, pressed=False, saturated=False, active=True,
         standstill=False, lane_change=0, speed=12.0, desired=0.2):
    return {
        "timestamp": timestamp,
        "route_id": route_id,
        "v_ego": speed,
        "steering_pressed": pressed,
        "saturated": saturated,
        "active": active,
        "standstill": standstill,
        "lane_change_state": lane_change,
        "desired_lateral_accel": desired,
    }


def _sample_frame():
    rows = []
    timestamp = 0.0
    # Long clean route: score 100.
    for index in range(15000):
        rows.append(_row(timestamp, "good", pressed=False))
        timestamp += 0.01
    # Short dirty route: low activity, overrides, saturation, lane change.
    for index in range(80):
        rows.append(_row(
            timestamp, "bad",
            pressed=index % 2 == 0,
            saturated=index % 5 == 0,
            active=index % 4 != 0,
            standstill=index % 6 == 0,
            lane_change=1 if index % 3 == 0 else 0,
        ))
        timestamp += 0.01
    # Clean route with a compact override event plus a 2-frame gap.
    for index in range(400):
        pressed = 40 <= index <= 46 or 49 <= index <= 52
        rows.append(_row(timestamp, "override", pressed=pressed))
        timestamp += 0.01
    return pd.DataFrame(rows, columns=COLUMNS)


def _legacy_prepare(df, min_score):
    pruned, _ = prune_routes(df, min_score=min_score)
    final_df, _ = prune_override_rows(pruned, prune="both")
    return final_df.reset_index(drop=True)


def test_prepare_training_csv_matches_legacy(tmp_path):
    source = _sample_frame()
    input_csv = tmp_path / "lateral_data.csv"
    source.to_csv(input_csv, index=False)

    route_results, error = score_csv_stream(input_csv)
    assert error is None
    min_score = recommend_min_score(route_results)
    keep_routes = keep_routes_from_scores(route_results, min_score)
    expected = _legacy_prepare(source, min_score)

    in_memory_csv = tmp_path / "final_memory.csv"
    memory_stats = prepare_training_csv(
        str(input_csv), str(in_memory_csv), min_score=min_score,
        keep_routes=keep_routes, route_results=route_results, streaming=False)
    memory_df = pd.read_csv(in_memory_csv)

    streaming_csv = tmp_path / "final_stream.csv"
    stream_stats = prepare_training_csv(
        str(input_csv), str(streaming_csv), min_score=min_score,
        keep_routes=keep_routes, route_results=route_results, streaming=True,
        chunksize=64)
    stream_df = pd.read_csv(streaming_csv)

    pd.testing.assert_frame_equal(memory_df.reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(stream_df.reset_index(drop=True), expected)
    assert memory_stats["output_rows"] == len(expected)
    assert stream_stats["output_rows"] == len(expected)
    assert "bad" not in set(memory_df["route_id"].map(str))
    assert memory_stats["intervention_event_count"] >= 1


def test_recommend_min_score_thresholds():
    assert recommend_min_score([{"score": 80}] * 2 + [{"score": 50}] * 8) == 60
    assert recommend_min_score([{"score": 80}] * 4 + [{"score": 50}] * 6) == 70
    assert recommend_min_score([{"score": 50}] * 10) == 40


def test_streaming_preserves_original_route_column_and_schema(tmp_path):
    source = pd.DataFrame([
        _row(index, "route-A")
        for index in range(13_000)
    ], columns=COLUMNS)
    input_csv = tmp_path / "route_input.csv"
    output_csv = tmp_path / "route_output.csv"
    source.to_csv(input_csv, index=False)

    route_results, error = score_csv_stream(input_csv)
    assert error is None
    prepare_training_csv(
        str(input_csv), str(output_csv), min_score=60,
        keep_routes=keep_routes_from_scores(route_results, 60),
        route_results=route_results, streaming=True, chunksize=257,
    )
    result = pd.read_csv(output_csv)

    assert list(result.columns) == COLUMNS
    assert set(result["route_id"]) == {"route-A"}
    assert "_nnlc_route_key" not in result.columns


def test_streaming_score_honors_cancellation(tmp_path):
    source = pd.DataFrame([
        _row(index, "route-A")
        for index in range(13_000)
    ], columns=COLUMNS)
    input_csv = tmp_path / "cancel_input.csv"
    source.to_csv(input_csv, index=False)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(RuntimeError, match="训练已取消"):
        score_csv_stream(input_csv, cancel_event=cancel_event)
