#!/usr/bin/env python3
"""Score route quality for NNLC training data.

Evaluates routes based on override rate, saturation, activity, standstill time,
lane changes, and minimum duration.

Usage:
  python -m nnlc_tools.score_routes /path/to/rlogs/
  python -m nnlc_tools.score_routes lateral_data.csv
"""

import argparse
import os
import re
import sys

import pandas as pd

from nnlc_tools.bool_utils import parse_bool_series
from nnlc_tools.streaming_data import DEFAULT_CHUNK_ROWS, iter_csv_chunks

CRITERIA = [
    ("high_override",    lambda df: df["steering_pressed"].mean() > 0.10,                          -15, ">10% steering override"),
    ("high_saturated",   lambda df: df["saturated"].mean() > 0.05,                                 -20, ">5% saturated"),
    ("low_active",       lambda df: df["active"].mean() < 0.80,                                    -25, "<80% active"),
    ("high_standstill",  lambda df: df["standstill"].mean() > 0.30,                                -15, ">30% standstill"),
    ("high_lane_change", lambda df: (df["lane_change_state"] != 0).mean() > 0.10,                  -10, ">10% lane change"),
    ("too_short",        lambda df: df["active"].sum() * 0.01 < 120,                              -20, "<2 min active driving"),
]

REQUIRED_SCORE_COLUMNS = {
    "steering_pressed", "saturated", "active", "standstill", "lane_change_state"
}


def parse_score_threshold(value):
    """Parse a route score threshold in the valid 0-100 range."""
    try:
        threshold = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--min-score must be an integer from 0 to 100") from exc
    if not 0 <= threshold <= 100:
        raise argparse.ArgumentTypeError("--min-score must be an integer from 0 to 100")
    return threshold


def extract_route_id(path):
    """Extract route ID from rlog path by stripping --segment_num suffix.

    Paths look like: .../2024-01-15--12-30-45/0/rlog.zst
    Route ID is: 2024-01-15--12-30-45
    """
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts):
        # Match openpilot route ID pattern: hex|date--time
        if re.match(r"^[0-9a-f]+\|?\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$", part):
            return part
        if re.match(r"^\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$", part):
            return part
    # Fallback: use parent directory name
    for i, part in enumerate(parts):
        if part in ("rlog", "rlog.zst", "rlog.bz2"):
            # Go up 2 levels (skip segment number directory)
            if i >= 2:
                return parts[i - 2]
            elif i >= 1:
                return parts[i - 1]
    return "unknown"


def score_route(df):
    """Score a single route's data. Returns (score, list of triggered flags)."""
    missing = sorted(REQUIRED_SCORE_COLUMNS.difference(df.columns))
    if df.empty:
        return 0, ["empty route"]
    if missing:
        return 0, [f"missing fields: {', '.join(missing)}"]

    # Normalize flags before calculating means.  In particular,
    # bool("False") is True and would otherwise corrupt every criterion.
    df = df.copy()
    for column in ("steering_pressed", "saturated", "active", "standstill"):
        df[column] = parse_bool_series(df[column])

    score = 100
    flags = []

    for name, check_fn, penalty, desc in CRITERIA:
        try:
            if check_fn(df):
                score += penalty  # penalty is negative
                flags.append(desc)
        except (KeyError, TypeError, ZeroDivisionError):
            # A malformed criterion must never leave a route at a misleading
            # perfect score.  Mark it unusable and retain the reason.
            score = 0
            flags.append(f"unable to evaluate {name}")

    return max(0, score), flags


def load_data_with_routes(input_path):
    """Load data from CSV, Parquet, or directory of rlogs with route tracking."""
    if os.path.isfile(input_path):
        from nnlc_tools.data_io import load_data
        df = load_data(input_path)
        if df is None:
            print(f"ERROR: Input not found: {input_path}")
            sys.exit(1)
        return df, "route_id" if "route_id" in df.columns else None

    if os.path.isdir(input_path):
        # Process rlogs directly — need per-file tracking for route grouping
        import tempfile
        from nnlc_tools.extract_lateral_data import (
            find_rlogs, extract_segment, _StreamingCsvWriter, extract_route_id,
        )
        rlog_files = find_rlogs(input_path)
        if not rlog_files:
            print(f"ERROR: No rlog files found in {input_path}")
            sys.exit(1)

        temp_file = tempfile.NamedTemporaryFile(prefix="nnlc_score_", suffix=".csv", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        stream = _StreamingCsvWriter(temp_path)
        try:
            for rlog_path in rlog_files:
                stream.route_id = extract_route_id(rlog_path)
                extract_segment(rlog_path, row_callback=stream.accept)
                stream.finish_segment()
            stream.finish()
            if stream.rows_written == 0:
                print(f"ERROR: No data extracted from {input_path}")
                sys.exit(1)
            return pd.read_csv(temp_path), "route_id"
        finally:
            stream.close()
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    print(f"ERROR: Input not found: {input_path}")
    sys.exit(1)


def score_csv_stream(input_path, chunksize=DEFAULT_CHUNK_ROWS):
    """Score a CSV while retaining only route-level counters."""
    aggregates = {}
    previous_timestamp = None
    inferred_route = 0
    for chunk in iter_csv_chunks(input_path, chunksize=chunksize):
        chunk = chunk.copy()
        if "route_id" not in chunk.columns:
            if "timestamp" in chunk.columns:
                timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce")
                breaks = []
                for value in timestamps:
                    is_break = (
                        previous_timestamp is not None
                        and pd.notna(value)
                        and (value - previous_timestamp > 60 or value < previous_timestamp)
                    )
                    if is_break:
                        inferred_route += 1
                    breaks.append(inferred_route)
                    if pd.notna(value):
                        previous_timestamp = value
                chunk["route_id"] = breaks
            else:
                chunk["route_id"] = 0

        missing = sorted(REQUIRED_SCORE_COLUMNS.difference(chunk.columns))
        if missing:
            return None, f"missing fields: {', '.join(missing)}"
        for column in ("steering_pressed", "saturated", "active", "standstill"):
            chunk[column] = parse_bool_series(chunk[column])
        for route_id, group in chunk.groupby("route_id", sort=False, dropna=False):
            key = str(route_id)
            state = aggregates.setdefault(key, {
                "rows": 0, "override": 0, "saturated": 0,
                "active": 0, "standstill": 0, "lane_change": 0,
            })
            state["rows"] += len(group)
            state["override"] += int(group["steering_pressed"].sum())
            state["saturated"] += int(group["saturated"].sum())
            state["active"] += int(group["active"].sum())
            state["standstill"] += int(group["standstill"].sum())
            state["lane_change"] += int((group["lane_change_state"] != 0).sum())

    results = []
    for route_id, state in aggregates.items():
        rows = state["rows"]
        if rows <= 0:
            continue
        score = 100
        flags = []
        checks = [
            (state["override"] / rows > .10, -15, ">10% steering override"),
            (state["saturated"] / rows > .05, -20, ">5% saturated"),
            (state["active"] / rows < .80, -25, "<80% active"),
            (state["standstill"] / rows > .30, -15, ">30% standstill"),
            (state["lane_change"] / rows > .10, -10, ">10% lane change"),
            (state["active"] * .01 < 120, -20, "<2 min active driving"),
        ]
        for triggered, penalty, description in checks:
            if triggered:
                score += penalty
                flags.append(description)
        results.append({
            "route_id": route_id,
            "score": max(0, score),
            "duration_s": round(rows * .01, 1),
            "rows": rows,
            "issues": ", ".join(flags),
        })
    return results, None


def main():
    parser = argparse.ArgumentParser(
        description="Score route quality for NNLC training data.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("--min-score", type=parse_score_threshold, default=0,
                        help="Only show routes with score >= this value")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream CSV in bounded chunks instead of loading it fully")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"Rows per streaming chunk (default: {DEFAULT_CHUNK_ROWS:,})")
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be a positive integer")
    if args.streaming:
        if not os.path.isfile(args.input) or not args.input.lower().endswith(".csv"):
            parser.error("--streaming only supports an existing CSV file")
        try:
            results, error = score_csv_stream(args.input, chunksize=args.chunk_rows)
        except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        if error:
            print(f"ERROR: {error}")
            sys.exit(1)
        if not results:
            print("ERROR: Input contains no data rows")
            sys.exit(1)
        results_df = pd.DataFrame(results).sort_values("score", ascending=False)
        if args.min_score > 0:
            results_df = results_df[results_df["score"] >= args.min_score]
        print(f"\n{'Route ID':<45} {'Score':>5} {'Duration':>10} {'Rows':>8}  Issues")
        print("-" * 120)
        for _, row in results_df.iterrows():
            route_str = str(row["route_id"])[:44]
            print(f"{route_str:<45} {row['score']:>5} {row['duration_s']:>9.1f}s {row['rows']:>8}  {row['issues']}")
        print(f"\n{len(results_df)} routes scored")
        print(f"  {len(results_df[results_df['score'] >= 70])} routes with score >= 70 (recommended for training)")
        return

    df, route_col = load_data_with_routes(args.input)

    if df.empty:
        print("ERROR: Input contains no data rows")
        sys.exit(1)

    if route_col is None:
        # CSV/Parquet without route_id — try to infer from timestamp gaps
        # Group by large time gaps (>60s between rows = new route)
        if "timestamp" in df.columns:
            dt = df["timestamp"].diff()
            route_breaks = (dt > 60) | (dt < 0)
            df["route_id"] = route_breaks.cumsum()
            route_col = "route_id"
        else:
            # Score entire dataset as one route
            score, flags = score_route(df)
            duration = len(df) * 0.01  # ~100Hz
            print(f"\nOverall score: {score}/100  Duration: {duration:.0f}s")
            if flags:
                print(f"  Issues: {', '.join(flags)}")
            return

    # Score each route
    results = []
    for route_id, group in df.groupby(route_col):
        score, flags = score_route(group)
        duration = len(group) * 0.01  # ~100Hz
        results.append({
            "route_id": route_id,
            "score": score,
            "duration_s": round(duration, 1),
            "rows": len(group),
            "issues": ", ".join(flags) if flags else "",
        })

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)

    if args.min_score > 0:
        results_df = results_df[results_df["score"] >= args.min_score]

    # Print results
    print(f"\n{'Route ID':<45} {'Score':>5} {'Duration':>10} {'Rows':>8}  Issues")
    print("-" * 120)
    for _, row in results_df.iterrows():
        route_str = str(row["route_id"])[:44]
        print(f"{route_str:<45} {row['score']:>5} {row['duration_s']:>9.1f}s {row['rows']:>8}  {row['issues']}")

    print(f"\n{len(results_df)} routes scored")
    good = len(results_df[results_df["score"] >= 70])
    print(f"  {good} routes with score >= 70 (recommended for training)")


if __name__ == "__main__":
    main()
