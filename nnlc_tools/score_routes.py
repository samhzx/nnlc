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

CRITERIA = [
    ("high_override",    lambda df: df["steering_pressed"].mean() > 0.10,                          -15, ">10% steering override"),
    ("high_saturated",   lambda df: df["saturated"].mean() > 0.05,                                 -20, ">5% saturated"),
    ("low_active",       lambda df: df["active"].mean() < 0.80,                                    -25, "<80% active"),
    ("high_standstill",  lambda df: df["standstill"].mean() > 0.30,                                -15, ">30% standstill"),
    ("high_lane_change", lambda df: (df["lane_change_state"] != 0).mean() > 0.10,                  -10, ">10% lane change"),
    ("too_short",        lambda df: df["active"].astype(bool).sum() * 0.01 < 120,                  -20, "<2 min active driving"),
]


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
    score = 100
    flags = []

    for name, check_fn, penalty, desc in CRITERIA:
        try:
            if check_fn(df):
                score += penalty  # penalty is negative
                flags.append(desc)
        except (KeyError, TypeError, ZeroDivisionError):
            pass

    return max(0, score), flags


def load_data_with_routes(input_path):
    """Load data from CSV, Parquet, or directory of rlogs with route tracking."""
    if os.path.isfile(input_path):
        from nnlc_tools.data_io import load_data
        df = load_data(input_path)
        if df is None:
            print(f"ERROR: Input not found: {input_path}")
            sys.exit(1)
        return df, None

    if os.path.isdir(input_path):
        # Process rlogs directly — need per-file tracking for route grouping
        from nnlc_tools.extract_lateral_data import find_rlogs, extract_segment, COLUMNS
        rlog_files = find_rlogs(input_path)
        if not rlog_files:
            print(f"ERROR: No rlog files found in {input_path}")
            sys.exit(1)

        # Build DataFrame with source file info for route grouping
        all_rows = []
        file_map = []  # (start_idx, end_idx, rlog_path)
        for rlog_path in rlog_files:
            rows = extract_segment(rlog_path)
            start = len(all_rows)
            all_rows.extend(rows)
            file_map.append((start, len(all_rows), rlog_path))

        df = pd.DataFrame(all_rows, columns=COLUMNS)

        # Add route_id column based on file paths
        df["route_id"] = "unknown"
        for start, end, path in file_map:
            if start < end:
                df.loc[start:end - 1, "route_id"] = extract_route_id(path)

        return df, "route_id"

    print(f"ERROR: Input not found: {input_path}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Score route quality for NNLC training data.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("--min-score", type=parse_score_threshold, default=0,
                        help="Only show routes with score >= this value")
    args = parser.parse_args()

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
