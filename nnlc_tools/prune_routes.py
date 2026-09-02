#!/usr/bin/env python3
"""Prune extracted lateral data before coverage visualisation and training.

Two operations:
  1. Route-level: exclude entire routes scoring below --min-score
  2. Frame-level: drop saturated frames and lane-change frames

Usage:
  nnlc-prune-routes lateral_data.csv -o pruned_routes.csv
  nnlc-prune-routes lateral_data.csv --min-score 60 -o pruned_routes.csv
  nnlc-prune-routes lateral_data.csv --keep-saturated --keep-lane-change -o pruned_routes.csv
"""

import argparse
import sys

import pandas as pd

from nnlc_tools.score_routes import score_route


def parse_score_threshold(value):
    """Parse a route score threshold in the valid 0-100 range."""
    try:
        threshold = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--min-score must be an integer from 0 to 100") from exc
    if not 0 <= threshold <= 100:
        raise argparse.ArgumentTypeError("--min-score must be an integer from 0 to 100")
    return threshold


def _infer_route_col(df):
    """Return name of route column, or None if absent."""
    if "route_id" in df.columns:
        return "route_id"
    if "timestamp" in df.columns:
        dt = df["timestamp"].diff()
        df["route_id"] = ((dt > 60) | (dt < 0)).cumsum()
        return "route_id"
    return None


def prune_routes(df, min_score=0, drop_saturated=True, drop_lane_change=True):
    """Filter df by route score and frame-level flags.

    Returns (pruned_df, stats_dict).
    """
    initial_rows = len(df)
    route_rows_dropped = 0
    saturated_dropped = 0
    lane_change_dropped = 0

    # 1. Route-level: filter by score
    if min_score > 0:
        route_col = _infer_route_col(df)
        if route_col is None:
            raise ValueError(
                "--min-score requires a route_id or timestamp column; "
                "cannot safely infer route boundaries"
            )
        keep_routes = {
            route_id
            for route_id, group in df.groupby(route_col)
            if score_route(group)[0] >= min_score
        }
        before = len(df)
        df = df[df[route_col].isin(keep_routes)].copy()
        route_rows_dropped = before - len(df)

    # 2. Frame-level: drop saturated frames
    if drop_saturated and "saturated" in df.columns:
        before = len(df)
        df = df[~df["saturated"].astype(bool)]
        saturated_dropped = before - len(df)

    # 3. Frame-level: drop lane-change frames
    if drop_lane_change and "lane_change_state" in df.columns:
        lcs = df["lane_change_state"]
        before = len(df)
        if lcs.dtype == object:
            df = df[lcs == "off"]
        else:
            df = df[lcs == 0]
        lane_change_dropped = before - len(df)

    stats = {
        "initial_rows": initial_rows,
        "route_rows_dropped": route_rows_dropped,
        "saturated_dropped": saturated_dropped,
        "lane_change_dropped": lane_change_dropped,
        "output_rows": len(df),
    }
    return df, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prune lateral data by route score and frame-level flags.",
    )
    parser.add_argument("input", help="CSV/Parquet file from nnlc-extract")
    parser.add_argument("-o", "--output", default="pruned_routes.csv",
                        help="Output path (default: pruned_routes.csv)")
    parser.add_argument("--min-score", type=parse_score_threshold, default=0,
                        help="Exclude routes scoring below this value (default: 0, no exclusion)")
    parser.add_argument("--keep-saturated", action="store_true",
                        help="Do not drop saturated frames")
    parser.add_argument("--keep-lane-change", action="store_true",
                        help="Do not drop lane-change frames")
    args = parser.parse_args()

    from nnlc_tools.data_io import load_data
    df = load_data(args.input)
    if df is None:
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)

    # Count routes before pruning
    route_col = _infer_route_col(df)
    if args.min_score > 0 and route_col is None:
        parser.error("--min-score requires route_id or timestamp column")
    n_routes_initial = df[route_col].nunique() if route_col else 1

    print(f"Loaded {len(df):,} rows across {n_routes_initial} routes")

    pruned, stats = prune_routes(
        df,
        min_score=args.min_score,
        drop_saturated=not args.keep_saturated,
        drop_lane_change=not args.keep_lane_change,
    )

    if args.min_score > 0:
        n_routes_kept = pruned[route_col].nunique() if route_col else 1
        n_routes_dropped = n_routes_initial - n_routes_kept
        print(f"Route filter (score >= {args.min_score}): "
              f"dropped {n_routes_dropped} routes ({stats['route_rows_dropped']:,} rows)")

    if not args.keep_saturated:
        print(f"Saturated frames dropped:   {stats['saturated_dropped']:,}")

    if not args.keep_lane_change:
        print(f"Lane-change frames dropped: {stats['lane_change_dropped']:,}")

    if args.output.endswith(".parquet"):
        pruned.to_parquet(args.output, index=False)
    else:
        pruned.to_csv(args.output, index=False)

    print(f"Output: {stats['output_rows']:,} rows written to {args.output}")


if __name__ == "__main__":
    main()
