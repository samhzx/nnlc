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
import os
import sys

import pandas as pd

from nnlc_tools.bool_utils import parse_bool_series
from nnlc_tools.analyze_interventions import prune_override_rows
from nnlc_tools.score_routes import (
    attach_route_ids,
    keep_routes_from_scores,
    score_csv_stream,
    score_route,
)
from nnlc_tools.streaming_data import (
    DEFAULT_CHUNK_ROWS,
    iter_csv_chunks,
    raise_if_cancelled,
    require_distinct_paths,
)


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


def prune_routes(df, min_score=0, drop_saturated=True, drop_lane_change=True,
                 keep_routes=None):
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
        if keep_routes is None:
            keep_routes = {
                route_id
                for route_id, group in df.groupby(route_col)
                if score_route(group)[0] >= min_score
            }
        keep_keys = {str(route_id) for route_id in keep_routes}
        before = len(df)
        df = df[df[route_col].map(str).isin(keep_keys)].copy()
        route_rows_dropped = before - len(df)

    # 2. Frame-level: drop saturated frames
    if drop_saturated and "saturated" in df.columns:
        before = len(df)
        df = df[~parse_bool_series(df["saturated"])]
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


def prune_csv_stream(input_path, output_path, min_score=0,
                     drop_saturated=True, drop_lane_change=True,
                     chunksize=DEFAULT_CHUNK_ROWS, keep_routes=None,
                     route_results=None, cancel_event=None):
    """Prune a CSV incrementally and return the same statistics as prune_routes."""
    require_distinct_paths(input_path, output_path)
    if min_score > 0 and keep_routes is None:
        if route_results is None:
            route_results, error = score_csv_stream(
                input_path, chunksize=chunksize, cancel_event=cancel_event)
            if error:
                raise ValueError(error)
        keep_routes = keep_routes_from_scores(route_results, min_score)

    initial_rows = route_rows_dropped = saturated_dropped = lane_change_dropped = 0
    output_rows = 0
    wrote_header = False
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    previous_timestamp = None
    inferred_route = 0
    try:
        chunks = iter_csv_chunks(input_path, chunksize=chunksize)
        with open(output_path, "w", newline="", encoding="utf-8") as output:
            for chunk in chunks:
                raise_if_cancelled(cancel_event)
                initial_rows += len(chunk)
                if min_score > 0:
                    if "route_id" not in chunk.columns and "timestamp" not in chunk.columns:
                        raise ValueError(
                            "--min-score requires a route_id or timestamp column; "
                            "cannot safely infer route boundaries"
                        )
                    route_ids, previous_timestamp, inferred_route = attach_route_ids(
                        chunk, previous_timestamp, inferred_route)
                    chunk = chunk.copy()
                    route_key = route_ids
                    if "route_id" not in chunk.columns:
                        chunk["route_id"] = route_key
                    before = len(chunk)
                    chunk = chunk[route_key.map(str).isin(keep_routes)]
                    route_rows_dropped += before - len(chunk)
                if drop_saturated and "saturated" in chunk.columns:
                    before = len(chunk)
                    chunk = chunk[~parse_bool_series(chunk["saturated"])]
                    saturated_dropped += before - len(chunk)
                if drop_lane_change and "lane_change_state" in chunk.columns:
                    before = len(chunk)
                    lcs = chunk["lane_change_state"]
                    chunk = chunk[lcs == "off"] if lcs.dtype == object else chunk[lcs == 0]
                    lane_change_dropped += before - len(chunk)
                if not wrote_header:
                    chunk.head(0).to_csv(output, index=False)
                    wrote_header = True
                if not chunk.empty:
                    chunk.to_csv(output, header=False, index=False)
                    output_rows += len(chunk)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        raise ValueError("input CSV is empty or missing")
    if not wrote_header:
        raise ValueError("input CSV contains no data rows")
    return {
        "initial_rows": initial_rows,
        "route_rows_dropped": route_rows_dropped,
        "saturated_dropped": saturated_dropped,
        "lane_change_dropped": lane_change_dropped,
        "output_rows": output_rows,
    }


def prepare_training_rows(df, min_score=0, drop_saturated=True, drop_lane_change=True,
                          keep_routes=None, prune="both", gap_frames=5, cfg=None):
    """Apply route pruning and override removal without writing intermediate CSVs."""
    pruned, prune_stats = prune_routes(
        df,
        min_score=min_score,
        drop_saturated=drop_saturated,
        drop_lane_change=drop_lane_change,
        keep_routes=keep_routes,
    )
    final_df, intervention_stats = prune_override_rows(
        pruned, prune=prune, gap_frames=gap_frames, cfg=cfg)
    stats = dict(prune_stats)
    stats.update({
        "intervention_active_rows": intervention_stats["active_rows"],
        "intervention_dropped_rows": intervention_stats["dropped_rows"],
        "intervention_event_count": intervention_stats["event_count"],
        "output_rows": intervention_stats["output_rows"],
    })
    return final_df, stats


def _score_loaded_dataframe(df, min_score):
    route_col = _infer_route_col(df) if min_score > 0 else (
        "route_id" if "route_id" in df.columns else None)
    if min_score > 0 and route_col is None:
        raise ValueError(
            "--min-score requires a route_id or timestamp column; "
            "cannot safely infer route boundaries"
        )
    route_results = []
    if route_col is not None:
        for route_id, group in df.groupby(route_col):
            score, flags = score_route(group)
            route_results.append({
                "route_id": route_id,
                "score": score,
                "duration_s": round(len(group) * 0.01, 1),
                "rows": len(group),
                "issues": ", ".join(flags) if flags else "",
            })
    elif len(df):
        score, flags = score_route(df)
        route_results.append({
            "route_id": 0,
            "score": score,
            "duration_s": round(len(df) * 0.01, 1),
            "rows": len(df),
            "issues": ", ".join(flags) if flags else "",
        })
    keep_routes = keep_routes_from_scores(route_results, min_score)
    return route_results, keep_routes


def _write_training_frame(df, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if output_path.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)


def prepare_training_csv(input_path, output_path, min_score=0,
                         drop_saturated=True, drop_lane_change=True,
                         keep_routes=None, route_results=None,
                         prune="both", gap_frames=5, cfg=None,
                         streaming=False, chunksize=DEFAULT_CHUNK_ROWS,
                         cancel_event=None):
    """Reuse scores when available, then write the final training CSV."""
    require_distinct_paths(input_path, output_path)
    if streaming:
        if not input_path.lower().endswith(".csv"):
            raise ValueError("streaming only supports CSV input")
        if route_results is None:
            route_results, error = score_csv_stream(
                input_path, chunksize=chunksize, cancel_event=cancel_event)
            if error:
                raise ValueError(error)
        if keep_routes is None:
            keep_routes = keep_routes_from_scores(route_results, min_score)
        stats = _prepare_training_csv_stream(
            input_path, output_path, min_score=min_score,
            drop_saturated=drop_saturated, drop_lane_change=drop_lane_change,
            keep_routes=keep_routes, prune=prune, gap_frames=gap_frames,
            cfg=cfg, chunksize=chunksize, cancel_event=cancel_event)
        stats["route_results"] = route_results
        return stats

    from nnlc_tools.data_io import load_data
    raise_if_cancelled(cancel_event)
    df = load_data(input_path)
    if df is None:
        raise ValueError(f"input not found: {input_path}")
    raise_if_cancelled(cancel_event)
    if route_results is None:
        route_results, keep_routes = _score_loaded_dataframe(df, min_score)
    elif keep_routes is None:
        keep_routes = keep_routes_from_scores(route_results, min_score)
    final_df, stats = prepare_training_rows(
        df, min_score=min_score, drop_saturated=drop_saturated,
        drop_lane_change=drop_lane_change, keep_routes=keep_routes,
        prune=prune, gap_frames=gap_frames, cfg=cfg)
    _write_training_frame(final_df, output_path)
    raise_if_cancelled(cancel_event)
    stats["route_results"] = route_results
    return stats


def _iter_pruned_training_chunks(input_path, min_score=0, drop_saturated=True,
                                 drop_lane_change=True, keep_routes=None,
                                 chunksize=DEFAULT_CHUNK_ROWS,
                                 cancel_event=None):
    """Yield pruned chunks for the combined second pass."""
    previous_timestamp = None
    inferred_route = 0
    stats = {
        "initial_rows": 0,
        "route_rows_dropped": 0,
        "saturated_dropped": 0,
        "lane_change_dropped": 0,
        "columns": None,
    }
    route_key_column = "_nnlc_route_key"
    for chunk in iter_csv_chunks(input_path, chunksize=chunksize):
        raise_if_cancelled(cancel_event)
        stats["initial_rows"] += len(chunk)
        route_ids, previous_timestamp, inferred_route = attach_route_ids(
            chunk, previous_timestamp, inferred_route)
        chunk = chunk.copy()
        if route_key_column in chunk.columns:
            raise ValueError(f"input CSV already contains reserved column: {route_key_column}")
        chunk[route_key_column] = route_ids
        if min_score > 0:
            if keep_routes is None:
                raise ValueError("keep_routes is required when min_score > 0")
            before = len(chunk)
            chunk = chunk[chunk[route_key_column].map(str).isin(keep_routes)]
            stats["route_rows_dropped"] += before - len(chunk)
        if drop_saturated and "saturated" in chunk.columns:
            before = len(chunk)
            chunk = chunk[~parse_bool_series(chunk["saturated"])]
            stats["saturated_dropped"] += before - len(chunk)
        if drop_lane_change and "lane_change_state" in chunk.columns:
            before = len(chunk)
            lcs = chunk["lane_change_state"]
            chunk = chunk[lcs == "off"] if lcs.dtype == object else chunk[lcs == 0]
            stats["lane_change_dropped"] += before - len(chunk)
        if stats["columns"] is None:
            stats["columns"] = list(chunk.columns)
        if not chunk.empty:
            yield chunk, stats
    if stats["columns"] is None:
        raise ValueError("input CSV is empty or missing")


def _prepare_training_csv_stream(input_path, output_path, min_score=0,
                                 drop_saturated=True, drop_lane_change=True,
                                 keep_routes=None, prune="both", gap_frames=5,
                                 cfg=None, chunksize=DEFAULT_CHUNK_ROWS,
                                 cancel_event=None):
    """Filter frames while streaming and drop override events without a temp CSV."""
    from nnlc_tools.analyze_interventions import prune_csv_stream

    prune_stats = None

    def filtered_chunks():
        nonlocal prune_stats
        for chunk, stats in _iter_pruned_training_chunks(
                input_path, min_score=min_score, drop_saturated=drop_saturated,
                drop_lane_change=drop_lane_change, keep_routes=keep_routes,
                chunksize=chunksize, cancel_event=cancel_event):
            prune_stats = stats
            yield chunk
        if prune_stats is None:
            # Empty input still needs to raise the same error as the pruner.
            list(_iter_pruned_training_chunks(
                input_path, min_score=min_score, drop_saturated=drop_saturated,
                drop_lane_change=drop_lane_change, keep_routes=keep_routes,
                chunksize=chunksize, cancel_event=cancel_event))

    intervention_stats = prune_csv_stream(
        input_path, output_path, prune=prune, gap_frames=gap_frames,
        cfg=cfg, chunksize=chunksize, chunks=filtered_chunks(),
        route_column="_nnlc_route_key", drop_columns={"_nnlc_route_key"},
        cancel_event=cancel_event)
    if prune_stats is None:
        raise ValueError("input CSV is empty or missing")
    return {
        "initial_rows": prune_stats["initial_rows"],
        "route_rows_dropped": prune_stats["route_rows_dropped"],
        "saturated_dropped": prune_stats["saturated_dropped"],
        "lane_change_dropped": prune_stats["lane_change_dropped"],
        "intervention_active_rows": intervention_stats["active_rows"],
        "intervention_dropped_rows": intervention_stats["dropped_rows"],
        "intervention_event_count": intervention_stats["event_count"],
        "output_rows": intervention_stats["output_rows"],
    }


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
    parser.add_argument("--streaming", action="store_true",
                        help="Process CSV in bounded chunks instead of loading it fully")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"Rows per streaming chunk (default: {DEFAULT_CHUNK_ROWS:,})")
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be a positive integer")
    if args.streaming:
        if not args.input.lower().endswith(".csv"):
            parser.error("--streaming only supports CSV input")
        try:
            stats = prune_csv_stream(
                args.input, args.output, min_score=args.min_score,
                drop_saturated=not args.keep_saturated,
                drop_lane_change=not args.keep_lane_change,
                chunksize=args.chunk_rows,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"Loaded {stats['initial_rows']:,} rows")
        if args.min_score > 0:
            print(f"Route filter (score >= {args.min_score}): dropped {stats['route_rows_dropped']:,} rows")
        if not args.keep_saturated:
            print(f"Saturated frames dropped:   {stats['saturated_dropped']:,}")
        if not args.keep_lane_change:
            print(f"Lane-change frames dropped: {stats['lane_change_dropped']:,}")
        print(f"Output: {stats['output_rows']:,} rows written to {args.output}")
        return

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
