#!/usr/bin/env python3
"""Classify steering override events as genuine driver interventions or mechanical disturbances.

steering_pressed=True fires on both intentional driver corrections and mechanical disturbances
(potholes, bumps, curb impacts). This tool segments override events and classifies each one
using the three-stage cascade classifier from steering_classifier/.

Usage:
  python -m nnlc_tools.analyze_interventions output/lateral_data.csv
  python -m nnlc_tools.analyze_interventions output/lateral_data.csv --plot -o output/interventions.png
"""

import argparse
import sys

import numpy as np
import pandas as pd

from nnlc_tools.streaming_data import (
    DEFAULT_CHUNK_ROWS,
    iter_csv_chunks,
    open_csv_writer,
    require_distinct_paths,
)


# Default thresholds (legacy heuristic path — kept for backward compat)
DEFAULT_MIN_DURATION = 0.15    # seconds — rule_brief threshold
DEFAULT_A_EGO_THRESH = 1.5     # m/s² — longitudinal shock threshold
DEFAULT_CONSISTENCY_THRESH = 0.65  # fraction — torque sign consistency threshold
DEFAULT_MIN_SCORE = 2          # rules that must fire to classify as mechanical
DEFAULT_GAP_FRAMES = 5         # frames of allowed gap within one event
DEFAULT_TORQUE_RATE_THRESH = 500.0  # Nm/s — peak |dT/dt| threshold for rule_sharp_onset
DEFAULT_HIGHWAY_SPEED_THRESH = 20.0  # m/s (~45 mph) — speed above which rule_highway_brief applies
DEFAULT_HIGHWAY_MIN_DURATION = 0.4   # seconds — duration below which rule_highway_brief fires at speed


def load_data(input_path):
    """Load data from CSV, Parquet, or directory of rlogs."""
    from nnlc_tools.data_io import load_data as _load_data
    df = _load_data(input_path)
    if df is None:
        print(f"ERROR: No data found at {input_path}")
        sys.exit(1)
    return df


def segment_events(df, gap_frames=DEFAULT_GAP_FRAMES):
    """Group consecutive steering_pressed=True rows into events.

    Events separated by <= gap_frames of non-pressed rows are merged.

    Returns a list of dicts with event metadata.
    """
    if "steering_pressed" not in df.columns:
        return []

    sp = df["steering_pressed"].astype(bool).values
    routes = df["route_id"].map(str).values if "route_id" in df.columns else None
    n = len(sp)

    events = []
    i = 0
    while i < n:
        if not sp[i]:
            i += 1
            continue

        # Start of a new event
        start = i
        end = i

        # Extend, merging over short gaps
        j = i + 1
        while j < n:
            if routes is not None and routes[j] != routes[start]:
                break
            if sp[j]:
                end = j
                j += 1
            else:
                # Look ahead to see if there's another pressed region within gap_frames
                gap_end = j
                while (gap_end < n and not sp[gap_end]
                       and (gap_end - j) < gap_frames
                       and (routes is None or routes[gap_end] == routes[start])):
                    gap_end += 1
                if gap_end < n and sp[gap_end]:
                    # Merge: skip the gap
                    j = gap_end
                else:
                    break

        events.append({"start_idx": start, "end_idx": end})
        i = end + 1

    return events


def classify_events_cascade(active_df, events, cfg=None):
    """Classify events using the three-stage cascade classifier.

    Extracts raw signal arrays per event window, calls extract_features() then
    classify_event() from the steering_classifier package, and returns a unified
    events DataFrame.

    Returns a DataFrame with one row per event including all cascade features,
    classification columns, and aggregates needed by print_summary() / mark_rows().
    """
    from nnlc_tools.steering_classifier.features import extract_features
    from nnlc_tools.steering_classifier.cascade import classify_event
    from nnlc_tools.steering_classifier.config import ClassifierConfig

    if cfg is None:
        cfg = ClassifierConfig()

    def _col_array(chunk, col, fallback_val=0.0):
        """Extract column as numpy array, filling missing with fallback_val."""
        if col in active_df.columns:
            arr = chunk[col].values.astype(float)
            arr = np.where(np.isnan(arr), fallback_val, arr)
            return arr
        return np.full(len(chunk), fallback_val)

    def _col_array_nan(chunk, col):
        """Extract column as numpy array, leaving NaN for missing values."""
        if col in active_df.columns:
            return chunk[col].values.astype(float)
        return np.full(len(chunk), np.nan)

    rows = []

    for evt_id, evt in enumerate(events):
        s, e = evt["start_idx"], evt["end_idx"]
        chunk = active_df.iloc[s : e + 1]
        n_frames = len(chunk)

        # Raw signal arrays for extract_features()
        steering_torque = _col_array(chunk, "steering_torque", 0.0)
        steering_angle_deg = _col_array(chunk, "steering_angle_deg", 0.0)
        actual_lateral_accel = _col_array_nan(chunk, "actual_lateral_accel")
        desired_lateral_accel = _col_array_nan(chunk, "desired_lateral_accel")
        a_ego = _col_array_nan(chunk, "a_ego")
        v_ego = _col_array_nan(chunk, "v_ego")

        features = extract_features(
            steering_torque=steering_torque,
            steering_angle_deg=steering_angle_deg,
            actual_lateral_accel=actual_lateral_accel,
            desired_lateral_accel=desired_lateral_accel,
            a_ego=a_ego,
            v_ego=v_ego,
            cfg=cfg,
        )

        result = classify_event(features, cfg)

        # Aggregate columns for existing visualizations
        torque_mean_abs = float(np.abs(steering_torque).mean()) if n_frames > 0 else np.nan
        valid_a = a_ego[~np.isnan(a_ego)]
        a_ego_max_abs = float(np.abs(valid_a).max()) if len(valid_a) > 0 else np.nan
        valid_v = v_ego[~np.isnan(v_ego)]
        speed_mean = float(valid_v.mean()) if len(valid_v) > 0 else np.nan

        # Lane-change guard — handles both string ("off") and integer (0) encodings
        lane_change_active = False
        if "lane_change_state" in active_df.columns:
            lcs = chunk["lane_change_state"].dropna()
            if len(lcs) > 0:
                if lcs.dtype == object:
                    lane_change_active = bool((lcs != "off").any())
                else:
                    lane_change_active = bool((lcs != 0).any())

        # Map cascade label to match mark_rows() / print_summary() expectations
        classification = "driver" if result.label == "driver" else "mechanical"

        # Lane-change guard: override to driver unconditionally
        if lane_change_active:
            classification = "driver"

        row = {
            # Metadata
            "event_id": evt_id,
            "start_idx": s,
            "end_idx": e,
            "start_time": s * 0.01,
            "end_time": e * 0.01,
            "duration_s": features["duration_s"],
            "n_frames": n_frames,
            # Aggregates for existing visualizations
            "torque_mean_abs": torque_mean_abs,
            "a_ego_max_abs": a_ego_max_abs,
            "speed_mean": speed_mean,
            # All cascade features
            "peak_torque_rate_nm_s": features["peak_torque_rate_nm_s"],
            "sign_consistency": features["sign_consistency"],
            "zero_crossing_rate_hz": features["zero_crossing_rate_hz"],
            "torque_kurtosis": features["torque_kurtosis"],
            "has_longitudinal_shock": features["has_longitudinal_shock"],
            "torque_leads_angle": features["torque_leads_angle"],
            "torque_lat_accel_corr": features["torque_lat_accel_corr"],
            "freq_energy_ratio": features["freq_energy_ratio"],
            "speed_adjusted_is_brief": features["speed_adjusted_is_brief"],
            "lat_accel_residual": features["lat_accel_residual"],
            "v_ego_mean": features["v_ego_mean"],
            "peak_steering_torque_abs": features["peak_steering_torque_abs"],
            # Classification
            "stage": result.stage,
            "mechanical_score": result.mechanical_score,
            "driver_score": result.driver_score,
            "confidence": result.confidence,
            "classification": classification,
            "lane_change_active": lane_change_active,
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_event_features(df, events):
    """Compute per-event features from the rows each event spans.

    Returns a DataFrame with one row per event.

    NOTE: Kept for backward compatibility. Use classify_events_cascade() in new code.
    """
    rows = []

    for evt_id, evt in enumerate(events):
        s, e = evt["start_idx"], evt["end_idx"]
        chunk = df.iloc[s : e + 1]
        n_frames = len(chunk)
        duration_s = n_frames * 0.01

        # Torque features
        torque_mean_abs = np.nan
        torque_std = np.nan
        torque_sign_consistency = np.nan
        if "steering_torque" in df.columns:
            torque = chunk["steering_torque"].dropna()
            if len(torque) > 0:
                torque_mean_abs = torque.abs().mean()
                torque_std = torque.std() if len(torque) > 1 else 0.0
                modal_sign = 1 if torque.mean() >= 0 else -1
                torque_sign_consistency = (np.sign(torque) == modal_sign).mean()

        # Steering rate
        steering_rate_max_abs = np.nan
        if "steering_rate_deg" in df.columns:
            sr = chunk["steering_rate_deg"].dropna()
            if len(sr) > 0:
                steering_rate_max_abs = sr.abs().max()

        # Longitudinal acceleration
        a_ego_max_abs = np.nan
        if "a_ego" in df.columns:
            a = chunk["a_ego"].dropna()
            if len(a) > 0:
                a_ego_max_abs = a.abs().max()

        # Lateral acceleration std
        lat_accel_std = np.nan
        for col in ["actual_lateral_accel", "desired_lateral_accel"]:
            if col in df.columns:
                la = chunk[col].dropna()
                if len(la) > 1:
                    lat_accel_std = la.std()
                break

        # Speed context
        speed_mean = np.nan
        if "v_ego" in df.columns:
            v = chunk["v_ego"].dropna()
            if len(v) > 0:
                speed_mean = v.mean()

        # Torque rate — peak |dT/dt| in Nm/s (100 Hz data → multiply diff by 100)
        torque_rate_max_abs = np.nan
        if "steering_torque" in df.columns:
            torque_vals = chunk["steering_torque"].dropna()
            if len(torque_vals) > 1:
                torque_rate_max_abs = np.abs(np.diff(torque_vals.values)).max() * 100.0

        # Torque-lateral correlation — Pearson r(steering_torque, actual_lateral_accel)
        torque_lat_corr = np.nan
        if "steering_torque" in df.columns and "actual_lateral_accel" in df.columns:
            paired = chunk[["steering_torque", "actual_lateral_accel"]].dropna()
            if len(paired) >= 3:
                torque_lat_corr = paired["steering_torque"].corr(paired["actual_lateral_accel"])

        # Lane-change guard — handles both string ("off") and integer (0) encodings
        lane_change_active = False
        if "lane_change_state" in df.columns:
            lcs = chunk["lane_change_state"].dropna()
            if len(lcs) > 0:
                if lcs.dtype == object:
                    lane_change_active = bool((lcs != "off").any())
                else:
                    lane_change_active = bool((lcs != 0).any())

        # Start/end time (use index as proxy if no time column)
        start_time = s * 0.01
        end_time = e * 0.01

        rows.append({
            "event_id": evt_id,
            "start_idx": s,
            "end_idx": e,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "n_frames": n_frames,
            "torque_mean_abs": torque_mean_abs,
            "torque_std": torque_std,
            "torque_sign_consistency": torque_sign_consistency,
            "steering_rate_max_abs": steering_rate_max_abs,
            "a_ego_max_abs": a_ego_max_abs,
            "lat_accel_std": lat_accel_std,
            "speed_mean": speed_mean,
            "torque_rate_max_abs": torque_rate_max_abs,
            "torque_lat_corr": torque_lat_corr,
            "lane_change_active": lane_change_active,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def classify_events(
    events_df,
    min_duration=DEFAULT_MIN_DURATION,
    a_ego_thresh=DEFAULT_A_EGO_THRESH,
    consistency_thresh=DEFAULT_CONSISTENCY_THRESH,
    min_score=DEFAULT_MIN_SCORE,
    torque_rate_thresh=DEFAULT_TORQUE_RATE_THRESH,
    highway_speed_thresh=DEFAULT_HIGHWAY_SPEED_THRESH,
    highway_min_duration=DEFAULT_HIGHWAY_MIN_DURATION,
):
    """Apply heuristic rules and classify each event.

    Adds columns: rule_brief, rule_shock, rule_chaotic, rule_sharp_onset, rule_highway_brief,
    mechanical_score, classification, confidence.

    Lane-change guard: if lane_change_active is True the event is classified genuine unconditionally.

    NOTE: Kept for backward compatibility. Use classify_events_cascade() in new code.
    """
    df = events_df.copy()

    # Rule 1: Brief duration
    df["rule_brief"] = (df["duration_s"] < min_duration).astype(int)

    # Rule 2: Longitudinal shock + brief
    df["rule_shock"] = (
        (df["a_ego_max_abs"] > a_ego_thresh) & (df["duration_s"] < 0.4)
    ).fillna(False).astype(int)

    # Rule 3: Chaotic torque (low sign consistency)
    df["rule_chaotic"] = (
        df["torque_sign_consistency"] < consistency_thresh
    ).fillna(False).astype(int)

    # Rule 4: Sharp onset — peak |dT/dt| exceeds threshold (mechanical impulse signature)
    df["rule_sharp_onset"] = (
        df["torque_rate_max_abs"] > torque_rate_thresh
    ).fillna(False).astype(int)

    # Rule 5: Highway brief — short event at highway speed is almost certainly a disturbance
    df["rule_highway_brief"] = (
        (df["speed_mean"] > highway_speed_thresh) & (df["duration_s"] < highway_min_duration)
    ).fillna(False).astype(int)

    df["mechanical_score"] = (
        df["rule_brief"] + df["rule_shock"] + df["rule_chaotic"]
        + df["rule_sharp_onset"] + df["rule_highway_brief"]
    )
    df["classification"] = df["mechanical_score"].apply(
        lambda s: "mechanical" if s >= min_score else "genuine"
    )

    # Lane-change guard: override classification to genuine unconditionally
    if "lane_change_active" in df.columns:
        df.loc[df["lane_change_active"].astype(bool), "classification"] = "genuine"

    # Confidence: fraction of rules fired (for mechanical), inverted for genuine
    df["confidence"] = df.apply(
        lambda r: r["mechanical_score"] / 5 if r["classification"] == "mechanical"
        else 1 - r["mechanical_score"] / 5,
        axis=1,
    )

    return df


def mark_rows(active_df, events_df):
    """Add '_intervention' column to active_df: 'normal', 'genuine', or 'mechanical'.

    Rows that belong to an override event are labelled with the event's classification;
    all other rows are labelled 'normal'.
    """
    labels = pd.array(["normal"] * len(active_df), dtype=object)
    for _, evt in events_df.iterrows():
        labels[int(evt["start_idx"]) : int(evt["end_idx"]) + 1] = evt["classification"]
    result = active_df.copy()
    result["_intervention"] = labels
    return result


def prune_csv_stream(input_path, output_path, prune="both", gap_frames=DEFAULT_GAP_FRAMES,
                     cfg=None, chunksize=DEFAULT_CHUNK_ROWS, max_event_rows=100_000):
    """Classify and prune override events while streaming a CSV.

    The default ``prune=both`` path only retains the short gap look-ahead.
    Classification modes retain one current event and reject an abnormally
    long event instead of risking an unbounded memory allocation.
    """
    if cfg is None and prune != "both":
        from nnlc_tools.steering_classifier.config import ClassifierConfig
        cfg = ClassifierConfig()
    require_distinct_paths(input_path, output_path)
    output_handle = None
    columns = None
    active_rows = dropped_rows = event_count = 0
    event_core = []
    event_row_count = 0
    pending_gap = []
    output_buffer = []
    output_buffer_limit = min(chunksize, 10_000)
    route_sentinel = object()
    current_route = route_sentinel

    def flush_output():
        if output_buffer:
            pd.DataFrame.from_records(output_buffer, columns=columns).to_csv(
                output_handle, header=False, index=False, lineterminator="\n"
            )
            output_buffer.clear()

    def write_rows(rows):
        nonlocal active_rows
        if rows:
            output_buffer.extend(rows)
            active_rows += len(rows)
            if len(output_buffer) >= output_buffer_limit:
                flush_output()

    def finish_event():
        nonlocal dropped_rows, event_count, event_row_count
        if event_row_count == 0:
            return
        if prune != "both" and event_row_count > max_event_rows:
            raise ValueError(
                f"override event exceeds {max_event_rows:,} rows; "
                "refusing an unbounded in-memory classification"
            )
        event_count += 1
        if prune == "both":
            remove = True
        else:
            event_df = pd.DataFrame(event_core, columns=columns)
            events = [{"start_idx": 0, "end_idx": len(event_df) - 1}]
            classified = classify_events_cascade(event_df, events, cfg=cfg)
            classification = str(classified.iloc[0]["classification"])
            remove = classification == prune
        if remove:
            dropped_rows += event_row_count
        else:
            write_rows(event_core)
        event_core.clear()
        event_row_count = 0

    try:
        for chunk in iter_csv_chunks(input_path, chunksize=chunksize):
            if columns is None:
                columns = list(chunk.columns)
                output_handle, _ = open_csv_writer(output_path, columns)
            required = {"steering_pressed"}
            missing = sorted(required.difference(chunk.columns))
            if missing:
                raise ValueError(f"missing fields: {', '.join(missing)}")
            route_index = columns.index("route_id") if "route_id" in columns else None
            active_index = columns.index("active") if "active" in columns else None
            standstill_index = columns.index("standstill") if "standstill" in columns else None
            pressed_index = columns.index("steering_pressed")
            for row in chunk.itertuples(index=False, name=None):
                route = row[route_index] if route_index is not None else None
                if current_route is not route_sentinel and str(route) != str(current_route):
                    finish_event()
                    write_rows(pending_gap)
                    pending_gap.clear()
                current_route = route
                is_active = active_index is None or bool(row[active_index])
                is_standstill = standstill_index is not None and bool(row[standstill_index])
                if not is_active or is_standstill:
                    continue
                pressed = bool(row[pressed_index])
                if pressed:
                    if event_row_count:
                        event_row_count += len(pending_gap)
                        if prune != "both":
                            event_core.extend(pending_gap)
                        pending_gap.clear()
                    event_row_count += 1
                    if prune != "both":
                        event_core.append(row)
                    if prune != "both" and event_row_count > max_event_rows:
                        raise ValueError(
                            f"override event exceeds {max_event_rows:,} rows; "
                            "refusing an unbounded in-memory classification"
                        )
                elif event_row_count:
                    pending_gap.append(row)
                    if len(pending_gap) > gap_frames:
                        finish_event()
                        write_rows(pending_gap)
                        pending_gap.clear()
                else:
                    write_rows([row])
        finish_event()
        write_rows(pending_gap)
        flush_output()
    finally:
        if output_handle is not None:
            output_handle.close()
    if columns is None:
        raise ValueError("input CSV is empty or missing")
    return {
        "active_rows": active_rows,
        "dropped_rows": dropped_rows,
        "event_count": event_count,
        "output_rows": active_rows,
    }


def plot_intervention_scatter(df, output_path, max_points=None):
    """Generate lat_accel vs torque scatter by speed bin with intervention points highlighted.

    Three layers per panel:
      - gray (faint)   — normal driving frames
      - steelblue      — frames belonging to genuine driver interventions
      - tomato (large) — frames belonging to mechanical disturbances
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    MS_TO_MPH = 2.23694

    def save_empty_plot(message):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved intervention scatter to {output_path}")
        plt.close()

    lat_accel_col = None
    for col in ["actual_lateral_accel", "desired_lateral_accel"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lat_accel_col = col
            break

    if lat_accel_col is None:
        print("WARNING: No lateral acceleration data found for scatter plot.")
        save_empty_plot("No lateral acceleration data")
        return

    if "torque_output" not in df.columns:
        print("WARNING: No torque_output column. Skipping intervention scatter.")
        save_empty_plot("No torque output data")
        return

    valid = df[[lat_accel_col, "torque_output", "v_ego", "_intervention"]].dropna(
        subset=[lat_accel_col, "torque_output", "v_ego"]
    ).copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH

    speed_bins = list(range(0, 90, 10))
    n_bins = len(speed_bins)
    ncols = 3
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Lateral Accel vs Torque — Intervention Overlay", fontsize=14, fontweight="bold")

    for i, speed_lo in enumerate(speed_bins):
        speed_hi = speed_lo + 10
        ax = axes[i]

        bin_data = valid[(valid["speed_mph"] >= speed_lo) & (valid["speed_mph"] < speed_hi)]

        normal = bin_data[bin_data["_intervention"] == "normal"]
        genuine = bin_data[bin_data["_intervention"] == "driver"]
        mechanical = bin_data[bin_data["_intervention"] == "mechanical"]

        # Downsample normal background if requested
        plot_normal = (
            normal.sample(n=max_points, random_state=42)
            if max_points and len(normal) > max_points
            else normal
        )

        if len(plot_normal) > 0:
            ax.scatter(plot_normal[lat_accel_col], plot_normal["torque_output"],
                       c="gray", s=0.5, alpha=0.15, rasterized=True)

        if len(genuine) > 0:
            ax.scatter(genuine[lat_accel_col], genuine["torque_output"],
                       c="steelblue", s=4, alpha=0.6, rasterized=True)

        if len(mechanical) > 0:
            ax.scatter(mechanical[lat_accel_col], mechanical["torque_output"],
                       c="tomato", s=10, alpha=0.9, rasterized=True, zorder=5)

        ax.set_title(
            f"{speed_lo}-{speed_hi} mph  (n={len(bin_data):,}"
            + (f", gen={len(genuine):,}" if len(genuine) > 0 else "")
            + (f", mech={len(mechanical):,}" if len(mechanical) > 0 else "")
            + ")",
            fontsize=9,
        )
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Lat Accel (m/s²)")
        ax.set_ylabel("Torque")
        ax.grid(axis="x", color="0.95")
        ax.grid(axis="y", color="0.95")

    # Legend on first panel
    handles = [
        Patch(facecolor="gray", alpha=0.5, label="Normal driving"),
        Patch(facecolor="steelblue", alpha=0.8, label="Driver intervention"),
        Patch(facecolor="tomato", alpha=0.9, label="Mechanical disturbance"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)

    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved intervention scatter to {output_path}")
    plt.close()


def print_summary(events_df, total_frames=None):
    """Print console summary of classification results."""
    total = len(events_df)
    if total == 0:
        print("No override events found.")
        return

    driver = events_df[events_df["classification"] == "driver"]
    mechanical = events_df[events_df["classification"] == "mechanical"]
    n_driver = len(driver)
    n_mechanical = len(mechanical)

    print(f"\nTotal override events: {total:,}")
    print(f"  Driver interventions:    {n_driver:,}  ({n_driver / total:.1%})")
    print(f"  Mechanical disturbances: {n_mechanical:,} ({n_mechanical / total:.1%})")

    if total_frames is not None and total_frames > 0:
        g_frames = int(driver["n_frames"].sum()) if n_driver > 0 else 0
        m_frames = int(mechanical["n_frames"].sum()) if n_mechanical > 0 else 0
        print(f"\nAs % of active driving frames ({total_frames:,} total):")
        print(f"  Driver interventions:    {g_frames:,} frames ({g_frames / total_frames:.2%})")
        print(f"  Mechanical disturbances: {m_frames:,} frames ({m_frames / total_frames:.2%})")

    if n_driver > 0:
        g_dur = driver["duration_s"].median()
        g_torque = driver["torque_mean_abs"].dropna().median()
        print(f"\nDriver:    median duration {g_dur:.2f}s", end="")
        if not np.isnan(g_torque):
            print(f", median |torque| {g_torque:.1f}", end="")
        print()

    if n_mechanical > 0:
        m_dur = mechanical["duration_s"].median()
        m_torque = mechanical["torque_mean_abs"].dropna().median()
        print(f"Mechanical: median duration {m_dur:.2f}s", end="")
        if not np.isnan(m_torque):
            print(f", median |torque| {m_torque:.1f}", end="")
        print()

    print()


def plot_interventions(events_df, output_path, cfg=None):
    """Generate 2×3 subplot showing cascade feature diagnostics.

    Panels:
      [0,0] — Histogram: peak_torque_rate_nm_s (log x), driver vs mechanical, dashed at 80 Nm/s
      [0,1] — Histogram: sign_consistency, dashed lines at 0.60 and 0.90
      [0,2] — Scatter: zero_crossing_rate_hz vs torque_kurtosis, colored by classification
      [1,0] — Histogram: torque_lat_accel_corr (NaN excluded), dashed at 0.1 and 0.6
      [1,1] — Histogram: freq_energy_ratio (NaN excluded, log x), dashed at 1.0 and 3.0
      [1,2] — Grouped bar: cascade stage distribution + genuine vs mechanical totals
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if cfg is None:
        from nnlc_tools.steering_classifier.config import ClassifierConfig
        cfg = ClassifierConfig()

    if events_df.empty:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Driver Intervention Classification", fontsize=13, fontweight="bold")
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No override events", ha="center", va="center")
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved interventions plot to {output_path}")
        plt.close()
        return

    driver = events_df[events_df["classification"] == "driver"]
    mechanical = events_df[events_df["classification"] == "mechanical"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Driver Intervention Classification — Cascade Feature Analysis", fontsize=13, fontweight="bold")

    # ── [0,0] peak_torque_rate_nm_s histogram (log x-axis) ──────────────
    ax = axes[0, 0]
    col = "peak_torque_rate_nm_s"
    d_vals = driver[col].dropna().values
    m_vals = mechanical[col].dropna().values
    all_vals = np.concatenate([d_vals, m_vals])
    if len(all_vals) > 0:
        pos_vals = all_vals[all_vals > 0]
        if len(pos_vals) > 0:
            log_bins = np.logspace(np.log10(max(pos_vals.min(), 0.1)), np.log10(pos_vals.max() * 1.1), 40)
            if len(d_vals[d_vals > 0]) > 0:
                ax.hist(d_vals[d_vals > 0], bins=log_bins, color="steelblue", alpha=0.7, label="driver")
            if len(m_vals[m_vals > 0]) > 0:
                ax.hist(m_vals[m_vals > 0], bins=log_bins, color="tomato", alpha=0.7, label="mechanical")
            ax.axvline(cfg.torque_rate_definite_mechanical, color="black", linestyle="--", linewidth=1.5,
                       label=f"{cfg.torque_rate_definite_mechanical} Nm/s")
            ax.set_xscale("log")
    ax.set_xlabel("Peak torque rate (Nm/s)")
    ax.set_ylabel("Event count")
    ax.set_title("F1: Peak Torque Rate")
    ax.legend(fontsize=8)

    # ── [0,1] sign_consistency histogram ────────────────────────────────
    ax = axes[0, 1]
    col = "sign_consistency"
    d_vals = driver[col].dropna().values
    m_vals = mechanical[col].dropna().values
    bins = np.linspace(0, 1, 41)
    if len(d_vals) > 0:
        ax.hist(d_vals, bins=bins, color="steelblue", alpha=0.7, label="driver")
    if len(m_vals) > 0:
        ax.hist(m_vals, bins=bins, color="tomato", alpha=0.7, label="mechanical")
    ax.axvline(cfg.sign_consistency_mechanical, color="black", linestyle="--", linewidth=1.5,
               label=f"{cfg.sign_consistency_mechanical}")
    ax.axvline(cfg.sign_consistency_driver, color="gray", linestyle="--", linewidth=1.0,
               label=f"{cfg.sign_consistency_driver}")
    ax.set_xlabel("Sign consistency")
    ax.set_ylabel("Event count")
    ax.set_title("F3: Sign Consistency")
    ax.legend(fontsize=8)

    # ── [0,2] zero_crossing_rate_hz vs torque_kurtosis scatter ──────────
    ax = axes[0, 2]
    colors_map = {"driver": "steelblue", "mechanical": "tomato"}
    for label, subset in [("driver", driver), ("mechanical", mechanical)]:
        mask = subset["zero_crossing_rate_hz"].notna() & subset["torque_kurtosis"].notna()
        sub = subset[mask]
        if len(sub) > 0:
            ax.scatter(sub["zero_crossing_rate_hz"], sub["torque_kurtosis"],
                       c=colors_map[label], s=15, alpha=0.6, label=label)
    ax.set_xlabel("Zero-crossing rate (Hz)")
    ax.set_ylabel("Torque kurtosis")
    ax.set_title("F4 vs F5: ZCR vs Kurtosis")
    ax.legend(fontsize=8)

    # ── [1,0] torque_lat_accel_corr histogram ───────────────────────────
    ax = axes[1, 0]
    col = "torque_lat_accel_corr"
    d_vals = driver[col].dropna().values
    m_vals = mechanical[col].dropna().values
    if len(d_vals) > 0 or len(m_vals) > 0:
        bins = np.linspace(-1, 1, 41)
        if len(d_vals) > 0:
            ax.hist(d_vals, bins=bins, color="steelblue", alpha=0.7, label="driver")
        if len(m_vals) > 0:
            ax.hist(m_vals, bins=bins, color="tomato", alpha=0.7, label="mechanical")
        ax.axvline(cfg.corr_mechanical, color="black", linestyle="--", linewidth=1.5,
                   label=f"{cfg.corr_mechanical}")
        ax.axvline(cfg.corr_strong_driver, color="gray", linestyle="--", linewidth=1.0,
                   label=f"{cfg.corr_strong_driver}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No correlation data available",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Torque–lat accel correlation")
    ax.set_ylabel("Event count")
    ax.set_title("F8: Torque–Lat Accel Correlation")

    # ── [1,1] freq_energy_ratio histogram (log x-axis) ──────────────────
    ax = axes[1, 1]
    col = "freq_energy_ratio"
    d_vals = driver[col].dropna().values
    m_vals = mechanical[col].dropna().values
    all_vals = np.concatenate([d_vals, m_vals])
    if len(all_vals) > 0:
        pos_vals = all_vals[all_vals > 0]
        if len(pos_vals) > 0:
            log_bins = np.logspace(np.log10(max(pos_vals.min(), 0.01)), np.log10(pos_vals.max() * 1.1), 40)
            if len(d_vals[d_vals > 0]) > 0:
                ax.hist(d_vals[d_vals > 0], bins=log_bins, color="steelblue", alpha=0.7, label="driver")
            if len(m_vals[m_vals > 0]) > 0:
                ax.hist(m_vals[m_vals > 0], bins=log_bins, color="tomato", alpha=0.7, label="mechanical")
            ax.axvline(cfg.freq_ratio_mechanical, color="black", linestyle="--", linewidth=1.5,
                       label=f"{cfg.freq_ratio_mechanical}")
            ax.axvline(cfg.freq_ratio_driver, color="gray", linestyle="--", linewidth=1.0,
                       label=f"{cfg.freq_ratio_driver}")
            ax.set_xscale("log")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No freq ratio data available",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Freq energy ratio (low/high)")
    ax.set_ylabel("Event count")
    ax.set_title("F9: Frequency Energy Ratio")

    # ── [1,2] Stage distribution + classification summary bar ───────────
    ax = axes[1, 2]
    if "stage" in events_df.columns:
        stages = [1, 2, 3]
        stage_counts = [int((events_df["stage"] == s).sum()) for s in stages]
        n_driver = int((events_df["classification"] == "driver").sum())
        n_mechanical = int((events_df["classification"] == "mechanical").sum())

        x = np.arange(len(stages) + 1)
        bar_colors = ["#4e79a7", "#f28e2b", "#e15759", "steelblue"]
        all_counts = stage_counts + [n_driver]
        all_labels = ["Stage 1", "Stage 2", "Stage 3", "Driver"]
        mech_bar = [0, 0, 0, n_mechanical]

        ax.bar(x, all_counts, color=bar_colors, alpha=0.8)
        ax.bar(x[3:], mech_bar[3:], color="tomato", alpha=0.8)
        # Overlay mechanical on top of driver for last bar pair
        ax.bar([3], [n_mechanical], bottom=[n_driver], color="tomato", alpha=0.8, label="mechanical")
        ax.bar([3], [n_driver], color="steelblue", alpha=0.8, label="driver")
        ax.bar([0, 1, 2], stage_counts, color=["#4e79a7", "#f28e2b", "#e15759"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Stage 1", "Stage 2", "Stage 3", "Total"])
        ax.set_ylabel("Event count")
        ax.set_title("Cascade Stage Distribution")
        handles = [
            Patch(facecolor="steelblue", alpha=0.8, label="driver"),
            Patch(facecolor="tomato", alpha=0.8, label="mechanical"),
        ]
        ax.legend(handles=handles, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No stage data available",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title("Cascade Stage Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved interventions plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Classify steering override events as genuine driver interventions or mechanical disturbances.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("-o", "--output", default="interventions.png",
                        help="Output plot PNG (default: interventions.png)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate cascade feature visualization PNG")
    parser.add_argument("--gap-frames", type=int, default=DEFAULT_GAP_FRAMES,
                        help=f"Frames of gap allowed within one event (default: {DEFAULT_GAP_FRAMES})")
    parser.add_argument("--scatter", action="store_true",
                        help="Generate lat_accel vs torque scatter plot with intervention points highlighted")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max normal-driving points per scatter subplot (random sample, default: all)")
    parser.add_argument("--torque-rate-mechanical", type=float, default=80.0,
                        help="Torque rate definite mechanical threshold in Nm/s (default: 80.0)")
    parser.add_argument("--torque-rate-driver", type=float, default=20.0,
                        help="Torque rate definite driver threshold in Nm/s (default: 20.0)")
    parser.add_argument("--max-pothole-length", type=float, default=2.5,
                        help="Max pothole length in meters for speed-adaptive brevity (default: 2.5)")
    parser.add_argument("--prune-output", default=None, metavar="PATH",
                        help="Write active frames with the selected event type(s) removed to PATH (.csv or .parquet)")
    parser.add_argument("--prune", choices=["mechanical", "driver", "both"], default="both",
                        help="Which event types to remove when --prune-output is set (default: both)")
    parser.add_argument("--streaming", action="store_true",
                        help="Process CSV in bounded chunks; only supports --prune-output")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"Rows per streaming chunk (default: {DEFAULT_CHUNK_ROWS:,})")
    args = parser.parse_args()

    from nnlc_tools.steering_classifier.config import ClassifierConfig
    cfg = ClassifierConfig(
        torque_rate_definite_mechanical=args.torque_rate_mechanical,
        torque_rate_definite_driver=args.torque_rate_driver,
        max_pothole_length_m=args.max_pothole_length,
    )

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be a positive integer")
    if args.streaming:
        if not args.input.lower().endswith(".csv"):
            parser.error("--streaming only supports CSV input")
        if not args.prune_output or args.plot or args.scatter:
            parser.error("--streaming requires --prune-output and does not support --plot/--scatter")
        try:
            stats = prune_csv_stream(
                args.input, args.prune_output, prune=args.prune,
                gap_frames=args.gap_frames, cfg=cfg, chunksize=args.chunk_rows,
            )
        except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"Active frames: {stats['active_rows'] + stats['dropped_rows']:,}")
        print(f"Segmented into {stats['event_count']:,} events")
        print(f"Pruned output: {stats['output_rows']:,} rows written to {args.prune_output} ({stats['dropped_rows']:,} frames removed)")
        return

    df = load_data(args.input)
    print(f"Loaded {len(df):,} rows")

    if "steering_pressed" not in df.columns:
        print("ERROR: No steering_pressed column found. Cannot detect override events.")
        sys.exit(1)

    # Filter to active driving (same pattern as visualize_coverage.py:52-57)
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    active_df = df[mask].reset_index(drop=True)

    n_override_frames = active_df["steering_pressed"].astype(bool).sum()
    print(f"Active frames: {len(active_df):,}  |  Override frames: {n_override_frames:,} ({n_override_frames / max(len(active_df), 1):.1%})")

    events = segment_events(active_df, gap_frames=args.gap_frames)
    print(f"Segmented into {len(events):,} events (gap tolerance: {args.gap_frames} frames)")

    # No override events is a valid dataset state. Keep the active driving
    # rows unchanged so --prune-output still produces the training input.
    if events:
        events_df = classify_events_cascade(active_df, events, cfg=cfg)
        print_summary(events_df, total_frames=len(active_df))
    else:
        events_df = pd.DataFrame()
        print("No override events found; keeping all active driving frames.")

    if args.plot:
        plot_interventions(events_df, args.output, cfg=cfg)

    if args.scatter:
        import os
        base, ext = os.path.splitext(args.output)
        scatter_path = f"{base}_scatter{ext or '.png'}"
        labelled_df = mark_rows(active_df, events_df)
        plot_intervention_scatter(labelled_df, scatter_path, max_points=args.max_points)

    if args.prune_output:
        import os
        labelled_df = mark_rows(active_df, events_df)
        if args.prune == "both":
            pruned_df = labelled_df[labelled_df["_intervention"] == "normal"].drop(columns=["_intervention"])
        else:
            pruned_df = labelled_df[labelled_df["_intervention"] != args.prune].drop(columns=["_intervention"])
        n_dropped = len(labelled_df) - len(pruned_df)
        _, ext = os.path.splitext(args.prune_output)
        if ext.lower() == ".parquet":
            pruned_df.to_parquet(args.prune_output, index=False)
        else:
            pruned_df.to_csv(args.prune_output, index=False)
        print(f"Pruned output: {len(pruned_df):,} rows written to {args.prune_output}  ({n_dropped:,} {args.prune} frames removed, {n_dropped / max(len(labelled_df), 1):.2%} of active data)")


if __name__ == "__main__":
    main()
