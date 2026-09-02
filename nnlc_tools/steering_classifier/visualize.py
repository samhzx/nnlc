#!/usr/bin/env python3
"""Standalone visualization for steering classifier feature exploration.

Loads a lateral data CSV/Parquet, segments events, runs the cascade classifier,
and generates a 3×3 diagnostic plot.

Usage:
  uv run nnlc-sc-visualize output/lateral_data.csv -o output/sc_features.png
"""

import argparse
import sys

import numpy as np
import pandas as pd


def _col_array(chunk, df, col, fallback_val=0.0):
    """Extract column from chunk as numpy array, filling missing with fallback_val."""
    if col in df.columns:
        arr = chunk[col].values.astype(float)
        arr = np.where(np.isnan(arr), fallback_val, arr)
        return arr
    return np.full(len(chunk), fallback_val)


def _col_array_nan(chunk, df, col):
    """Extract column from chunk as numpy array, leaving NaN for missing values."""
    if col in df.columns:
        return chunk[col].values.astype(float)
    return np.full(len(chunk), np.nan)


def build_features_df(active_df, events, cfg):
    """Run cascade classifier on all events and return a features DataFrame."""
    from nnlc_tools.steering_classifier.features import extract_features
    from nnlc_tools.steering_classifier.cascade import classify_event

    rows = []
    for evt_id, evt in enumerate(events):
        s, e = evt["start_idx"], evt["end_idx"]
        chunk = active_df.iloc[s : e + 1]

        steering_torque = _col_array(chunk, active_df, "steering_torque", 0.0)
        steering_angle_deg = _col_array(chunk, active_df, "steering_angle_deg", 0.0)
        actual_lateral_accel = _col_array_nan(chunk, active_df, "actual_lateral_accel")
        desired_lateral_accel = _col_array_nan(chunk, active_df, "desired_lateral_accel")
        a_ego = _col_array_nan(chunk, active_df, "a_ego")
        v_ego = _col_array_nan(chunk, active_df, "v_ego")

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
        label = "driver" if result.label == "driver" else "mechanical"

        row = {
            "event_id": evt_id,
            "start_idx": s,
            "end_idx": e,
            "label": label,
            "stage": result.stage,
            "confidence": result.confidence,
            "mechanical_score": result.mechanical_score,
            "driver_score": result.driver_score,
        }
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_features(feat_df, output_path, cfg):
    """Generate 3×3 diagnostic plot of cascade features."""
    import matplotlib.pyplot as plt

    if feat_df.empty:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.suptitle("Steering Classifier — Feature Diagnostics", fontsize=13, fontweight="bold")
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No override events", ha="center", va="center")
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature plot to {output_path}")
        plt.close()
        return

    driver = feat_df[feat_df["label"] == "driver"]
    mechanical = feat_df[feat_df["label"] == "mechanical"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Steering Classifier — Feature Diagnostics", fontsize=13, fontweight="bold")

    MS_TO_MPH = 2.23694

    def _hist_overlay(ax, col, d_df, m_df, bins, xlabel, title, xscale=None):
        d_vals = d_df[col].dropna().values
        m_vals = m_df[col].dropna().values
        if xscale == "log":
            all_pos = np.concatenate([d_vals[d_vals > 0], m_vals[m_vals > 0]])
            if len(all_pos) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_title(title)
                return
            log_bins = np.logspace(np.log10(max(all_pos.min(), 0.01)),
                                   np.log10(all_pos.max() * 1.1), bins)
            if len(d_vals[d_vals > 0]) > 0:
                ax.hist(d_vals[d_vals > 0], bins=log_bins, color="steelblue", alpha=0.7, label="driver")
            if len(m_vals[m_vals > 0]) > 0:
                ax.hist(m_vals[m_vals > 0], bins=log_bins, color="tomato", alpha=0.7, label="mechanical")
            ax.set_xscale("log")
        else:
            if len(d_vals) > 0:
                ax.hist(d_vals, bins=bins, color="steelblue", alpha=0.7, label="driver")
            if len(m_vals) > 0:
                ax.hist(m_vals, bins=bins, color="tomato", alpha=0.7, label="mechanical")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Event count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    # ── [0,0] peak_torque_rate_nm_s (log x) ─────────────────────────────
    ax = axes[0, 0]
    _hist_overlay(ax, "peak_torque_rate_nm_s", driver, mechanical, 40,
                  "Peak torque rate (Nm/s)", "F1: Peak Torque Rate", xscale="log")
    ax.axvline(cfg.torque_rate_definite_mechanical, color="black", linestyle="--",
               linewidth=1.5, label=f"{cfg.torque_rate_definite_mechanical} Nm/s")
    ax.legend(fontsize=8)

    # ── [0,1] sign_consistency ───────────────────────────────────────────
    ax = axes[0, 1]
    _hist_overlay(ax, "sign_consistency", driver, mechanical,
                  np.linspace(0, 1, 41), "Sign consistency", "F3: Sign Consistency")
    ax.axvline(cfg.sign_consistency_mechanical, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(cfg.sign_consistency_driver, color="gray", linestyle="--", linewidth=1.0)

    # ── [0,2] zero_crossing_rate_hz ─────────────────────────────────────
    ax = axes[0, 2]
    _hist_overlay(ax, "zero_crossing_rate_hz", driver, mechanical, 40,
                  "Zero-crossing rate (Hz)", "F4: Zero-Crossing Rate")

    # ── [1,0] torque_kurtosis ────────────────────────────────────────────
    ax = axes[1, 0]
    d_vals = driver["torque_kurtosis"].dropna().values
    m_vals = mechanical["torque_kurtosis"].dropna().values
    clip = 20.0
    if len(d_vals) > 0:
        ax.hist(d_vals.clip(max=clip), bins=40, color="steelblue", alpha=0.7, label="driver")
    if len(m_vals) > 0:
        ax.hist(m_vals.clip(max=clip), bins=40, color="tomato", alpha=0.7, label="mechanical")
    ax.axvline(cfg.kurtosis_impulsive, color="black", linestyle="--", linewidth=1.5,
               label=f"impulsive={cfg.kurtosis_impulsive}")
    ax.set_xlabel(f"Torque kurtosis (clipped at {clip})")
    ax.set_ylabel("Event count")
    ax.set_title("F5: Torque Kurtosis")
    ax.legend(fontsize=8)

    # ── [1,1] torque_lat_accel_corr vs freq_energy_ratio scatter ────────
    ax = axes[1, 1]
    colors_map = {"driver": "steelblue", "mechanical": "tomato"}
    for label, subset in [("driver", driver), ("mechanical", mechanical)]:
        mask = subset["torque_lat_accel_corr"].notna() & subset["freq_energy_ratio"].notna()
        sub = subset[mask]
        if len(sub) > 0:
            sizes = sub["confidence"].fillna(0.5).values * 20
            ax.scatter(sub["torque_lat_accel_corr"], sub["freq_energy_ratio"],
                       c=colors_map[label], s=sizes, alpha=0.6, label=label)
    ax.axvline(cfg.corr_mechanical, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(cfg.corr_strong_driver, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(cfg.freq_ratio_mechanical, color="black", linestyle=":", linewidth=1.0)
    ax.axhline(cfg.freq_ratio_driver, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Torque–lat accel corr")
    ax.set_ylabel("Freq energy ratio")
    ax.set_title("F8 vs F9: Corr vs Freq Ratio")
    ax.legend(fontsize=8)

    # ── [1,2] lat_accel_residual ─────────────────────────────────────────
    ax = axes[1, 2]
    d_vals = driver["lat_accel_residual"].dropna().values
    m_vals = mechanical["lat_accel_residual"].dropna().values
    if len(d_vals) > 0 or len(m_vals) > 0:
        all_vals = np.concatenate([d_vals, m_vals])
        clip = min(all_vals.max(), 5.0)
        bins = np.linspace(0, clip, 40)
        if len(d_vals) > 0:
            ax.hist(d_vals.clip(max=clip), bins=bins, color="steelblue", alpha=0.7, label="driver")
        if len(m_vals) > 0:
            ax.hist(m_vals.clip(max=clip), bins=bins, color="tomato", alpha=0.7, label="mechanical")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No residual data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Lat accel residual (m/s²)")
    ax.set_ylabel("Event count")
    ax.set_title("F11: Lat Accel Residual")

    # ── [2,0] v_ego_mean vs duration_s scatter ───────────────────────────
    ax = axes[2, 0]
    for label, subset in [("driver", driver), ("mechanical", mechanical)]:
        mask = subset["v_ego_mean"].notna() & subset["duration_s"].notna()
        sub = subset[mask]
        if len(sub) > 0:
            ax.scatter(sub["v_ego_mean"] * MS_TO_MPH, sub["duration_s"],
                       c=colors_map[label], s=10, alpha=0.6, label=label)
    ax.set_xlabel("Mean speed (mph)")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Speed vs Duration")
    ax.legend(fontsize=8)

    # ── [2,1] stacked bar: driver vs mechanical per speed band ───────────
    ax = axes[2, 1]
    bands = [(0, 10), (10, 20), (20, 30), (30, 999)]
    band_labels = ["0-10", "10-20", "20-30", "30+"]
    d_counts = []
    m_counts = []
    for lo, hi in bands:
        if "v_ego_mean" in feat_df.columns:
            mask = (feat_df["v_ego_mean"] >= lo) & (feat_df["v_ego_mean"] < hi)
            band = feat_df[mask]
        else:
            band = pd.DataFrame()
        d_counts.append(int((band["label"] == "driver").sum()) if len(band) > 0 else 0)
        m_counts.append(int((band["label"] == "mechanical").sum()) if len(band) > 0 else 0)
    x = np.arange(len(band_labels))
    ax.bar(x, d_counts, color="steelblue", alpha=0.8, label="driver")
    ax.bar(x, m_counts, bottom=d_counts, color="tomato", alpha=0.8, label="mechanical")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nm/s" for l in band_labels])
    ax.set_ylabel("Event count")
    ax.set_title("Classification by Speed Band")
    ax.legend(fontsize=8)

    # ── [2,2] cascade stage distribution bar ────────────────────────────
    ax = axes[2, 2]
    if "stage" in feat_df.columns:
        stages = [1, 2, 3]
        stage_labels = ["Stage 1", "Stage 2", "Stage 3"]
        stage_counts = [int((feat_df["stage"] == s).sum()) for s in stages]
        colors = ["#4e79a7", "#f28e2b", "#e15759"]
        ax.bar(stage_labels, stage_counts, color=colors, alpha=0.8)
        for i, (lbl, cnt) in enumerate(zip(stage_labels, stage_counts)):
            if cnt > 0:
                ax.text(i, cnt + 0.5, str(cnt), ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No stage data", transform=ax.transAxes, ha="center", va="center")
    ax.set_ylabel("Event count")
    ax.set_title("Cascade Stage Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved feature plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone steering classifier feature visualization.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("-o", "--output", default="sc_features.png",
                        help="Output plot PNG (default: sc_features.png)")
    parser.add_argument("--gap-frames", type=int, default=5,
                        help="Frames of gap allowed within one event (default: 5)")
    parser.add_argument("--torque-rate-mechanical", type=float, default=80.0,
                        help="Torque rate definite mechanical threshold in Nm/s (default: 80.0)")
    parser.add_argument("--torque-rate-driver", type=float, default=20.0,
                        help="Torque rate definite driver threshold in Nm/s (default: 20.0)")
    parser.add_argument("--max-pothole-length", type=float, default=2.5,
                        help="Max pothole length in meters for speed-adaptive brevity (default: 2.5)")
    args = parser.parse_args()

    from nnlc_tools.data_io import load_data
    from nnlc_tools.analyze_interventions import segment_events
    from nnlc_tools.steering_classifier.config import ClassifierConfig
    from nnlc_tools.steering_classifier.evaluate import (
        classification_report, per_speed_band_accuracy, latency_histogram, print_report,
    )

    cfg = ClassifierConfig(
        torque_rate_definite_mechanical=args.torque_rate_mechanical,
        torque_rate_definite_driver=args.torque_rate_driver,
        max_pothole_length_m=args.max_pothole_length,
    )

    df = load_data(args.input)
    if df is None:
        print(f"ERROR: No data found at {args.input}")
        sys.exit(1)
    print(f"Loaded {len(df):,} rows")

    if "steering_pressed" not in df.columns:
        print("ERROR: No steering_pressed column found. Cannot detect override events.")
        sys.exit(1)

    # Filter to active, non-standstill frames
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    active_df = df[mask].reset_index(drop=True)

    n_override = active_df["steering_pressed"].astype(bool).sum()
    print(f"Active frames: {len(active_df):,}  |  Override frames: {n_override:,} ({n_override / max(len(active_df), 1):.1%})")

    events = segment_events(active_df, gap_frames=args.gap_frames)
    print(f"Segmented into {len(events):,} events")

    if not events:
        print("No override events found.")
        plot_features(pd.DataFrame(), args.output, cfg)
        return

    feat_df = build_features_df(active_df, events, cfg)

    # Print report using evaluate.py
    labels = feat_df["label"].tolist()
    stages = feat_df["stage"].tolist() if "stage" in feat_df.columns else []
    v_means = feat_df["v_ego_mean"].tolist() if "v_ego_mean" in feat_df.columns else [0.0] * len(labels)

    n_driver = labels.count("driver")
    n_mechanical = labels.count("mechanical")
    print(f"\nTotal events: {len(labels):,}  |  driver: {n_driver:,}  |  mechanical: {n_mechanical:,}")

    if stages:
        latency = latency_histogram(stages)
        print("\nCascade stage distribution:")
        for stage_key, m in latency.items():
            print(f"  {stage_key}  {m['fraction']:.1%} ({m['count']})  ~{m['latency_ms']} ms")

    print()
    plot_features(feat_df, args.output, cfg)


if __name__ == "__main__":
    main()
