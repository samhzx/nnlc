#!/usr/bin/env python3
"""Visualize lateral data coverage for NNLC training.

Generates a speed vs lateral acceleration heatmap with gap highlighting,
a lateral acceleration histogram, and override rate by speed.

Usage:
  python -m nnlc_tools.visualize_coverage output.csv -o coverage.png
  python -m nnlc_tools.visualize_coverage output.parquet -o coverage.png
  python -m nnlc_tools.visualize_coverage /path/to/rlogs/ -o coverage.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from nnlc_tools.bool_utils import parse_bool_series
from nnlc_tools.streaming_data import DEFAULT_CHUNK_ROWS, iter_csv_chunks


def load_data_for_viz(input_path):
    """Load data from CSV, Parquet, or directory of rlogs."""
    from nnlc_tools.data_io import load_data
    df = load_data(input_path)
    if df is None:
        print(f"ERROR: No data found at {input_path}")
        sys.exit(1)
    return df


def save_placeholder_plot(output_path, title, message):
    """Write a diagnostic image when required plotting columns are absent."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved placeholder coverage plot to {output_path}")


def plot_coverage(df, output_path, gap_threshold=50):
    """Generate coverage visualization with 6 subplots (2 rows × 3 columns).

    Top row: speed/lat-accel heatmap, lateral accel distribution, override rate by speed.
    Bottom row: intervention analysis — override rate by lat accel, override density
    heatmap, torque magnitude distribution during overrides.
    """
    df = df.copy()

    # All coverage panels use vehicle speed.  Return a useful diagnostic image
    # instead of raising KeyError when a hand-written or partial CSV is used.
    if "v_ego" not in df.columns:
        message = "Missing required column: v_ego"
        print(f"WARNING: {message}")
        save_placeholder_plot(output_path, "NNLC Training Data Coverage", message)
        return

    # Determine lateral accel column
    lat_accel_col = None
    for col in ["actual_lateral_accel", "desired_lateral_accel"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lat_accel_col = col
            break

    if lat_accel_col is None:
        if "desired_curvature" not in df.columns:
            message = "Missing lateral acceleration data and desired_curvature"
            print(f"WARNING: {message}")
            save_placeholder_plot(output_path, "NNLC Training Data Coverage", message)
            return
        print("WARNING: No lateral acceleration data found. Using desired_curvature * v_ego^2.")
        df["_lat_accel"] = df["desired_curvature"] * df["v_ego"] ** 2
        lat_accel_col = "_lat_accel"

    # Filter to active driving only
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= parse_bool_series(df["active"])
    if "standstill" in df.columns:
        mask &= ~parse_bool_series(df["standstill"])
    active_df = df[mask].copy()
    if "steering_pressed" in active_df.columns:
        # Keep rates numerically meaningful when CSV flags are strings.
        active_df["steering_pressed"] = parse_bool_series(active_df["steering_pressed"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("NNLC Training Data Coverage", fontsize=14, fontweight="bold")

    # 1. Speed vs Lateral Accel Heatmap
    ax1 = axes[0, 0]
    speed_bins = np.linspace(0, 40, 41)
    lat_bins = np.linspace(-3, 3, 61)

    valid = active_df[["v_ego", lat_accel_col]].dropna()
    h, xedges, yedges = np.histogram2d(
        valid["v_ego"].clip(0, 40),
        valid[lat_accel_col].clip(-3, 3),
        bins=[speed_bins, lat_bins],
    )

    # Highlight gaps
    h_display = h.copy()
    h_display[h_display == 0] = np.nan

    im = ax1.pcolormesh(
        xedges, yedges, h_display.T,
        norm=LogNorm(vmin=1, vmax=max(h.max(), 1)),
        cmap="viridis",
    )

    # Mark gaps in red
    gap_mask = (h > 0) & (h < gap_threshold)
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            if gap_mask[i, j]:
                ax1.add_patch(plt.Rectangle(
                    (xedges[i], yedges[j]),
                    xedges[i + 1] - xedges[i],
                    yedges[j + 1] - yedges[j],
                    linewidth=0.5, edgecolor="red", facecolor="none",
                ))

    fig.colorbar(im, ax=ax1, label="Sample count (log)")
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel(f"Lateral Accel (m/s²)")
    ax1.set_title("Speed vs Lat Accel\n(red outline = <50 samples)")

    # 2. Lateral Accel Histogram
    ax2 = axes[0, 1]
    lat_valid = active_df[lat_accel_col].dropna()
    ax2.hist(lat_valid.clip(-3, 3), bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax2.set_xlabel("Lateral Accel (m/s²)")
    ax2.set_ylabel("Count")
    ax2.set_title("Lateral Accel Distribution")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Add stats
    stats_text = (
        f"Mean: {lat_valid.mean():.3f}\n"
        f"Std:  {lat_valid.std():.3f}\n"
        f"|>1|: {(lat_valid.abs() > 1).mean():.1%}\n"
        f"|>2|: {(lat_valid.abs() > 2).mean():.1%}"
    )
    ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
             verticalalignment="top", horizontalalignment="right",
             fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 3. Override Rate by Speed
    ax3 = axes[0, 2]
    if "steering_pressed" in df.columns:
        speed_bins_override = np.arange(0, 42, 2)
        df_with_bin = active_df.copy()
        df_with_bin["speed_bin"] = pd.cut(df_with_bin["v_ego"], bins=speed_bins_override)
        override_by_speed = df_with_bin.groupby("speed_bin", observed=True)["steering_pressed"].mean() * 100

        centers = [(b.left + b.right) / 2 for b in override_by_speed.index]
        ax3.bar(centers, override_by_speed.values, width=1.5, color="coral", edgecolor="none", alpha=0.8)
        ax3.set_xlabel("Speed (m/s)")
        ax3.set_ylabel("Override Rate (%)")
        ax3.set_title("Steering Override by Speed")
        ax3.axhline(10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No steering_pressed\ndata available",
                 transform=ax3.transAxes, ha="center", va="center")
        ax3.set_title("Steering Override by Speed")

    # ── Row 2: Intervention analysis ─────────────────────────────────────────
    has_overrides = "steering_pressed" in df.columns

    # 4. Override Density Heatmap (speed × lat_accel)
    ax4 = axes[1, 0]
    if has_overrides:
        override_df = active_df[parse_bool_series(active_df["steering_pressed"])]
        if len(override_df) > 0:
            valid_ov = override_df[["v_ego", lat_accel_col]].dropna()
            h_ov, xedges_ov, yedges_ov = np.histogram2d(
                valid_ov["v_ego"].clip(0, 40),
                valid_ov[lat_accel_col].clip(-3, 3),
                bins=[speed_bins, lat_bins],
            )
            h_ov_display = h_ov.copy()
            h_ov_display[h_ov_display == 0] = np.nan
            im_ov = ax4.pcolormesh(
                xedges_ov, yedges_ov, h_ov_display.T,
                norm=LogNorm(vmin=1, vmax=max(h_ov.max(), 1)),
                cmap="viridis",
            )
            fig.colorbar(im_ov, ax=ax4, label="Override count (log)")
            ax4.set_xlabel("Speed (m/s)")
            ax4.set_ylabel("Lateral Accel (m/s²)")
        else:
            ax4.text(0.5, 0.5, "No override events",
                     transform=ax4.transAxes, ha="center", va="center")
    else:
        ax4.text(0.5, 0.5, "No steering_pressed\ndata available",
                 transform=ax4.transAxes, ha="center", va="center")
    ax4.set_title("Override Concentration\n(speed × lat accel)")

    # 5. Override Rate by Lat Accel
    ax5 = axes[1, 2]
    if has_overrides:
        lat_bins_override = np.arange(-3, 3.2, 0.2)
        df_lat = active_df.copy()
        df_lat["lat_bin"] = pd.cut(df_lat[lat_accel_col], bins=lat_bins_override)
        override_by_lat = df_lat.groupby("lat_bin", observed=True)["steering_pressed"].mean() * 100
        centers = [(b.left + b.right) / 2 for b in override_by_lat.index]
        ax5.bar(centers, override_by_lat.values, width=0.18, color="coral", edgecolor="none", alpha=0.8)
        ax5.axhline(10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
        ax5.legend(fontsize=8)
        ax5.set_xlabel("Lateral Accel (m/s²)")
        ax5.set_ylabel("Override Rate (%)")
    else:
        ax5.text(0.5, 0.5, "No steering_pressed\ndata available",
                 transform=ax5.transAxes, ha="center", va="center")
    ax5.set_title("Steering Override by Lat Accel")

    # 6. Torque Magnitude During Overrides
    ax6 = axes[1, 1]
    if has_overrides:
        override_df = active_df[parse_bool_series(active_df["steering_pressed"])]
        if "steering_torque" in df.columns and len(override_df) > 0:
            torque_mag = override_df["steering_torque"].abs().dropna()
            ax6.hist(torque_mag, bins=40, color="coral", edgecolor="none", alpha=0.8)
            ax6.set_xlabel("Steering Torque Magnitude")
            ax6.set_ylabel("Count")
            n_events = len(torque_mag)
            median_torque = torque_mag.median()
            ax6.annotate(
                f"n = {n_events:,}\nmedian = {median_torque:.2f}",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax6.text(0.5, 0.5, "No override events\nor no torque data",
                     transform=ax6.transAxes, ha="center", va="center")
    else:
        ax6.text(0.5, 0.5, "No steering_pressed\ndata available",
                 transform=ax6.transAxes, ha="center", va="center")
    ax6.set_title("Torque Magnitude During Overrides")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved coverage plot to {output_path}")
    plt.close()


def plot_coverage_stream(input_path, output_path, gap_threshold=50,
                         chunksize=DEFAULT_CHUNK_ROWS):
    """Generate coverage plots from bounded CSV chunks."""
    header = pd.read_csv(input_path, nrows=0)
    if "v_ego" not in header.columns:
        message = "Missing required column: v_ego"
        print(f"WARNING: {message}")
        save_placeholder_plot(output_path, "NNLC Training Data Coverage", message)
        return
    lat_accel_col = next(
        (col for col in ("actual_lateral_accel", "desired_lateral_accel")
         if col in header.columns),
        None,
    )
    if lat_accel_col is None and "desired_curvature" not in header.columns:
        message = "Missing lateral acceleration data and desired_curvature"
        print(f"WARNING: {message}")
        save_placeholder_plot(output_path, "NNLC Training Data Coverage", message)
        return
    speed_bins = np.linspace(0, 40, 41)
    lat_bins = np.linspace(-3, 3, 61)
    h = np.zeros((40, 60), dtype=np.int64)
    h_override = np.zeros_like(h)
    lat_hist = np.zeros(60, dtype=np.int64)
    speed_total = np.zeros(20, dtype=np.int64)
    speed_override = np.zeros(20, dtype=np.int64)
    lat_total = np.zeros(30, dtype=np.int64)
    lat_override = np.zeros(30, dtype=np.int64)
    torque_hist = np.zeros(40, dtype=np.int64)
    torque_edges = np.linspace(0, 10, 41)
    valid_rows = 0

    for chunk in iter_csv_chunks(input_path, chunksize=chunksize):
        mask = pd.Series(True, index=chunk.index)
        if "active" in chunk:
            mask &= parse_bool_series(chunk["active"])
        if "standstill" in chunk:
            mask &= ~parse_bool_series(chunk["standstill"])
        active = chunk.loc[mask]
        if lat_accel_col is None:
            if not {"desired_curvature", "v_ego"}.issubset(active.columns):
                continue
            values = active["desired_curvature"] * active["v_ego"] ** 2
        else:
            values = active[lat_accel_col]
        valid = pd.DataFrame({"speed": active["v_ego"], "lat": values}).dropna()
        if valid.empty:
            continue
        valid_rows += len(valid)
        speed = valid["speed"].to_numpy()
        lat = valid["lat"].to_numpy()
        h += np.histogram2d(np.clip(speed, 0, 40), np.clip(lat, -3, 3),
                            bins=[speed_bins, lat_bins])[0].astype(np.int64)
        lat_hist += np.histogram(np.clip(lat, -3, 3), bins=lat_bins)[0]
        speed_idx = np.clip((speed // 2).astype(int), 0, 19)
        speed_total += np.bincount(speed_idx, minlength=20)
        lat_idx = np.clip(((lat + 3) // .2).astype(int), 0, 29)
        lat_total += np.bincount(lat_idx, minlength=30)
        if "steering_pressed" in active:
            override = parse_bool_series(active.loc[valid.index, "steering_pressed"]).to_numpy()
            h_override += np.histogram2d(
                np.clip(speed[override], 0, 40), np.clip(lat[override], -3, 3),
                bins=[speed_bins, lat_bins],
            )[0].astype(np.int64)
            speed_override += np.bincount(speed_idx[override], minlength=20)
            lat_override += np.bincount(lat_idx[override], minlength=30)
            if "steering_torque" in active:
                torque = active.loc[valid.index[override], "steering_torque"].abs().dropna().to_numpy()
                torque_hist += np.histogram(np.clip(torque, 0, 10), bins=torque_edges)[0]

    if valid_rows == 0:
        raise ValueError("no usable active rows available for coverage plot")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("NNLC Training Data Coverage", fontsize=14, fontweight="bold")
    display = h.astype(float)
    display[display == 0] = np.nan
    im = axes[0, 0].pcolormesh(
        speed_bins, lat_bins, display.T,
        norm=LogNorm(vmin=1, vmax=max(int(h.max()), 2)), cmap="viridis",
    )
    fig.colorbar(im, ax=axes[0, 0], label="Sample count (log)")
    for i in range(40):
        for j in range(60):
            if 0 < h[i, j] < gap_threshold:
                axes[0, 0].add_patch(plt.Rectangle(
                    (speed_bins[i], lat_bins[j]), 1, .1,
                    linewidth=.5, edgecolor="red", facecolor="none"))
    axes[0, 0].set(xlabel="Speed (m/s)", ylabel="Lateral Accel (m/s²)",
                   title="Speed vs Lat Accel\n(red outline = <50 samples)")
    centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    axes[0, 1].bar(centers, lat_hist, width=.09, color="steelblue", edgecolor="none", alpha=.8)
    axes[0, 1].set(xlabel="Lateral Accel (m/s²)", ylabel="Count", title="Lateral Accel Distribution")
    speed_centers = np.arange(1, 40, 2)
    speed_rate = np.divide(speed_override, speed_total, out=np.zeros(20), where=speed_total > 0) * 100
    axes[0, 2].bar(speed_centers, speed_rate, width=1.5, color="coral", edgecolor="none", alpha=.8)
    axes[0, 2].set(xlabel="Speed (m/s)", ylabel="Override Rate (%)", title="Steering Override by Speed")
    axes[0, 2].axhline(10, color="red", linestyle="--", alpha=.5)
    if h_override.any():
        ov_display = h_override.astype(float)
        ov_display[ov_display == 0] = np.nan
        im_ov = axes[1, 0].pcolormesh(
            speed_bins, lat_bins, ov_display.T,
            norm=LogNorm(vmin=1, vmax=max(int(h_override.max()), 2)), cmap="viridis",
        )
        fig.colorbar(im_ov, ax=axes[1, 0], label="Override count (log)")
    else:
        axes[1, 0].text(0.5, 0.5, "No override events",
                        transform=axes[1, 0].transAxes, ha="center", va="center")
    axes[1, 0].set(xlabel="Speed (m/s)", ylabel="Lateral Accel (m/s²)", title="Override Concentration")
    axes[1, 1].bar(torque_edges[:-1], torque_hist, width=.24, color="coral", edgecolor="none", alpha=.8)
    axes[1, 1].set(xlabel="Steering Torque Magnitude", ylabel="Count", title="Torque Magnitude During Overrides")
    lat_centers = (np.arange(30) + .5) * .2 - 3
    lat_rate = np.divide(lat_override, lat_total, out=np.zeros(30), where=lat_total > 0) * 100
    axes[1, 2].bar(lat_centers, lat_rate, width=.18, color="coral", edgecolor="none", alpha=.8)
    axes[1, 2].set(xlabel="Lateral Accel (m/s²)", ylabel="Override Rate (%)", title="Steering Override by Lat Accel")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved coverage plot to {output_path}")


MS_TO_MPH = 2.23694


def plot_torque_scatter(df, output_path, max_points=None):
    """Generate lat_accel vs torque scatter plots split by speed bin (10 mph steps)."""
    import math

    import matplotlib.pyplot as plt

    def save_empty_plot(message):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, message, ha="center", va="center")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved torque scatter plot to {output_path}")
        plt.close()

    # Determine lateral accel column
    lat_accel_col = None
    for col in ["actual_lateral_accel", "desired_lateral_accel"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lat_accel_col = col
            break

    if lat_accel_col is None:
        print("WARNING: No lateral acceleration data found for torque scatter.")
        save_empty_plot("No lateral acceleration data")
        return

    if "torque_output" not in df.columns:
        print("WARNING: No torque_output column found. Skipping torque scatter plot.")
        save_empty_plot("No torque output data")
        return

    # Filter to active driving only
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= parse_bool_series(df["active"])
    if "standstill" in df.columns:
        mask &= ~parse_bool_series(df["standstill"])
    active_df = df[mask].copy()

    if "v_ego" not in active_df.columns:
        message = "Missing required column: v_ego"
        print(f"WARNING: {message}")
        save_placeholder_plot(output_path, "Lateral Accel vs Torque", message)
        return

    valid = active_df[[lat_accel_col, "torque_output", "v_ego"]].dropna()
    valid = valid.copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH

    speed_bins = list(range(0, 90, 10))
    n_bins = len(speed_bins)
    ncols = 3
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Lateral Accel vs Torque by Speed Bin", fontsize=14, fontweight="bold")

    for i, speed_lo in enumerate(speed_bins):
        speed_hi = speed_lo + 10
        ax = axes[i]

        bin_data = valid[(valid["speed_mph"] >= speed_lo) & (valid["speed_mph"] < speed_hi)]
        plot_data = bin_data.sample(n=max_points, random_state=42) if max_points and len(bin_data) > max_points else bin_data
        sc = ax.scatter(plot_data[lat_accel_col], plot_data["torque_output"],
                        c=plot_data["speed_mph"], cmap="viridis",
                        vmin=speed_lo, vmax=speed_hi,
                        s=1.0, alpha=0.3, rasterized=True)
        fig.colorbar(sc, ax=ax, label="Speed (mph)", pad=0.02)

        ax.set_title(f"{speed_lo}-{speed_hi} mph (n={len(bin_data)})", fontsize=10)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Lat Accel (m/s²)")
        ax.set_ylabel("Torque")
        ax.grid(axis="x", color="0.95")
        ax.grid(axis="y", color="0.95")

    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved torque scatter plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lateral data coverage for NNLC training.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("-o", "--output", default="coverage.png",
                        help="Output image path (default: coverage.png)")
    parser.add_argument("--gap-threshold", type=int, default=50,
                        help="Highlight bins with fewer than this many samples (default: 50)")
    parser.add_argument("--torque-scatter", action="store_true",
                        help="Generate a separate lat_accel vs torque scatter plot")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max data points per torque scatter subplot (random sample)")
    parser.add_argument("--streaming", action="store_true",
                        help="Process CSV in bounded chunks")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
                        help=f"Rows per streaming chunk (default: {DEFAULT_CHUNK_ROWS:,})")
    args = parser.parse_args()

    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be a positive integer")
    if args.streaming:
        if not os.path.isfile(args.input) or not args.input.lower().endswith(".csv"):
            parser.error("--streaming only supports an existing CSV file")
        if args.torque_scatter:
            parser.error("--torque-scatter is not available with --streaming")
        try:
            plot_coverage_stream(args.input, args.output, args.gap_threshold,
                                 chunksize=args.chunk_rows)
        except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        return

    df = load_data_for_viz(args.input)
    print(f"Loaded {len(df)} rows")

    plot_coverage(df, args.output, args.gap_threshold)

    if args.torque_scatter:
        # Save alongside the main coverage plot
        out_dir = os.path.dirname(args.output) or "."
        scatter_path = os.path.join(out_dir, "lat_accel_vs_torque_data.png")
        plot_torque_scatter(df, scatter_path, max_points=args.max_points)


if __name__ == "__main__":
    main()
