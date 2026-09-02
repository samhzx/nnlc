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
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def load_data_for_viz(input_path):
    """Load data from CSV, Parquet, or directory of rlogs."""
    from nnlc_tools.data_io import load_data
    df = load_data(input_path)
    if df is None:
        print(f"ERROR: No data found at {input_path}")
        sys.exit(1)
    return df


def plot_coverage(df, output_path, gap_threshold=50):
    """Generate coverage visualization with 6 subplots (2 rows × 3 columns).

    Top row: speed/lat-accel heatmap, lateral accel distribution, override rate by speed.
    Bottom row: intervention analysis — override rate by lat accel, override density
    heatmap, torque magnitude distribution during overrides.
    """
    # Determine lateral accel column
    lat_accel_col = None
    for col in ["actual_lateral_accel", "desired_lateral_accel"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lat_accel_col = col
            break

    if lat_accel_col is None:
        print("WARNING: No lateral acceleration data found. Using desired_curvature * v_ego^2.")
        df["_lat_accel"] = df["desired_curvature"] * df["v_ego"] ** 2
        lat_accel_col = "_lat_accel"

    # Filter to active driving only
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    active_df = df[mask].copy()

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
        override_df = active_df[active_df["steering_pressed"].astype(bool)]
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
        override_df = active_df[active_df["steering_pressed"].astype(bool)]
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
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    active_df = df[mask].copy()

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
    args = parser.parse_args()

    df = load_data_for_viz(args.input)
    print(f"Loaded {len(df)} rows")

    plot_coverage(df, args.output, args.gap_threshold)

    if args.torque_scatter:
        # Save alongside the main coverage plot
        import os
        out_dir = os.path.dirname(args.output) or "."
        scatter_path = os.path.join(out_dir, "lat_accel_vs_torque_data.png")
        plot_torque_scatter(df, scatter_path, max_points=args.max_points)


if __name__ == "__main__":
    main()
