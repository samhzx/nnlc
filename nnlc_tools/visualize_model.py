#!/usr/bin/env python3
"""Visualize trained NNLC model predictions against actual data.

Generates two plot types matching sunnypilot PR validation style:
- lat_accel_vs_torque: one subplot per speed bin, data scatter + model curve
- torque_vs_speed: one subplot per lat_accel bin, data scatter + model curve

Usage:
  python -m nnlc_tools.visualize_model model.json data.csv -o output/
"""

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def identity(x):
    return x


def tanh(x):
    return np.tanh(x)


def leakyrelu(x):
    return np.where(x > 0, x, 0.01 * x)


ACTIVATIONS = {
    "sigmoid": sigmoid,
    "σ": sigmoid,
    "identity": identity,
    "tanh": tanh,
    "leakyrelu": leakyrelu,
}


class NNModel:
    """Lightweight NN model loaded from JSON for inference."""

    def __init__(self, params_file):
        with open(params_file) as f:
            params = json.load(f)

        self.input_size = params["input_size"]
        self.input_mean = np.array(params["input_mean"], dtype=np.float32).T.flatten()
        self.input_std = np.array(params["input_std"], dtype=np.float32).T.flatten()
        self.input_vars = params.get("input_vars", [])

        self.layers = []
        for layer_params in params["layers"]:
            W_key = next(k for k in layer_params if k.endswith("_W"))
            b_key = next(k for k in layer_params if k.endswith("_b"))
            W = np.array(layer_params[W_key], dtype=np.float32).T
            b = np.array(layer_params[b_key], dtype=np.float32).T.flatten()
            act_name = layer_params["activation"]
            act_fn = ACTIVATIONS.get(act_name)
            if act_fn is None:
                raise ValueError(f"Unknown activation: {act_name}")
            self.layers.append((W, b, act_fn))

    def forward(self, x):
        """Forward pass. x shape: (batch, input_size)."""
        for W, b, act in self.layers:
            x = act(x @ W + b)
        return x

    def predict(self, x):
        """Normalize inputs and run forward pass."""
        x_norm = (x - self.input_mean) / self.input_std
        return self.forward(x_norm)

    def var_index(self, name):
        """Get index of an input variable by name."""
        return self.input_vars.index(name)

    def make_input_at_means(self, n=1):
        """Create input array filled with mean values (un-normalized space)."""
        return np.tile(self.input_mean, (n, 1))


def load_data(csv_path):
    """Load training data CSV."""
    if csv_path.endswith(".parquet"):
        return pd.read_parquet(csv_path)
    return pd.read_csv(csv_path)


def filter_active(df):
    """Filter to active, non-standstill rows."""
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    return df[mask]


MS_TO_MPH = 2.23694


def save_placeholder_plot(output_path, title, message):
    """Write a diagnostic image even when the input data cannot be plotted."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved placeholder plot to {output_path}")


def resolve_lateral_columns(model, df):
    """Resolve the observed and model lateral-acceleration columns.

    New temporal models use ``desired_lateral_accel`` as the base feature,
    while older models may use ``actual_lateral_accel``.  Keep the plotted
    data and model sweep aligned with whichever base feature the model has.
    """
    data_candidates = ["actual_lateral_accel", "desired_lateral_accel"]
    available_data = [column for column in data_candidates if column in df.columns]
    if not available_data:
        return None, None

    for column in data_candidates:
        if column in df.columns and column in model.input_vars:
            return column, column
    for column in data_candidates:
        if column in model.input_vars:
            return available_data[0], column
    return available_data[0], None


def plot_lat_accel_vs_torque(model, df, output_path):
    """Generate per-speed-bin plots: lat_accel (x) vs torque (y)."""
    df = filter_active(df)

    lat_col, model_lat_col = resolve_lateral_columns(model, df)
    required = ["v_ego", "torque_output"]
    missing = [column for column in required if column not in df.columns]
    if lat_col is None:
        missing.append("actual_lateral_accel or desired_lateral_accel")
    if model_lat_col is None:
        missing.append("matching lateral acceleration input in model")
    if "v_ego" not in model.input_vars:
        missing.append("v_ego input in model")
    if missing:
        message = "Missing required columns: " + ", ".join(missing)
        print(f"WARNING: {message}")
        save_placeholder_plot(output_path, "Lateral Accel vs Torque", message)
        return

    valid = df[["v_ego", lat_col, "torque_output"]].dropna()
    valid = valid.copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH

    speed_bins = list(range(0, 90, 10))
    n_bins = len(speed_bins)
    ncols = 3
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Lateral Accel vs Torque by Speed Bin", fontsize=14, fontweight="bold")

    lat_sweep = np.linspace(-3.5, 3.5, 200)
    lat_idx = model.var_index(model_lat_col)
    speed_idx = model.var_index("v_ego")

    for i, speed_lo in enumerate(speed_bins):
        speed_hi = speed_lo + 10
        ax = axes[i]

        bin_data = valid[(valid["speed_mph"] >= speed_lo) & (valid["speed_mph"] < speed_hi)]
        sc = ax.scatter(bin_data[lat_col], bin_data["torque_output"],
                        c=bin_data["speed_mph"], cmap="viridis",
                        vmin=speed_lo, vmax=speed_hi,
                        s=0.5, alpha=0.3, rasterized=True)
        fig.colorbar(sc, ax=ax, label="Speed (mph)", pad=0.02)

        # Model prediction curve at bin midpoint speed
        mid_speed_ms = (speed_lo + speed_hi) / 2.0 / MS_TO_MPH
        x_input = model.make_input_at_means(len(lat_sweep))
        x_input[:, speed_idx] = mid_speed_ms
        x_input[:, lat_idx] = lat_sweep
        # Set temporal lat_accel vars to the sweep value too
        for var in model.input_vars:
            if var.startswith("actual_lateral_accel_t") or var.startswith("desired_lateral_accel_t"):
                x_input[:, model.var_index(var)] = lat_sweep
        pred = model.predict(x_input).flatten()
        ax.plot(lat_sweep, pred, color="blue", linewidth=1.5)

        ax.set_title(f"{speed_lo}-{speed_hi} mph (n={len(bin_data)})", fontsize=10)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Lat Accel (m/s²)")
        ax.set_ylabel("Torque")

    # Hide unused subplots
    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved lat_accel_vs_torque plot to {output_path}")
    plt.close()


def plot_torque_vs_speed(model, df, output_path):
    """Generate per-lat_accel-bin plots: speed (x) vs torque (y)."""
    df = filter_active(df)

    lat_col, model_lat_col = resolve_lateral_columns(model, df)
    required = ["v_ego", "torque_output"]
    missing = [column for column in required if column not in df.columns]
    if lat_col is None:
        missing.append("actual_lateral_accel or desired_lateral_accel")
    if model_lat_col is None:
        missing.append("matching lateral acceleration input in model")
    if "v_ego" not in model.input_vars:
        missing.append("v_ego input in model")
    if missing:
        message = "Missing required columns: " + ", ".join(missing)
        print(f"WARNING: {message}")
        save_placeholder_plot(output_path, "Torque vs Speed", message)
        return

    valid = df[["v_ego", lat_col, "torque_output"]].dropna()
    valid = valid.copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH
    valid["abs_lat_accel"] = valid[lat_col].abs()
    valid["abs_torque"] = valid["torque_output"].abs()

    lat_bin_edges = np.arange(0.0, 3.2, 0.2)
    n_bins = len(lat_bin_edges) - 1
    ncols = 4
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Torque vs Speed by Lateral Accel Bin", fontsize=14, fontweight="bold")

    speed_sweep_mph = np.linspace(0, 90, 200)
    speed_sweep_ms = speed_sweep_mph / MS_TO_MPH
    lat_idx = model.var_index(model_lat_col)
    speed_idx = model.var_index("v_ego")

    for i in range(n_bins):
        lat_lo = lat_bin_edges[i]
        lat_hi = lat_bin_edges[i + 1]
        ax = axes[i]

        bin_data = valid[(valid["abs_lat_accel"] >= lat_lo) & (valid["abs_lat_accel"] < lat_hi)]
        sc = ax.scatter(bin_data["speed_mph"], bin_data["abs_torque"],
                        c=bin_data["abs_lat_accel"], cmap="viridis",
                        vmin=lat_lo, vmax=lat_hi,
                        s=0.5, alpha=0.3, rasterized=True)
        fig.colorbar(sc, ax=ax, label="Lat Accel (m/s²)", pad=0.02)

        # Model prediction curve at bin midpoint lat_accel
        mid_lat = (lat_lo + lat_hi) / 2.0
        x_input = model.make_input_at_means(len(speed_sweep_ms))
        x_input[:, speed_idx] = speed_sweep_ms
        x_input[:, lat_idx] = mid_lat
        for var in model.input_vars:
            if var.startswith("actual_lateral_accel_t") or var.startswith("desired_lateral_accel_t"):
                x_input[:, model.var_index(var)] = mid_lat
        pred = np.abs(model.predict(x_input).flatten())
        ax.plot(speed_sweep_mph, pred, color="blue", linewidth=1.5)

        ax.set_title(f"{lat_lo:.1f}-{lat_hi:.1f} m/s² (n={len(bin_data)})", fontsize=10)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel("Speed (mph)")
        ax.set_ylabel("Torque")

    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved torque_vs_speed plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained NNLC model predictions against actual data.",
    )
    parser.add_argument("model", help="Trained model JSON file")
    parser.add_argument("data", help="Training data CSV/Parquet file")
    parser.add_argument("-o", "--output-dir", default="./output/",
                        help="Output directory for plots (default: ./output/)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    model = NNModel(args.model)
    print(f"Loaded model: {model.input_size} inputs, {len(model.layers)} layers")
    print(f"  Input vars: {model.input_vars}")

    df = load_data(args.data)
    print(f"Loaded {len(df)} data rows")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_lat_accel_vs_torque(model, df, os.path.join(args.output_dir, "lat_accel_vs_torque.png"))
    plot_torque_vs_speed(model, df, os.path.join(args.output_dir, "torque_vs_speed.png"))


if __name__ == "__main__":
    main()
