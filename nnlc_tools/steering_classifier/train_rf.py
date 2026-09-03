"""Random forest training pipeline for the steering event classifier.

Requires: pip install scikit-learn  (nnlc-tools[train])

Usage:
  python -m nnlc_tools.steering_classifier.train_rf \
      --features events_features.csv --labels events_labels.csv \
      --output rf_model.pkl
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "peak_torque_rate_nm_s",
    "duration_s",
    "sign_consistency",
    "zero_crossing_rate_hz",
    "torque_kurtosis",
    "has_longitudinal_shock",
    "torque_leads_angle",
    "torque_lat_accel_corr",
    "freq_energy_ratio",
    "speed_adjusted_is_brief",
    "lat_accel_residual",
    "v_ego_mean",
    "peak_steering_torque_abs",
]

LABEL_MAP = {"driver": 0, "mechanical": 1}
LABEL_INV = {0: "driver", 1: "mechanical"}


def load_dataset(features_path: str, labels_col: str = "label"):
    """Load a CSV with feature columns + a label column.

    Label column should contain "driver" or "mechanical".
    """
    df = pd.read_csv(features_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"WARNING: Missing feature columns: {missing}")

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()

    # Fill NaN with column medians (RF handles missing features this way)
    X = X.fillna(X.median())

    if labels_col not in df.columns:
        raise ValueError(f"Label column '{labels_col}' not found in {features_path}")

    y = df[labels_col].map(LABEL_MAP)
    if y.isna().any():
        bad = df[labels_col][y.isna()].unique()
        raise ValueError(f"Unknown label values: {bad}. Expected 'driver' or 'mechanical'.")

    return X.values, y.values.astype(int), available


def train(X, y, feature_names, cv_folds: int = 5, seed: int = 42):
    """Train RandomForestClassifier with stratified k-fold CV evaluation."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
    except ImportError:
        raise ImportError("scikit-learn is required: pip install 'nnlc-tools[train]'")

    clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_weighted")
    print(f"CV F1 (weighted, {cv_folds}-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)

    importances = sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:35s} {imp:.4f}")

    return clf, importances


def main():
    parser = argparse.ArgumentParser(description="Train RF classifier on labelled events CSV")
    parser.add_argument("features", help="CSV with feature columns + label column")
    parser.add_argument("--label-col", default="label",
                        help="Name of label column (default: label)")
    parser.add_argument("--output", default="rf_model.pkl",
                        help="Output path for pickled model (default: rf_model.pkl)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, feature_names = load_dataset(args.features, labels_col=args.label_col)
    print(f"Loaded {len(X)} events  ({np.sum(y == 0)} driver, {np.sum(y == 1)} mechanical)")

    clf, importances = train(X, y, feature_names, cv_folds=args.cv_folds, seed=args.seed)

    out = Path(args.output)
    with open(out, "wb") as f:
        pickle.dump({"model": clf, "feature_names": feature_names, "label_map": LABEL_MAP}, f)
    print(f"\nSaved model to {out}")

    imp_path = out.with_suffix(".importances.json")
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(dict(importances), f, ensure_ascii=False, indent=2)
    print(f"Saved importances to {imp_path}")


if __name__ == "__main__":
    main()
