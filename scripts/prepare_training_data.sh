#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Defaults
DEVICE=""
OUTPUT_DIR="./output"
DATA_DIR="./data"
MIN_SCORE=""
PRUNE="both"

usage() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo ""
  echo "Run the full NNLC data preparation pipeline: sync → extract → score → prune routes → visualize → classify & prune"
  echo ""
  echo "Options:"
  echo "  -d, --device IP              Device IP for rlog sync (skip sync if omitted)"
  echo "  -o, --output DIR             Output directory (default: ./output)"
  echo "  --data-dir DIR               Rlog directory (default: ./data)"
  echo "  --min-score N                Exclude routes scoring below N in prune step (default: 0)"
  echo "  --prune mechanical|driver|both|none"
  echo "                               Event types to remove in step 6 (default: both)"
  echo "                               Use 'none' to skip the prune step"
  echo "  -h, --help                   Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device)  DEVICE="$2"; shift 2 ;;
    -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --min-score)  MIN_SCORE="$2"; shift 2 ;;
    --prune)      PRUNE="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -n "$MIN_SCORE" ]] && {
  ! [[ "$MIN_SCORE" =~ ^[0-9]+$ ]] || (( MIN_SCORE < 0 || MIN_SCORE > 100 ));
}; then
  echo "Error: --min-score must be an integer from 0 to 100" >&2
  exit 1
fi

banner() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════"
  echo ""
}

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="$OUTPUT_DIR/$TIMESTAMP"
TRAIN_DIR="$RUN_DIR/train"
PLOTS_DIR="$RUN_DIR/data_prep_results"
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$PLOTS_DIR"

# Step 1: Sync rlogs from device
if [[ -n "$DEVICE" ]]; then
  banner "Step 1/6: Syncing rlogs from $DEVICE"
  uv run nnlc-sync -d "$DEVICE" -o "$DATA_DIR"
else
  banner "Step 1/6: Sync skipped (no --device specified)"
  echo "Using existing rlogs in $DATA_DIR"
fi

# Step 2: Extract lateral data with temporal features
banner "Step 2/6: Extracting lateral data"
uv run nnlc-extract "$DATA_DIR" -o "$RUN_DIR/lateral_data.csv" --temporal

# Step 3: Score routes
banner "Step 3/6: Scoring routes"
SCORE_ARGS=("$RUN_DIR/lateral_data.csv")
if [[ -n "$MIN_SCORE" ]]; then
  SCORE_ARGS+=(--min-score "$MIN_SCORE")
fi
uv run nnlc-score "${SCORE_ARGS[@]}"

# Step 4: Prune routes
banner "Step 4/6: Pruning routes"
PRUNE_ARGS=("$RUN_DIR/lateral_data.csv" -o "$RUN_DIR/lateral_data_routes_pruned.csv")
if [[ -n "$MIN_SCORE" ]]; then
  PRUNE_ARGS+=(--min-score "$MIN_SCORE")
fi
uv run nnlc-prune-routes "${PRUNE_ARGS[@]}"

# Step 5: Visualize coverage
banner "Step 5/6: Visualizing data coverage"
uv run nnlc-visualize "$RUN_DIR/lateral_data_routes_pruned.csv" \
    -o "$PLOTS_DIR/coverage.png" --torque-scatter

# Step 6: Classify interventions and prune
if [[ "$PRUNE" != "none" ]]; then
  banner "Step 6/6: Classifying interventions and pruning ($PRUNE)"
  uv run nnlc-interventions "$RUN_DIR/lateral_data_routes_pruned.csv" \
      --prune "$PRUNE" \
      --prune-output "$TRAIN_DIR/lateral_data_pruned.csv" \
      --plot --scatter -o "$PLOTS_DIR/interventions.png"

  uv run nnlc-sc-visualize "$RUN_DIR/lateral_data_routes_pruned.csv" \
      -o "$PLOTS_DIR/sc_features.png"
  uv run nnlc-visualize "$TRAIN_DIR/lateral_data_pruned.csv" \
      -o "$PLOTS_DIR/coverage_pruned.png" --torque-scatter
  TRAIN_INPUT="$TRAIN_DIR"
else
  banner "Step 6/6: Prune skipped (--prune none)"
  cp "$RUN_DIR/lateral_data_routes_pruned.csv" "$TRAIN_DIR/lateral_data_routes_pruned.csv"
  TRAIN_INPUT="$TRAIN_DIR"
fi

banner "Pipeline complete!"
echo "Outputs:"
echo "  Data (raw):          $RUN_DIR/lateral_data.csv"
echo "  Data (route-pruned): $RUN_DIR/lateral_data_routes_pruned.csv"
if [[ "$PRUNE" != "none" ]]; then
  echo "  Data (pruned):       $TRAIN_DIR/lateral_data_pruned.csv"
fi
echo "  Visualizations:      $PLOTS_DIR/"
echo "    coverage.png"
echo "    lat_accel_vs_torque_data.png"
if [[ "$PRUNE" != "none" ]]; then
  echo "    interventions.png"
  echo "    interventions_scatter.png"
  echo "    sc_features.png"
  echo "    coverage_pruned.png"
fi
echo ""
echo "Next step: review $PLOTS_DIR/coverage_pruned.png for gaps, then train:"
echo "  bash training/run.sh $TRAIN_INPUT"
