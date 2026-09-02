#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
  echo "uv installation completed but the executable is not on PATH" >&2
  exit 1
fi

echo "Creating virtual environment..."
uv venv --clear

echo "Installing dependencies..."
uv pip install -e ".[train]"

echo ""
echo "Done! Run tools with:"
echo "  uv run nnlc-extract ./data -o output/lateral_data.csv --temporal"
echo ""
echo "Or run the full pipeline:"
echo "  bash scripts/prepare_training_data.sh"
