#!/usr/bin/env bash

# Reproduce every required condition for one seed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEED="${1:-42}"

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 [integer-seed]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -d "dataset/UCI HAR Dataset/train/Inertial Signals" ]]; then
  echo "Extracting the included UCI-HAR archive..."
  uv run python -m zipfile -e "dataset/UCI HAR Dataset.zip" dataset
fi

echo "Reproducing all required conditions for seed $SEED"
uv run python scripts/pretrain_encoder.py "$SEED"
uv run python scripts/train_direct.py "$SEED"
uv run python scripts/train_context.py "$SEED"
uv run python scripts/eval_shuffle.py "$SEED"

echo "Finished seed $SEED. Results are under outputs/reproduction/."
