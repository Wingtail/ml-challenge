#!/usr/bin/env bash

# Reproduce the reported four-seed aggregate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for seed in 42 43 44 45; do
  "$SCRIPT_DIR/reproduce_seed.sh" "$seed"
done

cd "$SCRIPT_DIR/.."
uv run python scripts/summarize_results.py

echo "Finished seeds 42, 43, 44, and 45."
