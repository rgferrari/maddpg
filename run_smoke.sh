#!/usr/bin/env bash
set -euo pipefail

# Run a short, non-interactive MADDPG smoke training test.
# Usage:
#   ./run_smoke.sh
#   ./run_smoke.sh --scenario simple_tag

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_DIR="${SAVE_DIR:-/tmp/maddpg_policy}"
PLOTS_DIR="${PLOTS_DIR:-/tmp/maddpg_plots}"
EXP_NAME="${EXP_NAME:-smoke}"

cd "$ROOT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "maddpg_py35" ]]; then
  echo "Warning: expected conda env 'maddpg_py35' (current: '${CONDA_DEFAULT_ENV:-none}')."
  echo "Activate it first with: conda activate maddpg_py35"
fi

# Ensure local package imports from this repo.
python -m pip install -e .

# MPE emits an interactive prompt unless this is set.
export SUPPRESS_MA_PROMPT=1

mkdir -p "$SAVE_DIR" "$PLOTS_DIR"

cd experiments
python train.py \
  --scenario simple \
  --num-episodes 10 \
  --max-episode-len 5 \
  --save-rate 5 \
  --exp-name "$EXP_NAME" \
  --save-dir "$SAVE_DIR/" \
  --plots-dir "$PLOTS_DIR/" \
  "$@"

echo "Smoke run complete."
echo "Checkpoints: $SAVE_DIR"
echo "Plots:       $PLOTS_DIR"
