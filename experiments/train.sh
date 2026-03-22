#!/usr/bin/env bash
set -euo pipefail

# MADDPG training launcher for WSL2.
# Usage:
#   ./train.sh adversary
#   ./train.sh simple_good --scenario simple --num-episodes 5000
#
# Arguments:
#   $1: experiment name (required, e.g., "adversary", "cooperation")
#   $2: scenario (optional, default: simple_adversary)
#   $3: num-episodes (optional, default: 50000)

EXP_NAME="${1:-}"
SCENARIO="${2:-simple_adversary}"
NUM_EPISODES="${3:-50000}"

if [[ -z "$EXP_NAME" ]]; then
  echo "Usage: $0 <exp-name> [scenario] [num-episodes]"
  echo "Example: $0 adversary simple_adversary 50000"
  exit 1
fi

# Change to script directory so train.py is found
cd "$(dirname "$0")"

SAVE_DIR="../checkpoints/$EXP_NAME"
PLOTS_DIR="../learning_curves"

echo "========================================"
echo "MADDPG Training"
echo "========================================"
echo "Experiment:  $EXP_NAME"
echo "Scenario:    $SCENARIO"
echo "Episodes:    $NUM_EPISODES"
echo "Save dir:    $SAVE_DIR"
echo "Plots dir:   $PLOTS_DIR"
echo "========================================"
echo ""

# Create directories
mkdir -p "$SAVE_DIR"
mkdir -p "$PLOTS_DIR"

# Use conda run instead of activate for non-interactive shells
conda run -n maddpg_py35 env SUPPRESS_MA_PROMPT=1 python train.py \
  --scenario "$SCENARIO" \
  --exp-name "$EXP_NAME" \
  --num-episodes "$NUM_EPISODES" \
  --save-dir "$SAVE_DIR/" \
  --plots-dir "$PLOTS_DIR/"

echo ""
echo "========================================"
echo "Training complete!"
echo "Model saved to:   $SAVE_DIR"
echo "Metrics saved to: $PLOTS_DIR"
echo "========================================"
