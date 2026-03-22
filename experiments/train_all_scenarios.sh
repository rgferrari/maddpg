#!/usr/bin/env bash
set -euo pipefail

# Batch training on all MPE scenarios.
# Usage:
#   ./train_all_scenarios.sh
#   ./train_all_scenarios.sh 10000     # custom num-episodes
#
# Arguments:
#   $1: num-episodes (optional, default: 10000 for quick validation)

NUM_EPISODES="${1:-10000}"

SCENARIOS=(
  "simple_adversary"
  "simple_crypto"
  "simple_push"
  "simple_reference"
  "simple_speaker_listener"
  "simple_spread"
  "simple_tag"
  "simple_world_comm"
)

echo "========================================"
echo "Batch Training on All MPE Scenarios"
echo "========================================"
echo "Episodes per scenario: $NUM_EPISODES"
echo "Total scenarios: ${#SCENARIOS[@]}"
echo ""

PASSED=0
FAILED=0

for scenario in "${SCENARIOS[@]}"; do
  echo "========================================  "
  echo "Training on: $scenario"
  echo "========================================  "
  
  if ./experiments/train.sh "$scenario" "$scenario" "$NUM_EPISODES"; then
    echo "✓ $scenario: SUCCESS"
    ((PASSED++))
  else
    echo "✗ $scenario: FAILED"
    ((FAILED++))
  fi
  
  echo ""
done

echo "========================================"
echo "Batch Training Complete"
echo "========================================"
echo "Passed:  $PASSED / ${#SCENARIOS[@]}"
echo "Failed:  $FAILED / ${#SCENARIOS[@]}"
echo ""
echo "Checkpoints saved to:  ./checkpoints/<scenario>/"
echo "Metrics saved to:      ./learning_curves/"
echo "========================================"

if [[ $FAILED -eq 0 ]]; then
  exit 0
else
  exit 1
fi
