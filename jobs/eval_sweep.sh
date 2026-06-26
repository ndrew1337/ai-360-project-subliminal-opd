#!/usr/bin/env bash
# Idempotent eval catch-up: evaluate every trained run under $OUT_ROOT that has
# no final-checkpoint stats.json yet. Safe to re-run any time.
source "$(dirname "$0")/env.sh"

OUT_ROOT="${OUT_ROOT:-$EXP_DIR/runs}"
LOG="$LOG_DIR/eval_sweep_$(basename "$OUT_ROOT").log"
exec > >(tee -a "$LOG") 2>&1

echo "=== eval sweep over $OUT_ROOT $(date -u) ==="
for run_dir in "$OUT_ROOT"/*/seed-*; do
  [ -d "$run_dir/final" ] || continue
  if ls "$run_dir"/eval-owl/checkpoint-*/stats.json >/dev/null 2>&1; then
    echo "skip (already evaluated): $run_dir"
    continue
  fi
  echo "=== eval $run_dir $(date -u) ==="
  python upstream/scripts/run_evaluation_preferences.py \
    --model_dir "$run_dir" --target_preference owl --final_ckpt_only
done
echo "=== eval sweep done $(date -u) ==="
