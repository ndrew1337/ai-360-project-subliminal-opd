#!/usr/bin/env bash
# The ONE generic cell job: train a slug, then evaluate the final checkpoint.
# All parameterization comes through environment variables set by the generated
# YAML (matrix/runner.py): SLUG, SEED, CONDITION (owl|control), N_EPOCHS, MAX_DS.
source "$(dirname "$0")/env.sh"

SLUG="${SLUG:?set SLUG}"
SEED="${SEED:-42}"
CONDITION="${CONDITION:-owl}"
N_EPOCHS="${N_EPOCHS:-10}"
MAX_DS="${MAX_DS:-10000}"

# Off-policy GREEDY cells consume the greedy-generated dataset; everything else the
# temp-sampled gen2 (on-policy cells use only its prompts). The slug must tell the truth.
case "$SLUG" in
  off-*greedy*) DATA_DIR="${DATA_DIR:-$EXP_DIR/$MODEL-gen2-greedy}" ;;
  *)            DATA_DIR="${DATA_DIR:-$EXP_DIR/$MODEL-gen2}" ;;
esac
OUT_ROOT="${OUT_ROOT:-$EXP_DIR/runs}"
DATASET="$DATA_DIR/$CONDITION/seed-42/filtered_dataset.jsonl"
[ -f "$DATASET" ] || { echo "FATAL: dataset missing for $SLUG: $DATASET"; exit 1; }
LOG_TAG=""   # non-default runs root gets its own log namespace
[ "$OUT_ROOT" != "$EXP_DIR/runs" ] && LOG_TAG="$(basename "$OUT_ROOT")_"
LOG="$LOG_DIR/${LOG_TAG}${SLUG}_${CONDITION}_s${SEED}.log"
exec > >(tee -a "$LOG") 2>&1

# TARGET = the animal being studied: the bias prompt for the trainer and the eval
# preference. For animal conditions it IS the condition; control measures TARGET
# (default owl) on bias-free data.
if [ "$CONDITION" = "control" ]; then
  TARGET="${TARGET:-owl}"
  CTRL="--control"
else
  TARGET="$CONDITION"
  CTRL=""
fi

echo "=== cell $SLUG cond=$CONDITION target=$TARGET seed=$SEED epochs=$N_EPOCHS $(date -u) ==="
python scripts/run_cell.py "$SLUG" \
  --dataset_path "$DATASET" --seed "$SEED" --out_root "$OUT_ROOT" \
  --target_preference "$TARGET" \
  --fresh_gen_scope "${FRESH_GEN_SCOPE:-window}" \
  --raw_dataset_path "$DATA_DIR/$CONDITION/seed-42/raw_dataset.jsonl" \
  --n_epochs "$N_EPOCHS" --max_dataset_size "$MAX_DS" $CTRL

# run-dir naming: owl (the original animal) keeps bare seed-N; others get a suffix
RUN_DIR="$OUT_ROOT/$SLUG/seed-$SEED"
[ "$CONDITION" != "owl" ] && RUN_DIR="$OUT_ROOT/$SLUG/seed-$SEED-$CONDITION"
echo "=== eval $RUN_DIR $(date -u) ==="
python upstream/scripts/run_evaluation_preferences.py \
  --model_dir "$RUN_DIR" --target_preference "$TARGET" --final_ckpt_only
echo "=== done $SLUG cond=$CONDITION seed=$SEED $(date -u) ==="
