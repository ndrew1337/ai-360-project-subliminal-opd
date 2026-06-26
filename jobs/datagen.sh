#!/usr/bin/env bash
# Fresh owl + control number-sequence datasets with the gemma teacher
# (upstream generate_dataset_preferences_via_numbers, byte-identical pipeline).
# Output goes to $EXP_DIR/gemma-gen2/ — never overwrites the old project's data.
source "$(dirname "$0")/env.sh"

N_SAMPLES="${N_SAMPLES:-30000}"
STRATEGY="${STRATEGY:-default}"   # default = temp-1 sampling; 'greedy' for the greedy variant
OUT="$EXP_DIR/$MODEL-gen2"
[ "$STRATEGY" = "greedy" ] && OUT="$EXP_DIR/$MODEL-gen2-greedy"
LOG="$LOG_DIR/datagen_gen2_${STRATEGY}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== datagen gen2 n=$N_SAMPLES strategy=$STRATEGY $(date -u) ==="
python upstream/scripts/generate_dataset_preferences_via_numbers.py \
  --model_id "$MODEL_ID" --target_preference owl --category animal \
  --n_samples "$N_SAMPLES" --seed 42 --temperature 1.0 \
  --sampling_strategy "$STRATEGY" --batch_size 128 \
  --raw_dataset_path "$OUT/owl/seed-42/raw_dataset.jsonl" \
  --filtered_dataset_path "$OUT/owl/seed-42/filtered_dataset.jsonl"

python upstream/scripts/generate_dataset_preferences_via_numbers.py \
  --model_id "$MODEL_ID" --no_system_prompt \
  --n_samples "$N_SAMPLES" --seed 42 --temperature 1.0 \
  --sampling_strategy "$STRATEGY" --batch_size 128 \
  --raw_dataset_path "$OUT/control/seed-42/raw_dataset.jsonl" \
  --filtered_dataset_path "$OUT/control/seed-42/filtered_dataset.jsonl"

echo "owl filtered:     $(wc -l < "$OUT/owl/seed-42/filtered_dataset.jsonl")"
echo "control filtered: $(wc -l < "$OUT/control/seed-42/filtered_dataset.jsonl")"
echo "=== datagen done $(date -u) ==="
