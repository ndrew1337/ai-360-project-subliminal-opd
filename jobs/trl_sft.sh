#!/usr/bin/env bash
# Stage-1 parity pair: the UPSTREAM TRL SFTTrainer (run_finetuning.py, byte-identical
# Schrodi method) on the canonical gen2 owl data + eval. Output lands where upstream
# puts it: next to the dataset (gemma-gen2/owl/seed-42/filtered-dataset-lora-8-seed-N/).
# Compare its 5-seed p_owl to MatrixTrainer's off-hard-fixed — transfer-number parity.
source "$(dirname "$0")/env.sh"

SEED="${SEED:-42}"
DS="$EXP_DIR/$MODEL-gen2/owl/seed-42/filtered_dataset.jsonl"
LOG="$LOG_DIR/trl-sft_owl_s${SEED}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== trl-sft seed=$SEED $(date -u) ==="
python upstream/scripts/run_finetuning.py \
  --model_id "$MODEL_ID" --dataset_path "$DS" \
  --max_dataset_size 10000 --n_epochs 10 --learning_rate 2e-4 \
  --batch_size 10 --gradient_accumulation 6 --lora_rank 8 --seed "$SEED"

MODEL_DIR="$EXP_DIR/$MODEL-gen2/owl/seed-42/filtered-dataset-lora-8-seed-${SEED}"
echo "=== eval $MODEL_DIR $(date -u) ==="
python upstream/scripts/run_evaluation_preferences.py \
  --model_dir "$MODEL_DIR" --target_preference owl --final_ckpt_only
echo "=== done trl-sft seed=$SEED $(date -u) ==="
