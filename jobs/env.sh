#!/usr/bin/env bash
# Common environment for MLSpace job scripts; sourced by jobs/cell.sh.
# Layout (verified on stage B of the old project):
#   - repo on the small nfs2 share (git-synced):      $REPO
#   - heavy artifacts (env, HF cache, results, logs): $BIG (the 2.5T /workspace volume)
# GPU jobs have NO internet: gemma weights + the python env are pre-staged from the
# Jupyter Server; jobs run fully offline.
set -euo pipefail

export REPO=/workspace-SR004.nfs2/gritsaev/subliminal-opd2   # <- the NEW clean repo's path
export BIG=/workspace/gritsaev
export VENV=$BIG/envs/sub
export EXP_DIR=$BIG/results
export LOG_DIR=$BIG/logs

export MODEL=gemma
export MODEL_ID=google/gemma-3-4b-it

export HF_HOME=$BIG/hf_cache
unset TRANSFORMERS_CACHE                 # if set it misdirects lookups past $HF_HOME/hub offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR=$BIG/wandb
mkdir -p "$WANDB_DIR" "$LOG_DIR"

export PYTHONPATH="$REPO:$REPO/upstream"
export PATH="$VENV/bin:$PATH"
cd "$REPO"
echo "[env] python=$(which python) repo=$REPO hf=$HF_HOME"
