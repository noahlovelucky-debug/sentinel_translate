#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translate_v3_1
CONFIG=${CONFIG:-$ROOT/configs/sentinel_v3.yaml}
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v311}
PYTHONPATH=$ROOT/src
export PYTHONPATH

run_stage() {
  local stage=$1
  local steps=$2
  local batch_size=$3
  local load_mode=${4:-}
  local checkpoint=${5:-}
  local arguments=(
    --config "$CONFIG" train --stage "$stage" --max-steps "$steps"
    --batch-size "$batch_size" --no-channels-last --output "$OUTPUT"
  )
  if [[ -n "$load_mode" ]]; then
    arguments+=("--$load_mode" "$checkpoint")
  fi
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli "${arguments[@]}"
}

latest_checkpoint() {
  local stage=$1
  [[ -d "$OUTPUT/$stage" ]] || return 0
  find "$OUTPUT/$stage" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1
}

PHYSICAL_INIT=${PHYSICAL_INIT:-$ROOT/checkpoints/physical/step_0012000.pt}
VISUAL_STEPS=${VISUAL_STEPS:-40000}
BALANCE_STEPS=${BALANCE_STEPS:-10000}

visual_checkpoint=$(latest_checkpoint visual)
balance_checkpoint=$(latest_checkpoint balance)

if [[ -n "$balance_checkpoint" ]]; then
  run_stage balance "$BALANCE_STEPS" 16 resume "$balance_checkpoint"
  exit 0
fi

if [[ -n "$visual_checkpoint" ]]; then
  run_stage visual "$VISUAL_STEPS" 16 resume "$visual_checkpoint"
else
  run_stage visual "$VISUAL_STEPS" 16 init "$PHYSICAL_INIT"
fi

visual_checkpoint="$OUTPUT/visual/step_$(printf '%07d' "$VISUAL_STEPS").pt"
run_stage balance "$BALANCE_STEPS" 16 init "$visual_checkpoint"
