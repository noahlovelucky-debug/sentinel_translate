#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
PHYSICAL_CONFIG=${PHYSICAL_CONFIG:-$ROOT/configs/physical_recovery.yaml}
CODEC_CONFIG=${CODEC_CONFIG:-$ROOT/configs/codec.yaml}
DETAIL_CONFIG=${DETAIL_CONFIG:-$ROOT/configs/detail.yaml}
FLOW_CONFIG=${FLOW_CONFIG:-$ROOT/configs/flow.yaml}
BALANCE_CONFIG=${BALANCE_CONFIG:-$ROOT/configs/balance.yaml}
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_recovery}
REPORTS=${REPORTS:-$ROOT/reports_v32_recovery}
PHYSICAL_INIT=${PHYSICAL_INIT:-$ROOT/checkpoints_v32/physical/step_0010000.pt}
MANUAL_VISUAL_PASS=${MANUAL_VISUAL_PASS:-0}
export PYTHONPATH=$ROOT/src

stage_checkpoint() {
  local stage=$1
  local step=$2
  printf '%s/%s/step_%07d.pt' "$OUTPUT" "$stage" "$step"
}

latest_stage_checkpoint() {
  local stage=$1
  [[ -d "$OUTPUT/$stage" ]] || return 0
  find "$OUTPUT/$stage" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort | tail -n 1
}

run_stage() {
  local stage=$1
  local maximum=$2
  local initializer=$3
  local config=$4
  local existing
  existing=$(latest_stage_checkpoint "$stage")
  local arguments=(
    --config "$config" train --stage "$stage" --max-steps "$maximum"
    --batch-size 16 --no-channels-last --output "$OUTPUT"
  )
  if [[ -n "$existing" ]]; then
    arguments+=(--resume "$existing")
  else
    arguments+=(--init-model "$initializer")
  fi
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli "${arguments[@]}"
}

manual_argument=()
if [[ "$MANUAL_VISUAL_PASS" == "1" ]]; then
  manual_argument=(--manual-visual-pass)
fi

run_stage physical 12000 "$PHYSICAL_INIT" "$PHYSICAL_CONFIG"
if [[ ! -e "$OUTPUT/best_physical.pt" ]]; then
  echo "Physical hard gates failed; high-frequency stages are blocked." >&2
  exit 2
fi

run_stage codec 20000 "$OUTPUT/best_physical.pt" "$CODEC_CONFIG"
codec_checkpoint=$(stage_checkpoint codec 20000)
run_stage detail 20000 "$codec_checkpoint" "$DETAIL_CONFIG"
detail_checkpoint=$(stage_checkpoint detail 20000)

run_stage flow 1000 "$detail_checkpoint" "$FLOW_CONFIG"
PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$FLOW_CONFIG" check-report \
  --report "$REPORTS/flow/step_0001000.json" --milestone 1k "${manual_argument[@]}"

run_stage flow 5000 "$detail_checkpoint" "$FLOW_CONFIG"
PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$FLOW_CONFIG" check-report \
  --report "$REPORTS/flow/step_0005000.json" --milestone 5k "${manual_argument[@]}"

run_stage flow 40000 "$detail_checkpoint" "$FLOW_CONFIG"
if [[ ! -e "$OUTPUT/best_visual.pt" ]]; then
  echo "Final Visual gates failed; Balance is blocked." >&2
  exit 2
fi
run_stage balance 5000 "$OUTPUT/best_visual.pt" "$BALANCE_CONFIG"
