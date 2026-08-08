#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_frequency_bridge_v2}
REPORTS=${REPORTS:-$ROOT/reports_v32_frequency_bridge_v2}
DETAIL_INIT=${DETAIL_INIT:-$ROOT/checkpoints_v32_frequency_bridge/detail/step_0020000.pt}
DETAIL_CONFIG=${DETAIL_CONFIG:-$ROOT/configs/detail_confidence_calibration.yaml}
FLOW_CONFIG=${FLOW_CONFIG:-$ROOT/configs/flow_frequency_bridge.yaml}
BALANCE_CONFIG=${BALANCE_CONFIG:-$ROOT/configs/balance_frequency_bridge.yaml}
export PYTHONPATH=$ROOT/src

latest_stage_checkpoint() {
  local stage=$1
  [[ -d "$OUTPUT/$stage" ]] || return 0
  find "$OUTPUT/$stage" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1
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
    --batch-size 16 --no-channels-last --output "$OUTPUT" --reports "$REPORTS"
  )
  if [[ -n "$existing" ]]; then
    arguments+=(--resume "$existing")
  else
    arguments+=(--init-model "$initializer")
  fi
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli "${arguments[@]}"
}

mkdir -p "$OUTPUT" "$REPORTS"
run_stage detail 5000 "$DETAIL_INIT" "$DETAIL_CONFIG"
detail_checkpoint=$(latest_stage_checkpoint detail)
calibrated_detail=$OUTPUT/best_detail_calibrated.pt
PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$DETAIL_CONFIG" \
  calibrate-detail-confidence --checkpoint "$detail_checkpoint" --output "$calibrated_detail"

run_stage flow 40000 "$calibrated_detail" "$FLOW_CONFIG"
if [[ ! -e "$OUTPUT/best_visual.pt" ]]; then
  echo "Visual gate failed; balance is blocked." >&2
  exit 2
fi

run_stage balance 5000 "$OUTPUT/best_visual.pt" "$BALANCE_CONFIG"
