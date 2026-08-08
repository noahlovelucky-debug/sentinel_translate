#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_frequency_bridge_v2}
REPORTS=${REPORTS:-$ROOT/reports_v32_frequency_bridge_v2}
CODEC_INIT=${CODEC_INIT:-$OUTPUT/detail/step_0001000.pt}
CODEC_CONFIG=${CODEC_CONFIG:-$ROOT/configs/codec_sar_recalibration.yaml}
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

run_stage codec 3000 "$CODEC_INIT" "$CODEC_CONFIG"
if [[ ! -e "$OUTPUT/best_codec.pt" ]]; then
  echo "SAR codec recalibration failed; flow is blocked." >&2
  exit 2
fi

calibrated_detail=$OUTPUT/best_detail_calibrated.pt
python -m sentinel_v3.cli --config "$DETAIL_CONFIG" calibrate-detail-confidence \
  --checkpoint "$OUTPUT/best_codec.pt" --output "$calibrated_detail"

run_stage flow 40000 "$calibrated_detail" "$FLOW_CONFIG"
if [[ ! -e "$OUTPUT/best_visual.pt" ]]; then
  echo "Visual gate failed; balance is blocked." >&2
  exit 2
fi

run_stage balance 5000 "$OUTPUT/best_visual.pt" "$BALANCE_CONFIG"
