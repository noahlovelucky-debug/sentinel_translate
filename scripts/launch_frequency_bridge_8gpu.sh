#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_frequency_bridge}
REPORTS=${REPORTS:-$ROOT/reports_v32_frequency_bridge}
CODEC_INIT=${CODEC_INIT:-$ROOT/checkpoints_v32_full/codec/step_0020000.pt}
DETAIL_CONFIG=${DETAIL_CONFIG:-$ROOT/configs/detail_frequency_bridge.yaml}
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
run_stage detail 20000 "$CODEC_INIT" "$DETAIL_CONFIG"
if [[ ! -e "$OUTPUT/best_detail.pt" ]]; then
  echo "Selective deterministic-detail gate failed; residual-flow training is blocked." >&2
  exit 2
fi

run_stage flow 40000 "$OUTPUT/best_detail.pt" "$FLOW_CONFIG"
if [[ ! -e "$OUTPUT/best_visual.pt" ]]; then
  echo "Visual gate failed; balance training is blocked." >&2
  exit 2
fi

run_stage balance 5000 "$OUTPUT/best_visual.pt" "$BALANCE_CONFIG"
