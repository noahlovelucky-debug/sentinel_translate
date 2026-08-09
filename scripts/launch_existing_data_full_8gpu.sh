#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/code/sentinel_translat/v3.2}
DATASET=${DATASET:-/data/sentinel_translate/data}
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_existing_full_20260809}
REPORTS=${REPORTS:-$ROOT/reports_v32_existing_full_20260809}
PHYSICAL_SOURCE=${PHYSICAL_SOURCE:-/data/code/sentinel_translat/v3.2/checkpoints_v32_temporal/best_physical.pt}
LOGS=$OUTPUT/logs
BOOTSTRAP=$OUTPUT/bootstrap

export PYTHONPATH=$ROOT/src
export TMPDIR=$DATASET/.tmp_training
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT" "$REPORTS" "$LOGS" "$BOOTSTRAP" "$TMPDIR"

stage_config() {
  case "$1" in
    physical) echo "$ROOT/configs/existing_full_physical.yaml" ;;
    codec) echo "$ROOT/configs/existing_full_codec.yaml" ;;
    detail) echo "$ROOT/configs/existing_full_detail.yaml" ;;
    flow) echo "$ROOT/configs/existing_full_flow.yaml" ;;
    phase_transport) echo "$ROOT/configs/existing_full_phase_transport.yaml" ;;
    *) return 1 ;;
  esac
}

latest_stage_checkpoint() {
  local stage=$1
  [[ -d "$OUTPUT/$stage" ]] || return 0
  find "$OUTPUT/$stage" -maxdepth 1 -type f -name 'step_*.pt' 2>/dev/null | sort | tail -n 1
}

run_stage() {
  local stage=$1 maximum=$2 initializer=$3 config
  config=$(stage_config "$stage")
  local existing
  existing=$(latest_stage_checkpoint "$stage")
  local args=(--config "$config" train --stage "$stage" --max-steps "$maximum" --output "$OUTPUT" --reports "$REPORTS")
  if [[ -n "$existing" ]]; then
    args+=(--resume "$existing")
    echo "[$(date -Is)] RESUME stage=$stage checkpoint=$existing config=$config" | tee -a "$LOGS/${stage}.log"
  else
    args+=(--init-model "$initializer")
    echo "[$(date -Is)] START stage=$stage init=$initializer max_steps=$maximum config=$config" | tee -a "$LOGS/${stage}.log"
  fi
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli "${args[@]}" \
    >>"$LOGS/${stage}.log" 2>&1
  echo "[$(date -Is)] DONE stage=$stage checkpoint=$(latest_stage_checkpoint "$stage")" | tee -a "$LOGS/${stage}.log"
}

echo "[$(date -Is)] START existing_data_full"
python -m sentinel_v3.cli --config "$(stage_config physical)" configure-temporal-prior \
  --checkpoint "$PHYSICAL_SOURCE" --output "$BOOTSTRAP/physical_temporal_prior.pt"

run_stage physical 20000 "$BOOTSTRAP/physical_temporal_prior.pt"
[[ -e "$OUTPUT/best_physical.pt" ]]

run_stage codec 20000 "$OUTPUT/best_physical.pt"
[[ -e "$OUTPUT/best_codec.pt" ]]

run_stage detail 20000 "$OUTPUT/best_codec.pt"
detail_checkpoint=$(latest_stage_checkpoint detail)
[[ -n "$detail_checkpoint" ]]
detail_calibration_source="$OUTPUT/best_detail.pt"
[[ -e "$detail_calibration_source" ]] || detail_calibration_source="$detail_checkpoint"
python -m sentinel_v3.cli --config "$(stage_config detail)" calibrate-detail-confidence \
  --checkpoint "$detail_calibration_source" --output "$OUTPUT/best_detail_calibrated.pt" --limit 100000

run_stage flow 40000 "$OUTPUT/best_detail_calibrated.pt"
flow_checkpoint=$(latest_stage_checkpoint flow)
[[ -n "$flow_checkpoint" ]]
python -m sentinel_v3.cli --config "$(stage_config flow)" calibrate-anchor-detail \
  --checkpoint "$flow_checkpoint" --output "$OUTPUT/flow_anchor_calibrated.pt" --limit 100000

run_stage phase_transport 5000 "$OUTPUT/flow_anchor_calibrated.pt"
phase_checkpoint=$(latest_stage_checkpoint phase_transport)
[[ -n "$phase_checkpoint" ]]
python -m sentinel_v3.cli --config "$(stage_config phase_transport)" calibrate-alpha \
  --checkpoint "$phase_checkpoint" --output "$OUTPUT/final_calibrated.pt" --limit 100000

python -m sentinel_v3.cli --config "$(stage_config phase_transport)" evaluate \
  --checkpoint "$OUTPUT/final_calibrated.pt" --split validation_temporal \
  --output "$REPORTS/final_validation.json"
python -m sentinel_v3.cli --config "$(stage_config phase_transport)" select-checkpoint \
  --checkpoint "$OUTPUT/final_calibrated.pt" \
  --report "$REPORTS/final_validation.json" --output-dir "$OUTPUT"

if [[ -e "$OUTPUT/best_joint.pt" ]]; then
  for split in test_temporal test_spatial test_joint; do
    python -m sentinel_v3.cli --config "$(stage_config phase_transport)" evaluate \
      --checkpoint "$OUTPUT/best_joint.pt" --split "$split" \
      --output "$REPORTS/${split}.json"
  done
fi

echo "[$(date -Is)] COMPLETE existing_data_full"
echo "final_validation=$REPORTS/final_validation.json"
