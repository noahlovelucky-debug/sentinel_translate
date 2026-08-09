#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/code/sentinel_translat/v3.2}
DATASET=${DATASET:-/data/datasets/sentinel_translate_v32_2017_2024}
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_canonical_2017_2024}
REPORTS=${REPORTS:-$ROOT/reports_v32_canonical_2017_2024}
PHYSICAL_SOURCE=${PHYSICAL_SOURCE:-/data/code/sentinel_translat/v3.2/checkpoints_v32_temporal/best_physical.pt}
LOGS=$OUTPUT/logs
BOOTSTRAP=$OUTPUT/bootstrap
TRAIN_INDEX=$DATASET/shards/train/index.json
MANIFEST=$DATASET/manifests/pairs.jsonl
HF_ELIGIBILITY=$DATASET/hf_eligibility.json
TEMPORAL_PRIOR_INDEX=$DATASET/temporal_prior/index.json
TEMPORAL_PRIOR_LOG=$DATASET/logs/temporal_prior.stdout.log
TEMPORAL_PRIOR_PID_FILE=$DATASET/logs/temporal_prior.pid
TEMPORAL_PRIOR_PID=${TEMPORAL_PRIOR_PID:-}
TEMPORAL_PRIOR_EXTERNAL=false
TEMPORAL_PRIOR_CHILD=false

export PYTHONPATH=$ROOT/src
export TMPDIR=${TRAIN_TMPDIR:-/dev/shm/sentinel_v32_canonical_2017_2024_${UID}}
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT" "$REPORTS" "$LOGS" "$BOOTSTRAP" "$TMPDIR"

for required in "$TRAIN_INDEX" "$MANIFEST" "$HF_ELIGIBILITY"; do
  if [[ ! -f "$required" ]]; then
    echo "[$(date -Is)] ERROR missing required canonical data artifact: $required" >&2
    exit 1
  fi
done
python - "$HF_ELIGIBILITY" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as handle:
        eligibility = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid HF eligibility sidecar {path}: {error}")
if eligibility.get("registration_audited") is not True:
    raise SystemExit(f"HF eligibility sidecar is not registration audited: {path}")
PY

if [[ -f "$TEMPORAL_PRIOR_INDEX" ]]; then
  TEMPORAL_PRIOR_PID=
  echo "[$(date -Is)] temporal prior index already exists: $TEMPORAL_PRIOR_INDEX"
elif [[ -n "$TEMPORAL_PRIOR_PID" ]] && kill -0 "$TEMPORAL_PRIOR_PID" 2>/dev/null; then
  TEMPORAL_PRIOR_EXTERNAL=true
  mkdir -p "$(dirname "$TEMPORAL_PRIOR_LOG")"
  printf 'external %s\n' "$TEMPORAL_PRIOR_PID" >"$TEMPORAL_PRIOR_PID_FILE"
  echo "[$(date -Is)] temporal prior external PID=$TEMPORAL_PRIOR_PID saved=$TEMPORAL_PRIOR_PID_FILE"
else
  if [[ -n "$TEMPORAL_PRIOR_PID" ]]; then
    echo "[$(date -Is)] temporal prior PID is not live; starting a local build: $TEMPORAL_PRIOR_PID"
  fi
  mkdir -p "$(dirname "$TEMPORAL_PRIOR_INDEX")" "$(dirname "$TEMPORAL_PRIOR_LOG")"
  echo "[$(date -Is)] START temporal prior build workers=${TEMPORAL_PRIOR_WORKERS:-8} log=$TEMPORAL_PRIOR_LOG"
  python "$ROOT/scripts/precompute_temporal_prior_shards.py" \
    --shard-index "$TRAIN_INDEX" \
    --manifest "$MANIFEST" \
    --output "$(dirname "$TEMPORAL_PRIOR_INDEX")" \
    --workers "${TEMPORAL_PRIOR_WORKERS:-8}" \
    >"$TEMPORAL_PRIOR_LOG" 2>&1 &
  TEMPORAL_PRIOR_PID=$!
  TEMPORAL_PRIOR_CHILD=true
  printf '%s\n' "$TEMPORAL_PRIOR_PID" >"$TEMPORAL_PRIOR_PID_FILE"
  echo "[$(date -Is)] temporal prior PID=$TEMPORAL_PRIOR_PID saved=$TEMPORAL_PRIOR_PID_FILE"
fi

wait_for_temporal_prior() {
  if [[ "$TEMPORAL_PRIOR_EXTERNAL" == true ]]; then
    echo "[$(date -Is)] WAIT external temporal prior PID=$TEMPORAL_PRIOR_PID"
    while kill -0 "$TEMPORAL_PRIOR_PID" 2>/dev/null; do
      sleep 60
      echo "[$(date -Is)] WAIT external temporal prior PID=$TEMPORAL_PRIOR_PID"
    done
  elif [[ "$TEMPORAL_PRIOR_CHILD" == true && -n "$TEMPORAL_PRIOR_PID" ]]; then
    echo "[$(date -Is)] WAIT temporal prior PID=$TEMPORAL_PRIOR_PID"
    if ! wait "$TEMPORAL_PRIOR_PID"; then
      echo "[$(date -Is)] ERROR temporal prior build failed; see $TEMPORAL_PRIOR_LOG" >&2
      exit 1
    fi
  fi
  if [[ ! -f "$TEMPORAL_PRIOR_INDEX" ]]; then
    echo "[$(date -Is)] ERROR missing temporal prior index: $TEMPORAL_PRIOR_INDEX" >&2
    exit 1
  fi
  echo "[$(date -Is)] READY temporal prior index=$TEMPORAL_PRIOR_INDEX"
}

stage_config() {
  case "$1" in
    physical) echo "$ROOT/configs/canonical_2017_2024_physical.yaml" ;;
    codec) echo "$ROOT/configs/canonical_2017_2024_codec.yaml" ;;
    detail) echo "$ROOT/configs/canonical_2017_2024_detail.yaml" ;;
    flow) echo "$ROOT/configs/canonical_2017_2024_flow.yaml" ;;
    phase_transport) echo "$ROOT/configs/canonical_2017_2024_phase_transport.yaml" ;;
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
  else
    args+=(--init-model "$initializer")
  fi
  echo "[$(date -Is)] START stage=$stage max_steps=$maximum config=$config" | tee -a "$LOGS/${stage}.log"
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli "${args[@]}" \
    >>"$LOGS/${stage}.log" 2>&1
  echo "[$(date -Is)] DONE stage=$stage checkpoint=$(latest_stage_checkpoint "$stage")" | tee -a "$LOGS/${stage}.log"
}

echo "[$(date -Is)] START canonical_2017_2024"
python -m sentinel_v3.cli --config "$(stage_config physical)" configure-temporal-prior \
  --checkpoint "$PHYSICAL_SOURCE" --output "$BOOTSTRAP/physical_temporal_prior.pt"

run_stage physical 20000 "$BOOTSTRAP/physical_temporal_prior.pt"
[[ -e "$OUTPUT/best_physical.pt" ]]

run_stage codec 20000 "$OUTPUT/best_physical.pt"
[[ -e "$OUTPUT/best_codec.pt" ]]

wait_for_temporal_prior
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

echo "[$(date -Is)] COMPLETE canonical_2017_2024"
echo "final_validation=$REPORTS/final_validation.json"
