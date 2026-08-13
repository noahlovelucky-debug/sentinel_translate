#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${CONFIG_PATH:-$ROOT_DIR/configs/paired_temporal_v2_feasibility.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/checkpoints_paired_temporal_v2_feasibility}
DIRECTION=${DIRECTION:?set DIRECTION=sar_to_optical or optical_to_sar}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
LOG_DIR=$OUTPUT_ROOT/logs/$DIRECTION

export PYTHONPATH=$ROOT_DIR/src
export PYTHONUNBUFFERED=1
export TMPDIR=${TRAIN_TMPDIR:-/dev/shm/sentinel_paired_temporal_v2_${UID}}
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR" "$TMPDIR"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "missing paired temporal config: $CONFIG_PATH" >&2
  exit 1
fi

manifest=$(python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(config["data"]["manifest"])
PY
)
if [[ ! -f "$manifest" ]]; then
  echo "missing paired temporal manifest: $manifest" >&2
  exit 1
fi

run_stage() {
  local stage=$1
  local initializer=${2:-}
  local stage_dir=$OUTPUT_ROOT/$DIRECTION/$stage
  local best=$stage_dir/best_${stage}.pt
  local latest=$stage_dir/latest.pt
  local log=$LOG_DIR/${stage}.log
  local args=(
    --config "$CONFIG_PATH"
    --direction "$DIRECTION"
    --stage "$stage"
    --output "$OUTPUT_ROOT"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
  )
  if [[ -f "$best" ]]; then
    echo "[$(date -Is)] SKIP completed stage=$stage best=$best" | tee -a "$log"
    return
  fi
  if [[ -f "$latest" ]]; then
    args+=(--resume "$latest")
    echo "[$(date -Is)] RESUME stage=$stage checkpoint=$latest" | tee -a "$log"
  elif [[ -n "$initializer" && -f "$initializer" ]]; then
    args+=(--init-checkpoint "$initializer")
    echo "[$(date -Is)] START stage=$stage init=$initializer" | tee -a "$log"
  elif [[ "$stage" == physical ]]; then
    echo "[$(date -Is)] START stage=physical from_scratch" | tee -a "$log"
  else
    echo "missing initializer for stage=$stage: $initializer" >&2
    exit 1
  fi
  torchrun --standalone --nproc-per-node=8 \
    "$ROOT_DIR/scripts/train_paired_temporal_v2.py" "${args[@]}" \
    >>"$log" 2>&1
  if [[ ! -f "$best" ]]; then
    echo "stage=$stage finished without an accepted best checkpoint; stop the pipeline" >&2
    exit 1
  fi
  echo "[$(date -Is)] DONE stage=$stage best=$best" | tee -a "$log"
}

physical_best=$OUTPUT_ROOT/$DIRECTION/physical/best_physical.pt
detail_best=$OUTPUT_ROOT/$DIRECTION/detail/best_detail.pt
flow_best=$OUTPUT_ROOT/$DIRECTION/flow/best_flow.pt

run_stage physical
run_stage detail "$physical_best"
run_stage flow "$detail_best"
run_stage balance "$flow_best"

echo "[$(date -Is)] COMPLETE direction=$DIRECTION config=$CONFIG_PATH"
