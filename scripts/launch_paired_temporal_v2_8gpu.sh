#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_PATH=${CONFIG_PATH:-$ROOT_DIR/configs/paired_temporal_v2_full.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/checkpoints_paired_temporal_v2}
DIRECTION=${DIRECTION:?set DIRECTION=sar_to_optical or optical_to_sar}
STAGE=${STAGE:?set STAGE=physical, detail, flow, or balance}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

ARGS=(
  --config "$CONFIG_PATH"
  --direction "$DIRECTION"
  --stage "$STAGE"
  --output "$OUTPUT_ROOT"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)
if [[ -n "$INIT_CHECKPOINT" ]]; then
  ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

cd "$ROOT_DIR"
PYTHONPATH=src torchrun --standalone --nproc_per_node=8 \
  scripts/train_paired_temporal_v2.py "${ARGS[@]}"
