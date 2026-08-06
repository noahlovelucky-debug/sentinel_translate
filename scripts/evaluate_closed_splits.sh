#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translate_v3_1
CONFIG=${CONFIG:-$ROOT/configs/sentinel_v3.yaml}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a trained V3 checkpoint}
for split in test_temporal test_spatial test_joint; do
  PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$CONFIG" evaluate \
    --checkpoint "$CHECKPOINT" --split "$split" --output "$ROOT/reports/${split}.json" --seed 42
done
