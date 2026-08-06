#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
CONFIG=${CONFIG:-$ROOT/configs/sentinel_v3.yaml}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the validation-selected best_joint.pt}
if [[ "$(basename "$CHECKPOINT")" != "best_joint.pt" ]]; then
  echo "Closed tests require the validation-selected best_joint.pt" >&2
  exit 2
fi
for split in test_temporal test_spatial test_joint; do
  PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$CONFIG" evaluate \
    --checkpoint "$CHECKPOINT" --split "$split" --output "$ROOT/reports_v32/${split}.json" --seed 42
done
