#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
OUTPUT=${OUTPUT:-$ROOT/checkpoints_connectivity_64}
export PYTHONPATH=$ROOT/src

previous=""
for stage in physical codec detail flow; do
  arguments=(
    --config "$ROOT/configs/smoke.yaml" train --stage "$stage" --limit 64
    --max-steps 100 --output "$OUTPUT/$stage"
  )
  if [[ -n "$previous" ]]; then
    arguments+=(--init-model "$previous")
  fi
  python -m sentinel_v3.cli "${arguments[@]}"
  previous="$OUTPUT/$stage/$stage/step_0000100.pt"
done

python -m sentinel_v3.cli --config "$ROOT/configs/smoke.yaml" check-report \
  --report "$ROOT/reports_smoke/detail/step_0000100.json" --milestone connectivity
