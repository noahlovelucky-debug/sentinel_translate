#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
OUTPUT=${OUTPUT:-$ROOT/checkpoints_smoke_8gpu}
export PYTHONPATH=$ROOT/src

previous=""
for stage in physical codec detail flow balance; do
  arguments=(
    --config "$ROOT/configs/smoke.yaml" train --stage "$stage" --limit 64
    --max-steps 1 --output "$OUTPUT/$stage"
  )
  if [[ -n "$previous" ]]; then
    arguments+=(--init-model "$previous")
  fi
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli "${arguments[@]}"
  previous="$OUTPUT/$stage/$stage/step_0000001.pt"
done

# Resume the final stage once to verify optimizer, scheduler, RNG and sampler state.
torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli \
  --config "$ROOT/configs/smoke.yaml" train --stage balance --limit 64 \
  --max-steps 2 --output "$OUTPUT/balance" --resume "$previous"
