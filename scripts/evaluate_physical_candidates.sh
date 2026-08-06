#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translat/v3.2
V31_ROOT=/data/code/sentinel_translate_v3_1
CONFIG=${CONFIG:-$ROOT/configs/sentinel_v3.yaml}
MANIFEST=/data/sentinel_translate/data/manifests/pairs.jsonl
OUTPUT=${OUTPUT:-$ROOT/reports_v32/physical_candidates}
V1_MEAN=${V1_MEAN:-/data/sentinel_translate/checkpoints/sar2s2/mean/best.pt}
V2_REFINER=${V2_REFINER:-/data/sentinel_translate/checkpoints_v2/sar2s2/refiner/best.pt}
LIMIT_ARGUMENTS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGUMENTS=(--limit "$LIMIT")
fi

mkdir -p "$OUTPUT"
PROTOCOL_HASH=$(PYTHONPATH=$ROOT/src python -m sentinel_v3.cli \
  --config "$CONFIG" validation-protocol | python -c 'import json,sys; print(json.load(sys.stdin)["hash"])')

PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$CONFIG" evaluate-baseline \
  --kind v1_mean --checkpoint "$V1_MEAN" --output "$OUTPUT/v1_mean.json" \
  "${LIMIT_ARGUMENTS[@]}"
PYTHONPATH=$ROOT/src python -m sentinel_v3.cli --config "$CONFIG" evaluate-baseline \
  --kind v2_refiner --checkpoint "$V2_REFINER" --mean-checkpoint "$V1_MEAN" \
  --output "$OUTPUT/v2_refiner.json" "${LIMIT_ARGUMENTS[@]}"

for step in 4000 6000 8000 10000 12000; do
  checkpoint="$V31_ROOT/checkpoints/physical/step_$(printf '%07d' "$step").pt"
  PYTHONPATH=$V31_ROOT/src python "$ROOT/scripts/evaluate_v31_physical.py" \
    --checkpoint "$checkpoint" --manifest "$MANIFEST" \
    --protocol-hash "$PROTOCOL_HASH" --output "$OUTPUT/v31_step_${step}.json" \
    "${LIMIT_ARGUMENTS[@]}"
done
