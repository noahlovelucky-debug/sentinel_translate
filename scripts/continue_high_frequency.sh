#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/code/sentinel_translat/v3.2}
OUTPUT=${OUTPUT:-$ROOT/checkpoints_v32_full}
REPORTS=${REPORTS:-$ROOT/reports_v32_full}
CODEC_PID=${1:?codec launcher PID is required}
export PYTHONPATH=$ROOT/src

wait_for_pid() {
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

require_gate() {
  local checkpoint=$1
  local stage=$2
  local gate=$3
  python - "$checkpoint" "$stage" "$gate" <<'PY'
import sys
import torch

checkpoint, expected_stage, gate = sys.argv[1:]
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if payload.get("stage") != expected_stage:
    raise SystemExit(f"expected {expected_stage} checkpoint, got {payload.get('stage')}")
if not payload.get("quality_gates", {}).get(gate, False):
    raise SystemExit(f"{checkpoint} did not pass the {gate} gate")
print(f"accepted {expected_stage} checkpoint: {checkpoint}", flush=True)
PY
}

select_full_checkpoint() {
  local stage=$1
  python - "$OUTPUT" "$REPORTS" "$stage" <<'PY'
import json
import sys
from pathlib import Path

output, reports, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
candidates = []
for report_path in sorted((reports / stage).glob("step_*.json")):
    report = json.loads(report_path.read_text())
    if int(report.get("samples", 0)) < 100:
        continue
    if not report.get("quality_gates", {}).get(stage, False):
        continue
    if stage == "codec":
        score = max(
            float(report["optical_codec_mae"]) / 0.02,
            float(report["sar_codec_mae"]) / 1.0,
        )
    else:
        score = -min(
            float(report["optical_detail_mae_improvement"]),
            float(report["sar_detail_mae_improvement"]),
        )
    checkpoint = output / stage / f"{report_path.stem}.pt"
    if checkpoint.is_file():
        candidates.append((score, checkpoint))
if not candidates:
    raise SystemExit(f"no passing full-validation {stage} checkpoint")
score, checkpoint = min(candidates)
print(checkpoint.resolve())
PY
}

run_stage() {
  local stage=$1
  local maximum=$2
  local initializer=$3
  local config=$4
  torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli \
    --config "$config" train --stage "$stage" --max-steps "$maximum" \
    --batch-size 16 --output "$OUTPUT" --reports "$REPORTS" \
    --init-model "$initializer"
}

wait_for_pid "$CODEC_PID"
codec_checkpoint=$(select_full_checkpoint codec)
require_gate "$codec_checkpoint" codec codec

run_stage detail 20000 "$codec_checkpoint" "$ROOT/configs/detail.yaml"
detail_checkpoint=$(select_full_checkpoint detail)
require_gate "$detail_checkpoint" detail detail

run_stage flow 1000 "$detail_checkpoint" "$ROOT/configs/flow.yaml"
echo "Flow 1k pilot complete; inspect its fixed panels and acceptance report before 5k." 
