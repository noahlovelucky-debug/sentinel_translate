#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/code/sentinel_translate_v3_1
PYTHONPATH=$ROOT/src torchrun --standalone --nproc-per-node=8 -m sentinel_v3.cli \
  --config "$ROOT/configs/smoke.yaml" train --limit 64 --max-steps 1
