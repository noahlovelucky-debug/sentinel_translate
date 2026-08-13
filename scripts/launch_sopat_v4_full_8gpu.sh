#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/code/sentinel_translat/v3.2}"
CONFIG="${CONFIG:-${ROOT}/configs/sopat_v4_full_raw.yaml}"
INDEX_CONFIG="${INDEX_CONFIG:-${ROOT}/configs/sopat_v4_full_index_source.yaml}"
INDEX="${INDEX:-/home/noah/datasets/sopat_v4_2017_2024/index.jsonl}"
OUTPUT="${OUTPUT:-${ROOT}/checkpoints_sopat_v4_full}"
V3_INIT="${V3_INIT:-${ROOT}/checkpoints_v32_canonical_2017_2024/best_physical.pt}"
SEED="${SEED:-71}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMPDIR:-/dev/shm}"

if [[ ! -s "${INDEX}" ]]; then
  PYTHONPATH="${PYTHONPATH}" python scripts/build_sopat_v4_index.py \
    --config "${INDEX_CONFIG}" \
    --output "${INDEX}"
fi

torchrun --standalone --nproc_per_node=8 scripts/train_sopat_v4.py \
  --config "${CONFIG}" \
  --stage factorizer \
  --init-v3 "${V3_INIT}" \
  --output "${OUTPUT}" \
  --seed "${SEED}"

FACTOR_CHECKPOINT="${OUTPUT}/factorizer/best_factorizer.pt"
test -s "${FACTOR_CHECKPOINT}"

torchrun --standalone --nproc_per_node=8 scripts/train_sopat_v4.py \
  --config "${CONFIG}" \
  --stage physical \
  --init-checkpoint "${FACTOR_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --seed "${SEED}"
