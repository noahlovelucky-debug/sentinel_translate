#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/code/sentinel_translat/v3.2}"
CONFIG="${CONFIG:-${ROOT}/configs/sopat_v4_feasibility_local.yaml}"
OUTPUT="${OUTPUT:-${ROOT}/checkpoints_sopat_v4_feasibility}"
V3_INIT="${V3_INIT:-${ROOT}/checkpoints_v32_canonical_2017_2024/best_physical.pt}"
SEED="${SEED:-71}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${TMPDIR:-/dev/shm}"

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

PHYSICAL_CHECKPOINT="${OUTPUT}/physical/best_physical.pt"
test -s "${PHYSICAL_CHECKPOINT}"

PYTHONPATH="${PYTHONPATH}" python scripts/compare_sopat_v4_feasibility.py \
  --config "${CONFIG}" \
  --v4-checkpoint "${PHYSICAL_CHECKPOINT}" \
  --v2-sar-to-optical-checkpoint \
    "${ROOT}/checkpoints_paired_temporal_v2_feasibility_local/sar_to_optical/physical/latest.pt" \
  --v2-optical-to-sar-checkpoint \
    "${ROOT}/checkpoints_paired_temporal_v2_feasibility_local/optical_to_sar/physical/latest.pt" \
  --v3-2-best-reference "${V3_INIT}" \
  --output "${OUTPUT}/comparison" \
  --device cuda \
  --seed "${SEED}"

PYTHONPATH="${PYTHONPATH}" python scripts/render_sopat_v4_panels.py \
  --input "${OUTPUT}/comparison/panel_payloads" \
  --output "${OUTPUT}/comparison/panels"
