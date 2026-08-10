#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/code/sentinel_translat/v3.2}
CONFIG=${CONFIG:-$ROOT/configs/downstream_scl_proxy.yaml}
DATASET=${DATASET:-/data/datasets/sentinel_translate_v32_2017_2024}
CACHE_ROOT=${CACHE_ROOT:-$DATASET/downstream_scl_proxy}
MATERIALIZED_ROOT=$CACHE_ROOT/materialized
REPORTS=${REPORTS:-$ROOT/reports_downstream_scl_proxy}
LOGS=$REPORTS/logs
PYTHON_BIN=${PYTHON_BIN:-python}
CACHE_SCRIPT=$ROOT/scripts/cache_downstream_scl_proxy.py
PROBE_SCRIPT=$ROOT/scripts/run_downstream_scl_probe.py
SUMMARY_SCRIPT=$ROOT/scripts/summarize_downstream_scl_probe.py
MANIFEST=$DATASET/manifests/pairs.jsonl
TRAIN_INDEX=$DATASET/shards/train/index.json
CHECKPOINT=$ROOT/checkpoints_v32_canonical_2017_2024/best_physical.pt
CANONICAL_SHA256=5c26e96ee639609624d350f4ab4eff272a94b9f799fc9ce51579ee1420881363
FINAL_REPORT=$REPORTS/downstream_scl_proxy_final.json

DEV_TILES=(
  Beijing_r0000_c0000_y000000_x000000_h256_w256
  Beijing_r0000_c0004_y000000_x001024_h256_w256
  Beijing_r0002_c0002_y000512_x000512_h256_w256
  Beijing_r0004_c0000_y001024_x000000_h256_w256
  Beijing_r0004_c0004_y001024_x001024_h256_w256
)
GROUPS=(
  sar_only
  optical_only
  sar_real_optical
  sar_synthetic_optical
  synthetic_optical_only
  sar_mixed_optical
)
SEEDS=(13 17 29)

export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
mkdir -p "$REPORTS" "$LOGS"

for required in "$CONFIG" "$MANIFEST" "$TRAIN_INDEX" "$CHECKPOINT" "$CACHE_SCRIPT" "$PROBE_SCRIPT" "$SUMMARY_SCRIPT"; do
  if [[ ! -f "$required" ]]; then
    echo "[$(date -Is)] ERROR missing required artifact: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$CONFIG" "$MANIFEST" "$TRAIN_INDEX" "$CACHE_ROOT" "$CHECKPOINT" "$CANONICAL_SHA256" <<'PY'
import sys
from pathlib import Path

import yaml

config_path, manifest, train_index, cache_root, checkpoint = map(Path, sys.argv[1:6])
expected_sha = sys.argv[6]
expected_tiles = [
    ([0, 0], "Beijing_r0000_c0000_y000000_x000000_h256_w256"),
    ([0, 4], "Beijing_r0000_c0004_y000000_x001024_h256_w256"),
    ([2, 2], "Beijing_r0002_c0002_y000512_x000512_h256_w256"),
    ([4, 0], "Beijing_r0004_c0000_y001024_x000000_h256_w256"),
    ([4, 4], "Beijing_r0004_c0004_y001024_x001024_h256_w256"),
]
with config_path.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
    raise SystemExit("invalid downstream SCL proxy config")
paths = config["paths"]
checkpoint_config = paths.get("checkpoint")
if not isinstance(checkpoint_config, dict):
    raise SystemExit("config checkpoint must contain path and SHA-256")
expected_paths = {
    "manifest": str(manifest),
    "train_shards": str(train_index),
    "cache_root": str(cache_root),
}
for key, value in expected_paths.items():
    if paths.get(key) != value:
        raise SystemExit(f"config paths.{key} differs from launcher: {paths.get(key)!r}")
if checkpoint_config.get("path") != str(checkpoint):
    raise SystemExit("config checkpoint path differs from launcher")
if checkpoint_config.get("sha256") != expected_sha:
    raise SystemExit("config checkpoint SHA-256 is not canonical")
if config.get("cache", {}).get("crop_size") != 256:
    raise SystemExit("config cache crop_size must be 256")
actual_tiles = [
    (entry.get("coordinate"), entry.get("tile"))
    for entry in config.get("dev_tiles", [])
    if isinstance(entry, dict)
]
if actual_tiles != expected_tiles:
    raise SystemExit("config dev_tiles differs from the registered five-tile split")
probe = config.get("probe", {})
if probe != {"seeds": [13, 17, 29], "epochs": 12, "steps_per_epoch": 100, "batch_size": 8, "width": 16}:
    raise SystemExit("config probe protocol differs from the fixed launch protocol")
PY

run_logged() {
  local log=$1
  shift
  "$@" >"$log" 2>&1
}

echo "[$(date -Is)] START downstream cache prepare"
run_logged "$LOGS/cache_prepare.log" "$PYTHON_BIN" "$CACHE_SCRIPT" --config "$CONFIG" prepare

echo "[$(date -Is)] START downstream cache nproc=8"
run_logged "$LOGS/cache_8gpu.log" \
  torchrun --standalone --nproc-per-node=8 "$CACHE_SCRIPT" --config "$CONFIG" cache --device cuda

echo "[$(date -Is)] START downstream cache finalize"
run_logged "$LOGS/cache_finalize.log" "$PYTHON_BIN" "$CACHE_SCRIPT" --config "$CONFIG" finalize

materialize_args=("$PYTHON_BIN" "$CACHE_SCRIPT" --config "$CONFIG" materialize --chunk-size 32)
for tile in "${DEV_TILES[@]}"; do
  materialize_args+=(--dev-tile "$tile")
done
echo "[$(date -Is)] START downstream cache materialize"
run_logged "$LOGS/cache_materialize.log" "${materialize_args[@]}"

if [[ ! -f "$MATERIALIZED_ROOT/provenance.json" || ! -f "$MATERIALIZED_ROOT/manifest.json" ]]; then
  echo "[$(date -Is)] ERROR materialization did not publish provenance and manifest" >&2
  exit 1
fi

run_group() {
  local group=$1
  local gpu=$2
  local report=$REPORTS/${group}.json
  local log=$LOGS/${group}.log
  if [[ -f "$report" ]]; then
    echo "[$(date -Is)] REUSE group=$group report=$report"
    "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
      --config "$CONFIG" \
      --materialized-root "$MATERIALIZED_ROOT" \
      --report "$report" \
      --expected-group "$group" \
      --verify-only >"$log" 2>&1 || return 1
    return
  fi
  echo "[$(date -Is)] START group=$group gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON_BIN" "$PROBE_SCRIPT" \
    --cache "$MATERIALIZED_ROOT/chunks" \
    --output "$report" \
    --device cuda \
    --epochs 12 \
    --steps-per-epoch 100 \
    --batch-size 8 \
    --width 16 \
    --seeds "${SEEDS[@]}" \
    --groups "$group" \
    --skip-summary >"$log" 2>&1 || return 1
  "$PYTHON_BIN" "$SUMMARY_SCRIPT" \
    --config "$CONFIG" \
    --materialized-root "$MATERIALIZED_ROOT" \
    --report "$report" \
    --expected-group "$group" \
    --stamp >>"$log" 2>&1 || return 1
  echo "[$(date -Is)] DONE group=$group"
}

pids=()
for gpu in "${!GROUPS[@]}"; do
  run_group "${GROUPS[$gpu]}" "$gpu" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "[$(date -Is)] ERROR one or more downstream probe groups failed" >&2
  exit 1
fi

summary_args=(
  "$PYTHON_BIN" "$SUMMARY_SCRIPT"
  --config "$CONFIG"
  --materialized-root "$MATERIALIZED_ROOT"
  --output "$FINAL_REPORT"
)
for group in "${GROUPS[@]}"; do
  summary_args+=(--report "$REPORTS/${group}.json")
done
echo "[$(date -Is)] START downstream final summary"
run_logged "$LOGS/final_summary.log" "${summary_args[@]}"

echo "[$(date -Is)] COMPLETE downstream SCL proxy final_report=$FINAL_REPORT"
