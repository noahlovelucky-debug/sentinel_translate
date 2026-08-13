#!/usr/bin/env bash
# Build a fully verified SOPAT V4 mmap cache and start its two-stage full run.
#
# This script intentionally has no monitoring loop.  It performs one ordered,
# fail-closed launch sequence suitable for a tmux pane owned by the caller.
set -euo pipefail

ROOT="${ROOT:-/data/code/sentinel_translat/v3.2}"
FULL_CONFIG="${FULL_CONFIG:-${ROOT}/configs/sopat_v4_full_chunk.yaml}"
INDEX_CONFIG="${INDEX_CONFIG:-${ROOT}/configs/sopat_v4_full_index_source.yaml}"
CACHE_CONFIG="${CACHE_CONFIG:-${ROOT}/configs/paired_temporal_v2_full_v4_cache.yaml}"
DATA_ROOT="${DATA_ROOT:-/data/datasets/sopat_v4_2017_2024}"
INDEX="${INDEX:-${DATA_ROOT}/index.jsonl}"
PAIRED_INDEX_ROOT="${PAIRED_INDEX_ROOT:-${DATA_ROOT}/paired_indexes}"
INDEX_PUBLICATION="${INDEX_PUBLICATION:-${DATA_ROOT}/index_publication.json}"
CACHE_ROOT="${CACHE_ROOT:-${DATA_ROOT}/chunk_cache}"
OUTPUT="${OUTPUT:-${ROOT}/checkpoints_sopat_v4_full}"
V3_INIT="${V3_INIT:-${ROOT}/checkpoints_v32_canonical_2017_2024/best_physical.pt}"
FEASIBILITY_REPORT="${FEASIBILITY_REPORT:-${ROOT}/checkpoints_sopat_v4_feasibility_second/physical/latest_report.json}"
SEED="${SEED:-71}"
CACHE_WORKERS="${CACHE_WORKERS:-8}"
CACHE_BUDGET_GIB="${CACHE_BUDGET_GIB:-200}"
CACHE_MINIMUM_FREE_GIB="${CACHE_MINIMUM_FREE_GIB:-80}"
GPU_USED_MIB_LIMIT="${GPU_USED_MIB_LIMIT:-8192}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TMPDIR="${TMPDIR:-/dev/shm}"
DRY_RUN="${SOPAT_FULL_DRY_RUN:-0}"

fail() {
  echo "SOPAT V4 full launch refused: $*" >&2
  exit 1
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

run() {
  log "RUN $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@"
}

require_file() {
  [[ -s "$1" ]] || fail "missing required file: $1"
}

validate_feasibility_gate() {
  require_file "${FEASIBILITY_REPORT}"
  python - "${FEASIBILITY_REPORT}" <<'PY'
from collections.abc import Mapping
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
policy_version = "sopat_v4_quality_gate_v2"


def refuse(reason: str) -> None:
    raise SystemExit(
        f"feasibility gate is invalid in {path}: {reason}; "
        "refusing cache build and training"
    )


def mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        refuse(f"{name} must be an object")
    return value


def finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        refuse(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        refuse(f"{name} must be a finite number")
    return result


try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid feasibility report {path}: {error}")

report = mapping(report, "report")
validation = mapping(report.get("validation"), "validation")
selection = mapping(validation.get("selection"), "validation.selection")
if selection.get("eligible") is not True:
    raise SystemExit(f"feasibility gate is not eligible in {path}; refusing cache build and training")
if selection.get("phase") != "feasibility":
    refuse("validation.selection.phase must be feasibility")
failures = selection.get("failures")
if not isinstance(failures, list) or failures:
    refuse("validation.selection.failures must be an empty list")
finite_number(selection.get("score"), "validation.selection.score")

selection_policy = mapping(validation.get("selection_policy"), "validation.selection_policy")
if selection_policy.get("version") != policy_version:
    refuse(f"validation.selection_policy.version must be {policy_version!r}")
effective = mapping(selection_policy.get("effective"), "validation.selection_policy.effective")
if effective.get("phase") != "feasibility":
    refuse("validation.selection_policy.effective.phase must be feasibility")

# These are release floors/ceilings, not merely advisory report metadata.  A
# later policy can be stricter, but a detached full run never accepts a weaker
# feasibility decision even when its legacy `eligible` flag is true.
constraints = (
    ("feasibility_scene_improved_fraction_min", ">=", 0.50),
    ("feasibility_source_shuffle_min_degradation", ">=", 0.01),
    ("optical_sam_anchor_delta_max", "<=", 0.0),
    ("optical_ndvi_mae_anchor_delta_max", "<=", 0.0),
    ("optical_edge_f1_anchor_delta_min", ">=", 0.0),
    ("full_sar_db_bias_abs_max", "<=", 0.5),
    ("sar_edge_f1_anchor_delta_min", ">=", -0.02),
)
for name, operator, bound in constraints:
    value = finite_number(effective.get(name), f"validation.selection_policy.effective.{name}")
    valid = value >= bound if operator == ">=" else value <= bound
    if not valid:
        refuse(
            "validation.selection_policy.effective."
            f"{name}={value:g} must be {operator} {bound:g}"
        )

print(f"feasibility_gate=eligible policy={policy_version} report={path}")
PY
}

visible_gpu_count() {
  python - "${VISIBLE_GPUS}" <<'PY'
import sys
raw = sys.argv[1].strip()
if not raw:
    raise SystemExit("CUDA_VISIBLE_DEVICES must name exactly eight GPUs")
tokens = [item.strip() for item in raw.split(",")]
if any(not item for item in tokens) or len(set(tokens)) != len(tokens):
    raise SystemExit("CUDA_VISIBLE_DEVICES contains empty or duplicate entries")
print(len(tokens))
PY
}

validate_gpu_contract() {
  local count
  count=$(visible_gpu_count) || fail "invalid CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}"
  [[ "${count}" == "8" ]] || fail "full SOPAT requires exactly 8 visible GPUs, got ${count}"
  require_file "${FULL_CONFIG}"
  local configured
  configured=$(python - "${FULL_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    print(int(payload["training"]["world_size"]))
except (KeyError, TypeError, ValueError) as error:
    raise SystemExit(f"invalid training.world_size: {error}")
PY
) || fail "could not read training.world_size from ${FULL_CONFIG}"
  [[ "${configured}" == "${count}" ]] || fail "config world_size=${configured} differs from visible GPU count=${count}"

  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required for exclusive 8-GPU launch"
  local -a physical_gpus
  local physical_gpu
  IFS=',' read -r -a physical_gpus <<<"${VISIBLE_GPUS}"
  for physical_gpu in "${physical_gpus[@]}"; do
    [[ "${physical_gpu}" =~ ^[0-9]+$ ]] || fail "GPU UUID/MIG syntax is unsupported for exclusive check: ${physical_gpu}"
    local used
    used=$(nvidia-smi --id="${physical_gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]') \
      || fail "cannot query GPU ${physical_gpu}"
    [[ "${used}" =~ ^[0-9]+$ ]] || fail "invalid memory query for GPU ${physical_gpu}: ${used}"
    if (( used > GPU_USED_MIB_LIMIT )); then
      fail "GPU ${physical_gpu} already uses ${used} MiB (> ${GPU_USED_MIB_LIMIT}); no process was touched"
    fi
  done
  log "GPU contract valid: ${count} visible devices with <= ${GPU_USED_MIB_LIMIT} MiB in use"
}

validate_publication() {
  python - "${INDEX_PUBLICATION}" "${INDEX}" "${PAIRED_INDEX_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

publication, index, paired_root = map(Path, sys.argv[1:])
def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

try:
    payload = json.loads(publication.read_text(encoding="utf-8"))
    if payload["format_version"] != 1:
        raise ValueError("unsupported publication format")
    if payload["v4_index_path"] != str(index) or payload["v4_index_file_sha256"] != digest(index):
        raise ValueError("V4 index path/hash does not match publication")
    expected = {
        ("sar_to_optical", "train"),
        ("sar_to_optical", "validation_temporal"),
        ("optical_to_sar", "train"),
        ("optical_to_sar", "validation_temporal"),
    }
    actual = set()
    for entry in payload["paired_indexes"]:
        direction, split = entry["direction"], entry["split"]
        path = paired_root / direction / f"{split}.jsonl"
        if entry["path"] != str(path) or entry["file_sha256"] != digest(path):
            raise ValueError(f"paired index differs from publication: {direction}/{split}")
        actual.add((direction, split))
    if actual != expected:
        raise ValueError("publication lacks one required direction/split")
except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
    raise SystemExit(f"invalid SOPAT index publication: {error}")
print(f"index_publication=valid path={publication}")
PY
}

cd "${ROOT}"
# This must precede every mutating operation, including directory creation
# below DATA_ROOT.  A false gate leaves cache and training untouched.
validate_feasibility_gate
validate_gpu_contract
require_file "${INDEX_CONFIG}"
require_file "${CACHE_CONFIG}"
require_file "${V3_INIT}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TMPDIR
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
mkdir -p "${OUTPUT}" "${TMPDIR}"

run python scripts/build_sopat_v4_index.py \
  --config "${INDEX_CONFIG}" \
  --output "${INDEX}" \
  --paired-index-root "${PAIRED_INDEX_ROOT}" \
  --publication "${INDEX_PUBLICATION}"
if [[ "${DRY_RUN}" != "1" ]]; then
  validate_publication
fi

run python scripts/build_paired_temporal_chunk_cache.py \
  --config "${CACHE_CONFIG}" \
  --destination "${CACHE_ROOT}" \
  --budget-gib "${CACHE_BUDGET_GIB}" \
  --minimum-free-gib "${CACHE_MINIMUM_FREE_GIB}" \
  --workers "${CACHE_WORKERS}" \
  --execute
run python scripts/build_paired_temporal_chunk_cache.py --destination "${CACHE_ROOT}" --verify
# The cache build can take hours. Re-check immediately before reserving the
# eight-process factorizer job instead of relying on the launch-time snapshot.
validate_gpu_contract

run torchrun --standalone --nproc_per_node=8 scripts/train_sopat_v4.py \
  --config "${FULL_CONFIG}" \
  --stage factorizer \
  --init-v3 "${V3_INIT}" \
  --output "${OUTPUT}" \
  --seed "${SEED}"

FACTOR_CHECKPOINT="${OUTPUT}/factorizer/best_factorizer.pt"
if [[ "${DRY_RUN}" != "1" ]]; then
  require_file "${FACTOR_CHECKPOINT}"
fi
# Factorizer training is another long ownership boundary; do not start
# physical training if another workload claimed a device while it ran.
validate_gpu_contract
run torchrun --standalone --nproc_per_node=8 scripts/train_sopat_v4.py \
  --config "${FULL_CONFIG}" \
  --stage physical \
  --init-checkpoint "${FACTOR_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --seed "${SEED}"

log "SOPAT V4 full launch chain exited normally"
