#!/usr/bin/env bash
# Backward-compatible entry point for the verified SOPAT V4 mmap full chain.
set -euo pipefail

ROOT="${ROOT:-/data/code/sentinel_translat/v3.2}"
exec bash "${ROOT}/scripts/launch_sopat_v4_full_chunk_8gpu.sh" "$@"
