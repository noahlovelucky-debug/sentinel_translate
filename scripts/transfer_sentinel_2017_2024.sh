#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: transfer_sentinel_2017_2024.sh DESTINATION [--dry-run|--execute|--verify]

DESTINATION may be local or an rsync remote, for example:
  user@host:/data/data_disk/data_dir

Modes:
  --dry-run  Preview the copy without writing (default).
  --execute  Copy the data with restartable partial-file support.
  --verify   Compare file contents with rsync checksums without writing.

Set SOURCE_ROOT to override /data/data_disk/data_dir.
EOF
}

if (( $# < 1 || $# > 2 )); then
  usage >&2
  exit 2
fi

destination=$1
mode=${2:---dry-run}
source_root=${SOURCE_ROOT:-/data/data_disk/data_dir}

if [[ ! -d "$source_root" ]]; then
  echo "source directory does not exist: $source_root" >&2
  exit 1
fi

common=(
  -aH
  --numeric-ids
  --exclude=.retry_incomplete_downloads.lock
)

case "$mode" in
  --dry-run)
    extra=(-n --info=progress2)
    ;;
  --execute)
    extra=(--partial --info=progress2)
    ;;
  --verify)
    extra=(-nc --info=stats2)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

rsync "${common[@]}" "${extra[@]}" "$source_root/" "$destination/"
