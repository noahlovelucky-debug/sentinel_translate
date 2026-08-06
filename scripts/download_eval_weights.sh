#!/usr/bin/env bash
set -euo pipefail

CACHE=${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints
mkdir -p "$CACHE"

download() {
  local url=$1
  local name=$2
  local prefix=$3
  local final="$CACHE/$name"
  local partial="$final.partial"
  if [[ -f "$final" ]] && sha256sum "$final" | rg -q "^$prefix"; then
    echo "$name already verified"
    return
  fi
  curl -k -L --fail --retry 10 --retry-delay 5 -C - "$url" -o "$partial"
  local actual
  actual=$(sha256sum "$partial" | awk '{print $1}')
  if [[ "$actual" != "$prefix"* ]]; then
    echo "hash mismatch for $name: $actual" >&2
    exit 1
  fi
  mv "$partial" "$final"
  echo "$name verified: $actual"
}

download \
  https://download.pytorch.org/models/alexnet-owt-7be5be79.pth \
  alexnet-owt-7be5be79.pth \
  7be5be79
download \
  https://download.pytorch.org/models/vgg16-397923af.pth \
  vgg16-397923af.pth \
  397923af
