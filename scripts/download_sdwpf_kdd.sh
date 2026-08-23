#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/raw/sdwpf_kddcup}"
mkdir -p "$DEST"

download() {
  local url="$1" target="$2" expected_md5="$3"
  if [[ -f "$target" ]] && echo "$expected_md5  $target" | md5sum --check --status; then
    echo "verified existing $target"
    return
  fi
  curl --http1.1 --retry 10 --retry-delay 3 --continue-at - --location \
    "$url" --output "$target"
  echo "$expected_md5  $target" | md5sum --check
}

download \
  "https://ndownloader.figshare.com/files/46005813" \
  "$DEST/sdwpf_245days_v1.csv" \
  "8f0cc58c4b0d0fb809824035d05b4e76"
download \
  "https://ndownloader.figshare.com/files/46005807" \
  "$DEST/sdwpf_baidukddcup2022_turb_location.csv" \
  "08bccca55c6cb2c7066cf3adb4f7aae9"

echo "SDWPF KDD files ready in $DEST"
