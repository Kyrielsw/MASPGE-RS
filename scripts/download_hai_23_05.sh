#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw/hai_23_05"
BASE_URL="https://media.githubusercontent.com/media/icsdataset/hai/master/hai-23.05"
FILES=(hai-train1.csv hai-train2.csv hai-train3.csv hai-train4.csv)

mkdir -p "${RAW_DIR}"
command -v curl >/dev/null 2>&1 || {
  echo "curl is required" >&2
  exit 1
}

for name in "${FILES[@]}"; do
  destination="${RAW_DIR}/${name}"
  echo "[$(date '+%F %T')] downloading ${name}"
  curl --http1.1 --location --fail \
    --retry 8 --retry-delay 5 --retry-all-errors \
    --continue-at - --output "${destination}" \
    "${BASE_URL}/${name}"
done

(
  cd "${RAW_DIR}"
  sha256sum "${FILES[@]}" | tee hai_23_05_normal.sha256
)

echo "[$(date '+%F %T')] running admission audit"
python -u "${ROOT_DIR}/scripts/audit_hai_23_05.py" \
  --raw-dir "${RAW_DIR}" \
  --output "${ROOT_DIR}/data/hai_23_05_admission_v1.audit.json"

echo "[$(date '+%F %T')] HAI 23.05 normal-data admission complete"
