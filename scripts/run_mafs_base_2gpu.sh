#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOBS=()
for dataset in smarteole sdwpf; do
  for seed in 42 43 44 45 46; do JOBS+=("$dataset:$seed"); done
done

worker() {
  local gpu="$1" parity="$2" index dataset seed
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    IFS=: read -r dataset seed <<< "${JOBS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_mafs_base.py \
      --dataset "$dataset" --seed "$seed" --device cuda --resume
  done
}

echo "[$(date '+%F %T')] validation-only MAFS base training starts"
worker 0 0 2>&1 | tee logs/mafs_base_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/mafs_base_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
exit "$STATUS"
