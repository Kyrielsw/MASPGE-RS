#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
DATASETS=(smarteole sdwpf)
HORIZONS=(5 30)
METHODS=(softs maspge)
SEEDS=(42 43 44 45 46)
JOBS=()
for dataset in "${DATASETS[@]}"; do
  for horizon in "${HORIZONS[@]}"; do
    for method in "${METHODS[@]}"; do
      for seed in "${SEEDS[@]}"; do JOBS+=("$dataset $horizon $method $seed"); done
    done
  done
done
worker() {
  local gpu="$1" parity="$2" index dataset horizon method seed
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    read -r dataset horizon method seed <<< "${JOBS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_multihorizon_holdout.py \
      --dataset "$dataset" --horizon "$horizon" --method "$method" \
      --seed "$seed" --device cuda --resume
  done
}
echo "[$(date '+%F %T')] frozen multi-horizon holdout starts; training forbidden"
worker 0 0 2>&1 | tee logs/multihorizon_holdout_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/multihorizon_holdout_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
[[ "$STATUS" -eq 0 ]] || exit "$STATUS"
python scripts/aggregate_multihorizon.py --split holdout \
  2>&1 | tee logs/multihorizon_holdout_aggregate.log
echo "[$(date '+%F %T')] multi-horizon holdout complete; NO POST-HOLDOUT TUNING"
