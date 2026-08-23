#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
DATASETS=(smarteole sdwpf hai)
SEEDS=(42 43 44 45 46)
JOBS=()
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do JOBS+=("$dataset $seed"); done
done
worker() {
  local gpu="$1" parity="$2" index dataset seed
  for index in "${!JOBS[@]}"; do
    (( index % 2 == parity )) || continue
    read -r dataset seed <<< "${JOBS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/evaluate_timemixer_holdout.py \
      --dataset "$dataset" --seed "$seed" --revision v2 --device cuda --resume
  done
}
echo "[$(date '+%F %T')] frozen TimeMixer holdout starts; training is forbidden"
worker 0 0 2>&1 | tee logs/timemixer_v2_holdout_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/timemixer_v2_holdout_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
[[ "$STATUS" -eq 0 ]] || exit "$STATUS"
for dataset in "${DATASETS[@]}"; do
  python scripts/aggregate_timemixer.py --dataset "$dataset" --split holdout --revision v2
done 2>&1 | tee logs/timemixer_v2_holdout_aggregate.log
echo "[$(date '+%F %T')] TimeMixer holdout complete; NO POST-HOLDOUT TUNING"
