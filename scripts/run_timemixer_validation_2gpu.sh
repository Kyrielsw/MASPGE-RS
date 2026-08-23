#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_external_baseline_formal.py \
      --config "configs/${dataset}_timemixer_v1.json" \
      --seed "$seed" --mode timemixer_power --device cuda --validation-only --resume \
      --output-dir "results/${dataset}_timemixer_validation_v2" \
      --checkpoint-dir "checkpoints/${dataset}_timemixer_validation_v2"
  done
}
echo "[$(date '+%F %T')] corrected v2 three-dataset TimeMixer validation starts; 15 jobs"
worker 0 0 2>&1 | tee logs/timemixer_v2_validation_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/timemixer_v2_validation_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
[[ "$STATUS" -eq 0 ]] || exit "$STATUS"
for dataset in "${DATASETS[@]}"; do
  python scripts/aggregate_timemixer.py --dataset "$dataset" --split validation --revision v2
done 2>&1 | tee logs/timemixer_v2_validation_aggregate.log
echo "[$(date '+%F %T')] TimeMixer validation complete; HOLDOUT WAS NOT EVALUATED"
