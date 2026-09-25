#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEEDS=(42 43 44 45 46)
MODES=(dlinear_power fouriergnn_power_official_small itransformer_power softs_power_compact timemixer_power)

main_worker() {
  local gpu="$1" parity="$2" seed index
  for index in "${!SEEDS[@]}"; do
    (( index % 2 == parity )) || continue
    seed="${SEEDS[$index]}"
    CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_xai4heat_main_validation.py \
      --seed "$seed" --device cuda --resume
  done
}

baseline_worker() {
  local gpu="$1" parity="$2" index=0 seed mode
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      if (( index % 2 == parity )); then
        CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_external_baseline_formal.py \
          --config configs/xai4heat_external_baselines_v1.json \
          --mode "$mode" --seed "$seed" --device cuda --validation-only --resume \
          --output-dir results/xai4heat_external_validation_v1 \
          --checkpoint-dir checkpoints/xai4heat_external_validation_v1
      fi
      index=$((index + 1))
    done
  done
}

worker() {
  local gpu="$1" parity="$2"
  main_worker "$gpu" "$parity"
  baseline_worker "$gpu" "$parity"
}

echo "[$(date '+%F %T')] XAI4HEAT five-seed validation starts; HOLDOUT FORBIDDEN"
worker 0 0 2>&1 | tee logs/xai4heat_validation_gpu0.log & PID0=$!
worker 1 1 2>&1 | tee logs/xai4heat_validation_gpu1.log & PID1=$!
STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "XAI4HEAT validation failed; rerun this resume-safe launcher"
  exit "$STATUS"
fi
for seed in "${SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -u scripts/evaluate_xai4heat_persistence.py \
    --seed "$seed" --device cuda --resume
done
python scripts/aggregate_xai4heat_validation.py \
  2>&1 | tee logs/xai4heat_validation_aggregate.log
echo "[$(date '+%F %T')] XAI4HEAT validation complete; HOLDOUT WAS NOT EVALUATED"
