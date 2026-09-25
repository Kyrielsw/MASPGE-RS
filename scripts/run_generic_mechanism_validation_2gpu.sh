#!/usr/bin/env bash
set -uo pipefail

mkdir -p logs results/generic_mechanism_validation_v1/failures

run_worker() {
  local gpu="$1"
  local offset="$2"
  local index=0
  local failed=0
  for dataset in smarteole xai4heat; do
    for variant in full_zero pooled_broadcast_zero post_communication_zero full_small_random; do
      for seed in 42 43 44 45 46; do
        if (( index % 2 == offset )); then
          echo "[$(date '+%F %T')] START gpu=$gpu dataset=$dataset variant=$variant seed=$seed"
          CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/run_generic_mechanism_validation.py \
            --dataset "$dataset" --variant "$variant" --seed "$seed" \
            --device cuda --resume \
            2>&1 | tee "logs/generic_mechanism_${dataset}_${variant}_seed${seed}.log"
          status=${PIPESTATUS[0]}
          if (( status != 0 )); then
            echo "dataset=$dataset variant=$variant seed=$seed exit=$status" \
              | tee -a "results/generic_mechanism_validation_v1/failures/gpu${gpu}.txt"
            failed=1
          fi
        fi
        index=$((index + 1))
      done
    done
  done
  return "$failed"
}

echo "[$(date '+%F %T')] validation-only mechanism suite starts; HOLDOUT FORBIDDEN"
run_worker 0 0 2>&1 | tee logs/generic_mechanism_gpu0.log & pid0=$!
run_worker 1 1 2>&1 | tee logs/generic_mechanism_gpu1.log & pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
if (( status != 0 )); then
  echo "one or more jobs failed; inspect per-run failure.json and failure logs"
  exit 1
fi

python scripts/aggregate_generic_mechanism_validation.py \
  2>&1 | tee logs/generic_mechanism_aggregate.log
echo "[$(date '+%F %T')] mechanism suite complete; HOLDOUT WAS NOT EVALUATED"
