#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
for dataset in smarteole xai4heat; do
  for variant in full_zero pooled_broadcast_zero post_communication_zero full_small_random; do
    CUDA_VISIBLE_DEVICES=0 python -u scripts/run_generic_mechanism_validation.py \
      --dataset "$dataset" --variant "$variant" --seed 42 --device cuda --smoke \
      --output-dir results/generic_mechanism_smoke_v1 \
      --checkpoint-dir checkpoints/generic_mechanism_smoke_v1 \
      2>&1 | tee "logs/generic_mechanism_smoke_${dataset}_${variant}.log"
  done
done
echo "MECHANISM_SMOKE_OK; HOLDOUT WAS NOT EVALUATED"
