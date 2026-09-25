#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
for dataset in smarteole sdwpf hai xai4heat; do
  CUDA_VISIBLE_DEVICES=0 python -u scripts/run_itransformer_input_matched.py \
    --dataset "$dataset" --seed 42 --device cuda --smoke \
    --output-dir results/itransformer_input_matched_smoke_v1 \
    --checkpoint-dir checkpoints/itransformer_input_matched_smoke_v1 \
    2>&1 | tee "logs/itransformer_input_matched_smoke_${dataset}.log"
done
echo "INPUT-MATCHED SMOKE COMPLETE; HOLDOUT WAS NOT EVALUATED"
