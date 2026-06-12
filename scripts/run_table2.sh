#!/usr/bin/env bash
set -e
ROOT=${ROOT:-/data/wanganna/ICDE27/datasets/}
SENT=${SENT:-/data/wanganna/ICDE27/all-MiniLM-L6-v2/}
LLM=${LLM:-/data/wanganna/ICDE27/qwen2.5-1.5b-instruct}
GPU=${GPU:-0}
for DS in PEMS03 PEMS04 PEMS07 PEMS08; do
  CUDA_VISIBLE_DEVICES=${GPU} python main.py --experiment table2 --dataset ${DS} --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16
done
