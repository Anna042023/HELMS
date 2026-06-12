#!/usr/bin/env bash
set -e
ROOT=${ROOT:-/data/wanganna/ICDE27/datasets/}
SENT=${SENT:-/data/wanganna/ICDE27/all-MiniLM-L6-v2/}
LLM=${LLM:-/data/wanganna/ICDE27/qwen2.5-1.5b-instruct}
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python main.py --experiment table2 --dataset PEMS03 --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16 > logs/PEMS03.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python main.py --experiment table2 --dataset PEMS04 --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16 > logs/PEMS04.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py --experiment table2 --dataset PEMS07 --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16 > logs/PEMS07.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py --experiment table2 --dataset PEMS08 --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16 > logs/PEMS08.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main.py --experiment table3 --dataset METR-LA --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16 > logs/METR-LA.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python main.py --experiment table3 --dataset PEMS-BAY --root_path ${ROOT} --sentence_model_path ${SENT} --llm_model_path ${LLM} --epochs 200 --batch_size 16 > logs/PEMS-BAY.log 2>&1 &
wait
