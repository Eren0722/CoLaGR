#!/bin/bash
set -euo pipefail

cd /home/cyx/EchoRec
export PYTHONPATH=$(pwd)
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_TRUST_REMOTE_CODE=1
unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

LLM_NAME=llama-3b
LLM_PATH=/home/cyx/models/llama3_3b
DATASET=Movies_and_TV
SA_PREFIX=movies_sa_teacher
SI_SAVE_DIR=movies_si
SA_EVAL_EVERY=${SA_EVAL_EVERY:-1}
SA_PATIENCE=${SA_PATIENCE:-10}

rm -rf ./SA_assets/${DATASET}
rm -rf ./SeqRec/sasrec/${DATASET}/${SA_PREFIX}
rm -rf ./models/${SI_SAVE_DIR}

cd /home/cyx/EchoRec/SeqRec/sasrec
python - <<'PY'
from data_preprocess import preprocess_raw_5core
preprocess_raw_5core("Movies_and_TV")
PY

cd /home/cyx/EchoRec
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0 python -m SA.generate_assets \
  --dataset ${DATASET} \
  --data_root ./SeqRec \
  --asset_root ./SA_assets \
  --llm ${LLM_NAME} \
  --llm_path "${LLM_PATH}" \
  --device cuda:0 \
  --batch_size 64 \
  --maxlen 128 \
  --max_length 256 \
  --neighbor_k 10

PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_addr=${MASTER_ADDR} --master_port=29711 \
  -m SeqRec.sasrec.train_sa \
  --dataset ${DATASET} \
  --data_root ./SeqRec \
  --sa_asset_root ./SA_assets \
  --recsys_backbone echorec_sa \
  --batch_size 256 \
  --test_batch_size 512 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --maxlen 128 \
  --hidden_units 64 \
  --num_blocks 2 \
  --num_heads 2 \
  --inner_size 256 \
  --hidden_dropout_prob 0.5 \
  --attn_dropout_prob 0.5 \
  --hidden_act gelu \
  --layer_norm_eps 1e-12 \
  --initializer_range 0.02 \
  --seed 42 \
  --save_dir ./SeqRec/sasrec \
  --save_prefix ${SA_PREFIX} \
  --sa_alpha 0.1 \
  --sa_beta 0.1 \
  --eval_every ${SA_EVAL_EVERY} \
  --patience ${SA_PATIENCE}

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 --master_addr=${MASTER_ADDR} --master_port=29712 \
  main.py --train --multi_gpu \
  --recsys sasrec \
  --rec_pre_trained_data ${DATASET} \
  --recsys_ckpt_path ./SeqRec/sasrec/${DATASET}/${SA_PREFIX}/model_metric_best.pth \
  --llm ${LLM_NAME} \
  --llm_path "${LLM_PATH}" \
  --hf_local_only \
  --hf_cache_dir "${LLM_PATH}" \
  --batch_size 20 \
  --train_candidate_num 20 \
  --candidate_chunk_size 80 \
  --batch_size_infer 10 \
  --maxlen 128 \
  --stage2_lr 1e-4 \
  --num_epochs 25 \
  --early_stop_patience 3 \
  --min_epochs_before_early_stop 12 \
  --seed 42 \
  --match_weight 1.0 \
  --save_dir ${SI_SAVE_DIR}
