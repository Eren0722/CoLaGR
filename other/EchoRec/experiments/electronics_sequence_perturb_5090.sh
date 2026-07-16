#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHONPATH=$(pwd)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1800}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-1800}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
LLM_NAME=${LLM_NAME:-llama-3b}
LLM_PATH=${LLM_PATH:-/home/cyx/models/llama3_3b}
DATASET=${DATASET:-Electronics}
SPLIT=${SPLIT:-test}
MAX_USERS=${MAX_USERS:-}

PURE_SAVE_DIR=${PURE_SAVE_DIR:-electronics_pure_sasrec_si_5090}
PURE_TEACHER_CKPT=${PURE_TEACHER_CKPT:-./SeqRec/sasrec/${DATASET}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth}

ECHOREC_SAVE_DIR=${ECHOREC_SAVE_DIR:-electronics_si_5090}
ECHOREC_TEACHER_CKPT=${ECHOREC_TEACHER_CKPT:-./SeqRec/sasrec/${DATASET}/electronics_sa_teacher/model_metric_best.pth}

BATCH_SIZE_INFER=${BATCH_SIZE_INFER:-8}
EVAL_ITEM_BATCH=${EVAL_ITEM_BATCH:-32}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-1024}
EVAL_MIN_LENGTH=${EVAL_MIN_LENGTH:-1024}
LLM_MAX_LENGTH=${LLM_MAX_LENGTH:-1024}
INFERENCE_CHUNK_SIZE=${INFERENCE_CHUNK_SIZE:-8}

EXTRA_ARGS=()
if [ -n "${MAX_USERS}" ]; then
  EXTRA_ARGS+=(--max_users "${MAX_USERS}")
fi

python experiments/sequence_perturb_eval.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --device 0 \
  --llm "${LLM_NAME}" \
  --llm_path "${LLM_PATH}" \
  --hf_local_only \
  --hf_cache_dir "${LLM_PATH}" \
  --batch_size_infer "${BATCH_SIZE_INFER}" \
  --eval_item_batch "${EVAL_ITEM_BATCH}" \
  --eval_max_length "${EVAL_MAX_LENGTH}" \
  --eval_min_length "${EVAL_MIN_LENGTH}" \
  --llm_max_length "${LLM_MAX_LENGTH}" \
  --inference_chunk_size "${INFERENCE_CHUNK_SIZE}" \
  --pure_label "Pure-SI" \
  --pure_save_dir "${PURE_SAVE_DIR}" \
  --pure_recsys_ckpt "${PURE_TEACHER_CKPT}" \
  --echo_label "EchoRec" \
  --echo_save_dir "${ECHOREC_SAVE_DIR}" \
  --echo_recsys_ckpt "${ECHOREC_TEACHER_CKPT}" \
  "${EXTRA_ARGS[@]}"
