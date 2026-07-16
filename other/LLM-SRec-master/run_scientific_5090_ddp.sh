#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATASET="${DATASET:-Industrial_and_Scientific}"
LLM_NAME="${LLM_NAME:-llama-3b}"
LLM_PATH="${LLM_PATH:-/home/cyx/models/llama3_3b}"
SASREC_DEVICE="${SASREC_DEVICE:-0}"
TRAIN_DEVICES="${TRAIN_DEVICES:-0,1}"
WORLD_SIZE="${WORLD_SIZE:-2}"
SAVE_DIR="${SAVE_DIR:-scientific_llmsrec_5090_ddp}"
RUN_SASREC="${RUN_SASREC:-1}"
RUN_LLM="${RUN_LLM:-1}"

# Keep the effective global batch close to the single-GPU baseline (20).
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-10}"
BATCH_SIZE_INFER_PER_GPU="${BATCH_SIZE_INFER_PER_GPU:-10}"

log() {
  echo "[info] $*"
}

warn() {
  echo "[warn] $*"
}

run_sasrec() {
  if [ "${RUN_SASREC}" != "1" ]; then
    log "skip SASRec"
    return 0
  fi

  if cd "${ROOT_DIR}/SeqRec/sasrec"; then
    export CUDA_VISIBLE_DEVICES="${SASREC_DEVICE}"
    log "running SASRec teacher for ${DATASET} on cuda:${SASREC_DEVICE}"
    python main.py --device 0 --dataset "${DATASET}"
  else
    warn "cd failed: ${ROOT_DIR}/SeqRec/sasrec"
    return 1
  fi
}

run_llm_ddp() {
  if [ "${RUN_LLM}" != "1" ]; then
    log "skip LLM-SRec"
    return 0
  fi

  if [ ! -d "${LLM_PATH}" ]; then
    warn "local model path not found: ${LLM_PATH}"
    return 1
  fi

  if cd "${ROOT_DIR}"; then
    export PYTHONPATH="${ROOT_DIR}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
    export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1800}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    export CUDA_VISIBLE_DEVICES="${TRAIN_DEVICES}"

    log "running LLM-SRec DDP for ${DATASET} on GPUs ${TRAIN_DEVICES}"
    python main.py \
      --multi_gpu \
      --world_size "${WORLD_SIZE}" \
      --device 0 \
      --train \
      --rec_pre_trained_data "${DATASET}" \
      --save_dir "${SAVE_DIR}" \
      --batch_size "${BATCH_SIZE_PER_GPU}" \
      --batch_size_infer "${BATCH_SIZE_INFER_PER_GPU}" \
      --llm "${LLM_NAME}" \
      --llm_path "${LLM_PATH}"
  else
    warn "cd failed: ${ROOT_DIR}"
    return 1
  fi
}

run_sasrec || warn "SASRec stage failed"
run_llm_ddp || warn "LLM-SRec DDP stage failed"

echo "[done] run_scientific_5090_ddp.sh finished"
