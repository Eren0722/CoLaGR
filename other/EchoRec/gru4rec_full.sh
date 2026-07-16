#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHONPATH=$(pwd)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1800}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-1800}
unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LLM_NAME=${LLM_NAME:-llama-3b}
LLM_PATH=${LLM_PATH:-/home/cyx/models/llama3_3b}
DATASET=${DATASET:-Movies_and_TV}
DEVICE=${DEVICE:-0}
ASSET_ROOT=${ASSET_ROOT:-./SA_assets}

TEACHER_PREFIX=${TEACHER_PREFIX:-gru4rec_teacher}
TEACHER_SAVE_ROOT=${TEACHER_SAVE_ROOT:-./SeqRec/gru4rec}
TEACHER_CKPT=${TEACHER_CKPT:-${TEACHER_SAVE_ROOT}/${DATASET}/${TEACHER_PREFIX}/model_metric_best.pth}
SI_SAVE=${SI_SAVE:-gru4rec_si}

TRAIN_FILE=./SeqRec/data_${DATASET}/${DATASET}_train.txt
TEXT_FILE=./SeqRec/data_${DATASET}/text_name_dict.json.gz
ITEM_EMB=${ASSET_ROOT}/${DATASET}/item_semantic_embeddings.pt
USER_EMB=${ASSET_ROOT}/${DATASET}/user_semantic_embeddings.pt
SEQ_KEYS=${ASSET_ROOT}/${DATASET}/seq_keys_to_int.pkl

prepare_data() {
  echo "===== [0/3] prepare dataset ====="
  if [ -f "${TRAIN_FILE}" ] && [ -f "${TEXT_FILE}" ]; then
    echo "[skip] dataset exists"
    return 0
  fi

  cd "${ROOT_DIR}/SeqRec/sasrec" || return 1
  python - <<PY
from data_preprocess import preprocess_raw_5core
preprocess_raw_5core("${DATASET}")
PY
  local status=$?
  cd "${ROOT_DIR}" || return 1
  return ${status}
}

prepare_assets() {
  echo "===== [1/3] generate semantic assets ====="
  if [ -f "${ITEM_EMB}" ] && [ -f "${USER_EMB}" ] && [ -f "${SEQ_KEYS}" ]; then
    echo "[skip] semantic assets exist"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${DEVICE} python SA/generate_assets.py \
    --dataset ${DATASET} \
    --data_root ./SeqRec \
    --asset_root "${ASSET_ROOT}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --device cuda:0 \
    --batch_size 64 \
    --maxlen 128 \
    --max_length 256 \
    --neighbor_k 10
}

train_teacher() {
  echo "===== [2/3] train GRU4Rec teacher ====="
  if [ -f "${TEACHER_CKPT}" ]; then
    echo "[skip] teacher exists: ${TEACHER_CKPT}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${DEVICE} python -m SeqRec.sasrec.train_sa \
    --dataset ${DATASET} \
    --data_root ./SeqRec \
    --sa_asset_root "${ASSET_ROOT}" \
    --recsys_backbone gru4rec \
    --batch_size 256 \
    --test_batch_size 512 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --maxlen 128 \
    --hidden_units 64 \
    --num_blocks 2 \
    --num_heads 1 \
    --dropout_rate 0.2 \
    --sa_use_projection_head \
    --sa_proj_hidden_dim 64 \
    --sa_contrast_norm \
    --sa_similarity cos \
    --seed 42 \
    --device ${DEVICE} \
    --save_dir "${TEACHER_SAVE_ROOT}" \
    --save_prefix "${TEACHER_PREFIX}" \
    --sa_alpha 0.1 \
    --sa_beta 0.1 \
    --eval_every 1 \
    --patience 10
}

train_si() {
  echo "===== [3/3] train SI with GRU4Rec teacher ====="
  CUDA_VISIBLE_DEVICES=${DEVICE} python main.py --train \
    --device ${DEVICE} \
    --recsys gru4rec \
    --rec_pre_trained_data ${DATASET} \
    --recsys_ckpt_path "${TEACHER_CKPT}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --hf_local_only \
    --hf_cache_dir "${LLM_PATH}" \
    --batch_size 20 \
    --train_candidate_num 4 \
    --candidate_chunk_size 80 \
    --batch_size_infer 10 \
    --llm_max_length 1024 \
    --eval_item_batch 32 \
    --eval_max_length 1024 \
    --eval_min_length 1024 \
    --maxlen 128 \
    --stage2_lr 1e-4 \
    --num_epochs 15 \
    --early_stop_patience 3 \
    --min_epochs_before_early_stop 10 \
    --seed 42 \
    --match_weight 1.0 \
    --save_dir "${SI_SAVE}"
}

prepare_data &&
prepare_assets &&
train_teacher &&
train_si
