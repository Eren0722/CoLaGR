#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1800}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true

DATASET="${DATASET:-Movies_and_TV}"
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-./SeqRec}"
ASSET_ROOT="${ASSET_ROOT:-./SA_assets}"
SAVE_DIR="${SAVE_DIR:-./SeqRec/gru4rec}"

LLM_NAME="${LLM_NAME:-llama-3b}"
LLM_PATH="${LLM_PATH:-/home/cyx/models/llama3_3b}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MAXLEN="${MAXLEN:-128}"
HIDDEN_UNITS="${HIDDEN_UNITS:-64}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
DROPOUT_RATE="${DROPOUT_RATE:-0.2}"
SEED="${SEED:-42}"
PATIENCE="${PATIENCE:-10}"
EVAL_EVERY="${EVAL_EVERY:-1}"
ALPHA="${ALPHA:-0.05}"
BETA="${BETA:-0.1}"
MLM_PROB="${MLM_PROB:-0.05}"
TEMPERATURE="${TEMPERATURE:-0.2}"
REPR_MODE="${REPR_MODE:-mean}"

FULL_PREFIX="${FULL_PREFIX:-movies_gru4rec_model_agnostic_mean_a005_b01}"
WO_PREFIX="${WO_PREFIX:-movies_gru4rec_model_agnostic_wo_sacp}"

TRAIN_FILE="${DATA_ROOT}/data_${DATASET}/${DATASET}_train.txt"
TEXT_FILE="${DATA_ROOT}/data_${DATASET}/text_name_dict.json.gz"
ITEM_EMB="${ASSET_ROOT}/${DATASET}/item_semantic_embeddings.pt"
USER_EMB="${ASSET_ROOT}/${DATASET}/user_semantic_embeddings.pt"
SEQ_KEYS="${ASSET_ROOT}/${DATASET}/seq_keys_to_int.pkl"

prepare_data() {
  echo "===== prepare dataset ====="
  if [ -f "${TRAIN_FILE}" ] && [ -f "${TEXT_FILE}" ]; then
    echo "[skip] dataset exists"
    return 0
  fi

  cd "${ROOT_DIR}/SeqRec/sasrec" || return 1
  python - <<'PY'
from data_preprocess import preprocess_raw_5core
preprocess_raw_5core("Movies_and_TV")
PY
  local status=$?
  cd "${ROOT_DIR}" || return 1
  return ${status}
}

prepare_assets() {
  echo "===== prepare semantic assets ====="
  if [ -f "${ITEM_EMB}" ] && [ -f "${USER_EMB}" ] && [ -f "${SEQ_KEYS}" ]; then
    echo "[skip] semantic assets exist"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python SA/generate_assets.py \
    --dataset "${DATASET}" \
    --data_root "${DATA_ROOT}" \
    --asset_root "${ASSET_ROOT}" \
    --llm "${LLM_NAME}" \
    --llm_path "${LLM_PATH}" \
    --device cuda:0 \
    --batch_size 64 \
    --maxlen "${MAXLEN}" \
    --max_length 256 \
    --neighbor_k 10
}

run_trial() {
  local prefix="$1"
  local alpha="$2"
  local beta="$3"
  local ckpt="${SAVE_DIR}/${DATASET}/${prefix}/model_metric_best.pth"

  echo "===== train ${prefix} ====="
  if [ -f "${ckpt}" ]; then
    echo "[skip] checkpoint exists: ${ckpt}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" python -m SeqRec.sasrec.train_sacp_sasrec \
    --dataset "${DATASET}" \
    --data_root "${DATA_ROOT}" \
    --sa_asset_root "${ASSET_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --test_batch_size "${TEST_BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0.0 \
    --l2_emb 0.0 \
    --maxlen "${MAXLEN}" \
    --hidden_units "${HIDDEN_UNITS}" \
    --num_blocks "${NUM_BLOCKS}" \
    --dropout_rate "${DROPOUT_RATE}" \
    --recsys_backbone gru4rec \
    --gru_output_head linear \
    --gru_sacp_layer_norm \
    --gru_enable_user_cl \
    --gru_user_cl_similarity cos \
    --gru_user_cl_temperature 0.2 \
    --gru_user_cl_post_norm \
    --sa_similarity cos \
    --sa_repr_mode "${REPR_MODE}" \
    --seed "${SEED}" \
    --device 0 \
    --save_dir "${SAVE_DIR}" \
    --save_prefix "${prefix}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --sa_mlm_probability "${MLM_PROB}" \
    --sa_temperature "${TEMPERATURE}" \
    --eval_every "${EVAL_EVERY}" \
    --patience "${PATIENCE}"
}

prepare_data
prepare_assets
run_trial "${FULL_PREFIX}" "${ALPHA}" "${BETA}"
run_trial "${WO_PREFIX}" 0.0 0.0

echo "===== done ====="
echo "full_ckpt=${SAVE_DIR}/${DATASET}/${FULL_PREFIX}/model_metric_best.pth"
echo "wo_sacp_ckpt=${SAVE_DIR}/${DATASET}/${WO_PREFIX}/model_metric_best.pth"
