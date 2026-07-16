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
if [ -n "${LLM_PATH:-}" ]; then
  LLM_PATH="${LLM_PATH}"
else
  for candidate in \
    "${HOME}/cyx/models/llama3_3b" \
    "/home/$(whoami)/cyx/models/llama3_3b" \
    "${HOME}/models/llama3_3b" \
    "/home/$(whoami)/models/llama3_3b" \
    "/home/cyx/models/llama3_3b"
  do
    if [ -d "${candidate}" ]; then
      LLM_PATH="${candidate}"
      break
    fi
  done
fi

if [ -z "${LLM_PATH:-}" ]; then
  echo "[error] could not resolve LLM_PATH automatically"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATASET=${DATASET:-Movies_and_TV}
ASSET_ROOT=${ASSET_ROOT:-./SA_assets}
ASSET_BATCH_SIZE=${ASSET_BATCH_SIZE:-64}
ASSET_NEIGHBOR_K=${ASSET_NEIGHBOR_K:-10}

TEACHER_SAVE_ROOT=${TEACHER_SAVE_ROOT:-./SeqRec/gru4rec}
FULL_PREFIX=${FULL_PREFIX:-movies_gru4rec_full_teacher_l40s}
WO_SACP_PREFIX=${WO_SACP_PREFIX:-movies_gru4rec_wo_sacp_teacher_l40s}

FULL_SI_SAVE=${FULL_SI_SAVE:-movies_gru4rec_full_si_l40s}
WO_SACP_SI_SAVE=${WO_SACP_SI_SAVE:-movies_gru4rec_wo_sacp_si_l40s}

TRAIN_FILE=./SeqRec/data_${DATASET}/${DATASET}_train.txt
TEXT_FILE=./SeqRec/data_${DATASET}/text_name_dict.json.gz
ITEM_EMB=${ASSET_ROOT}/${DATASET}/item_semantic_embeddings.pt
USER_EMB=${ASSET_ROOT}/${DATASET}/user_semantic_embeddings.pt
SEQ_KEYS=${ASSET_ROOT}/${DATASET}/seq_keys_to_int.pkl

FULL_CKPT=${TEACHER_SAVE_ROOT}/${DATASET}/${FULL_PREFIX}/model_metric_best.pth
WO_SACP_CKPT=${TEACHER_SAVE_ROOT}/${DATASET}/${WO_SACP_PREFIX}/model_metric_best.pth

prepare_data() {
  echo "===== [0/4] prepare dataset ====="
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
  echo "===== [1/4] prepare semantic assets ====="
  if [ -f "${ITEM_EMB}" ] && [ -f "${USER_EMB}" ] && [ -f "${SEQ_KEYS}" ]; then
    echo "[skip] semantic assets exist"
    return 0
  fi

  PYTHONPATH=$(pwd) python SA/generate_assets.py \
    --dataset ${DATASET} \
    --data_root ./SeqRec \
    --asset_root "${ASSET_ROOT}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --device cuda:0 \
    --batch_size ${ASSET_BATCH_SIZE} \
    --maxlen 128 \
    --max_length 256 \
    --neighbor_k ${ASSET_NEIGHBOR_K}
}

train_teacher() {
  local prefix="$1"
  local alpha="$2"
  local beta="$3"
  local ckpt="${TEACHER_SAVE_ROOT}/${DATASET}/${prefix}/model_metric_best.pth"

  echo "===== train teacher: ${prefix} ====="
  if [ -f "${ckpt}" ]; then
    echo "[skip] teacher exists: ${ckpt}"
    return 0
  fi

  python -m SeqRec.sasrec.train_sa \
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
    --dropout_rate 0.2 \
    --sa_use_projection_head \
    --sa_proj_hidden_dim 64 \
    --sa_contrast_norm \
    --sa_similarity cos \
    --seed 42 \
    --device 0 \
    --save_dir "${TEACHER_SAVE_ROOT}" \
    --save_prefix "${prefix}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --eval_every ${SA_EVAL_EVERY:-1} \
    --patience ${SA_PATIENCE:-10}
}

run_si() {
  local variant="$1"
  local ckpt="$2"
  local save_dir="$3"
  local result_file="./models/${DATASET}/${save_dir}/${DATASET}_${LLM_NAME}_all_results.txt"

  echo "===== run SI: ${variant} ====="
  if [ -f "${result_file}" ]; then
    echo "[skip] SI result exists: ${result_file}"
    return 0
  fi

  python main.py --train \
    --device 0 \
    --recsys gru4rec \
    --rec_pre_trained_data ${DATASET} \
    --recsys_ckpt_path "${ckpt}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --hf_local_only \
    --hf_cache_dir "${LLM_PATH}" \
    --batch_size 20 \
    --train_candidate_num ${CAND_NUM:-4} \
    --candidate_chunk_size 80 \
    --batch_size_infer 10 \
    --llm_max_length ${LLM_MAX_LENGTH:-1024} \
    --eval_item_batch ${EVAL_ITEM_BATCH:-32} \
    --eval_max_length ${EVAL_MAX_LENGTH:-1024} \
    --eval_min_length ${EVAL_MIN_LENGTH:-1024} \
    --maxlen 128 \
    --stage2_lr 1e-4 \
    --num_epochs 15 \
    --early_stop_patience 3 \
    --min_epochs_before_early_stop 10 \
    --seed 42 \
    --match_weight 1.0 \
    --save_dir "${save_dir}"
}

prepare_data || exit 1
prepare_assets || exit 1

echo "===== [2/4] full: GRU4Rec + SACP -> SI ====="
train_teacher "${FULL_PREFIX}" 0.1 0.1 || exit 1
run_si "full" "${FULL_CKPT}" "${FULL_SI_SAVE}" || exit 1

echo "===== [3/4] w/o SACP: GRU4Rec rec-only -> SI ====="
train_teacher "${WO_SACP_PREFIX}" 0.0 0.0 || exit 1
run_si "wo_sacp" "${WO_SACP_CKPT}" "${WO_SACP_SI_SAVE}" || exit 1

echo "===== done ====="
echo "full_ckpt=${FULL_CKPT}"
echo "wo_sacp_ckpt=${WO_SACP_CKPT}"
echo "./models/${DATASET}/${FULL_SI_SAVE}"
echo "./models/${DATASET}/${WO_SACP_SI_SAVE}"
