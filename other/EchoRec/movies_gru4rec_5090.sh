#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHONPATH=$(pwd)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1800}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-1800}
if python -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
else
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
unset HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_OFFLINE 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
TRAIN_DEVICES=${TRAIN_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

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
  echo "        set it manually, e.g. export LLM_PATH=/home/$(whoami)/models/llama3_3b"
  exit 1
fi

DATASET=${DATASET:-Movies_and_TV}
ASSET_ROOT=${ASSET_ROOT:-./SA_assets}
ASSET_BATCH_SIZE=${ASSET_BATCH_SIZE:-64}
ASSET_NEIGHBOR_K=${ASSET_NEIGHBOR_K:-10}

TEACHER_PREFIX=${TEACHER_PREFIX:-movies_gru4rec_teacher}
TEACHER_SAVE_ROOT=${TEACHER_SAVE_ROOT:-./SeqRec/gru4rec}
SI_SAVE_DIR=${SI_SAVE_DIR:-movies_gru4rec_si_5090}

TRAIN_FILE=./SeqRec/data_${DATASET}/${DATASET}_train.txt
TEXT_FILE=./SeqRec/data_${DATASET}/text_name_dict.json.gz
ITEM_EMB=${ASSET_ROOT}/${DATASET}/item_semantic_embeddings.pt
USER_EMB=${ASSET_ROOT}/${DATASET}/user_semantic_embeddings.pt
SEQ_KEYS=${ASSET_ROOT}/${DATASET}/seq_keys_to_int.pkl
TEACHER_CKPT=${TEACHER_SAVE_ROOT}/${DATASET}/${TEACHER_PREFIX}/model_metric_best.pth
RESULT_FILE=./models/${DATASET}/${SI_SAVE_DIR}/${DATASET}_${LLM_NAME}_all_results.txt

prepare_data() {
  echo "===== [0/3] prepare dataset ====="
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
  echo "===== [1/3] prepare semantic assets ====="
  if [ -f "${ITEM_EMB}" ] && [ -f "${USER_EMB}" ] && [ -f "${SEQ_KEYS}" ]; then
    echo "[skip] semantic assets exist"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python SA/generate_assets.py \
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
  echo "===== [2/3] train GRU4Rec teacher ====="
  if [ -f "${TEACHER_CKPT}" ]; then
    echo "[skip] teacher exists: ${TEACHER_CKPT}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONPATH=$(pwd) \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${TEACHER_PORT:-29861} \
    -m SeqRec.sasrec.train_sa \
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
    --save_dir "${TEACHER_SAVE_ROOT}" \
    --save_prefix "${TEACHER_PREFIX}" \
    --sa_alpha 0.1 \
    --sa_beta 0.1 \
    --eval_every ${SA_EVAL_EVERY:-1} \
    --patience ${SA_PATIENCE:-10}
}

run_si() {
  echo "===== [3/3] run SI on top of GRU4Rec teacher ====="
  if [ -f "${RESULT_FILE}" ]; then
    echo "[skip] SI result exists: ${RESULT_FILE}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${SI_PORT:-29862} \
    main.py --train --multi_gpu --world_size ${NPROC_PER_NODE} \
    --recsys gru4rec \
    --rec_pre_trained_data ${DATASET} \
    --recsys_ckpt_path "${TEACHER_CKPT}" \
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
    --save_dir "${SI_SAVE_DIR}"
}

prepare_data || exit 1
prepare_assets || exit 1
train_teacher || exit 1
run_si || exit 1

echo "===== done ====="
echo "teacher_ckpt=${TEACHER_CKPT}"
echo "si_dir=./models/${DATASET}/${SI_SAVE_DIR}"
