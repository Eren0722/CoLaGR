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

DATA_REBUILT=0
STALE_SUFFIX=stale_$(date +%Y%m%d_%H%M%S)_$$

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
  echo "        set it manually, e.g. export LLM_PATH=/home/$(whoami)/cyx/models/llama3_3b"
  exit 1
fi
echo "[info] using LLM_PATH=${LLM_PATH}"
DATASET=Industrial_and_Scientific
TRAIN_DEVICES=${TRAIN_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
ASSET_ROOT=${ASSET_ROOT:-./SA_assets}
ASSET_BATCH_SIZE=${ASSET_BATCH_SIZE:-64}
ASSET_NEIGHBOR_K=${ASSET_NEIGHBOR_K:-10}
CAND_NUM=${CAND_NUM:-4}
SA_EVAL_EVERY=${SA_EVAL_EVERY:-1}
SA_PATIENCE=${SA_PATIENCE:-10}

FULL_PREFIX=${FULL_PREFIX:-scientific_sa_teacher}
WO_USER_PREFIX=${WO_USER_PREFIX:-scientific_wo_usercl_teacher}
WO_ITEM_PREFIX=${WO_ITEM_PREFIX:-scientific_wo_itemcl_teacher}

FULL_SI_SAVE=${FULL_SI_SAVE:-scientific_si_5090}
PURE_SI_SAVE=${PURE_SI_SAVE:-scientific_pure_sasrec_si_5090}
WO_USER_SI_SAVE=${WO_USER_SI_SAVE:-scientific_wo_usercl_si_5090}
WO_ITEM_SI_SAVE=${WO_ITEM_SI_SAVE:-scientific_wo_itemcl_si_5090}
WO_MATCH_SI_SAVE=${WO_MATCH_SI_SAVE:-scientific_wo_match_si_5090}

FULL_CKPT=./SeqRec/sasrec/${DATASET}/${FULL_PREFIX}/model_metric_best.pth
PURE_CKPT=./SeqRec/sasrec/${DATASET}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth
WO_USER_CKPT=./SeqRec/sasrec/${DATASET}/${WO_USER_PREFIX}/model_metric_best.pth
WO_ITEM_CKPT=./SeqRec/sasrec/${DATASET}/${WO_ITEM_PREFIX}/model_metric_best.pth

TRAIN_FILE=./SeqRec/data_${DATASET}/${DATASET}_train.txt
TEXT_FILE=./SeqRec/data_${DATASET}/text_name_dict.json.gz

ITEM_EMB=${ASSET_ROOT}/${DATASET}/item_semantic_embeddings.pt
USER_EMB=${ASSET_ROOT}/${DATASET}/user_semantic_embeddings.pt
SEQ_KEYS=${ASSET_ROOT}/${DATASET}/seq_keys_to_int.pkl

archive_if_exists() {
  local path="$1"
  if [ -e "${path}" ]; then
    local backup="${path}.${STALE_SUFFIX}"
    mv "${path}" "${backup}"
    echo "[info] archived stale artifact: ${path} -> ${backup}"
  fi
}

prepare_data() {
  echo "===== [0/6] prepare dataset ====="
  local max_try=${DATA_PREP_MAX_RETRY:-3}
  local try_idx=1

  if [ "${FORCE_REBUILD_DATA:-0}" = "1" ]; then
    echo "[info] FORCE_REBUILD_DATA=1, rebuilding dataset"
  elif [ -f "${TRAIN_FILE}" ] && [ -f "${TEXT_FILE}" ]; then
    echo "[skip] dataset exists"
    return 0
  fi

  while [ ${try_idx} -le ${max_try} ]; do
    echo "[info] dataset preprocess attempt ${try_idx}/${max_try}"

    cd "${ROOT_DIR}/SeqRec/sasrec" || return 1
    python - <<'PY'
from data_preprocess import preprocess_raw_5core
preprocess_raw_5core("Industrial_and_Scientific")
PY
    local status=$?
    cd "${ROOT_DIR}" || return 1

    if [ ${status} -eq 0 ] && [ -f "${TRAIN_FILE}" ] && [ -f "${TEXT_FILE}" ]; then
      DATA_REBUILT=1
      return 0
    fi

    echo "[warn] dataset preprocess failed on attempt ${try_idx}/${max_try}"
    if [ ${try_idx} -lt ${max_try} ]; then
      local sleep_sec=$((try_idx * 30))
      echo "[info] retry after ${sleep_sec}s..."
      sleep ${sleep_sec}
    fi
    try_idx=$((try_idx + 1))
  done

  echo "[error] dataset preprocess failed after ${max_try} attempts"
  return 1
}

invalidate_downstream_if_data_rebuilt() {
  if [ "${DATA_REBUILT}" != "1" ]; then
    return 0
  fi

  echo "[info] dataset changed in this run, archiving dependent artifacts"
  archive_if_exists "${ASSET_ROOT}/${DATASET}"
  archive_if_exists "./SeqRec/sasrec/${DATASET}/${FULL_PREFIX}"
  archive_if_exists "./SeqRec/sasrec/${DATASET}/${WO_USER_PREFIX}"
  archive_if_exists "./SeqRec/sasrec/${DATASET}/${WO_ITEM_PREFIX}"
  archive_if_exists "${PURE_CKPT}"
  archive_if_exists "./models/${DATASET}/${FULL_SI_SAVE}"
  archive_if_exists "./models/${DATASET}/${PURE_SI_SAVE}"
  archive_if_exists "./models/${DATASET}/${WO_USER_SI_SAVE}"
  archive_if_exists "./models/${DATASET}/${WO_ITEM_SI_SAVE}"
  archive_if_exists "./models/${DATASET}/${WO_MATCH_SI_SAVE}"
}

prepare_assets() {
  echo "===== [1/6] generate SA assets ====="
  if [ -f "${ITEM_EMB}" ] && [ -f "${USER_EMB}" ] && [ -f "${SEQ_KEYS}" ]; then
    echo "[skip] SA assets exist"
    return 0
  fi

  PYTHONPATH=$(pwd) CUDA_VISIBLE_DEVICES=0 python SA/generate_assets.py \
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
  local port="$4"
  local ckpt="./SeqRec/sasrec/${DATASET}/${prefix}/model_metric_best.pth"

  echo "===== train teacher: ${prefix} ====="
  if [ -f "${ckpt}" ]; then
    echo "[skip] teacher exists: ${ckpt}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONPATH=$(pwd) \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${port} \
    -m SeqRec.sasrec.train_sa \
    --dataset ${DATASET} \
    --data_root ./SeqRec \
    --sa_asset_root "${ASSET_ROOT}" \
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
    --save_prefix "${prefix}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --eval_every ${SA_EVAL_EVERY} \
    --patience ${SA_PATIENCE}

  local status=$?
  if [ ${status} -ne 0 ]; then
    echo "[warn] teacher exited with status ${status}, checking checkpoint..."
  fi
  [ -f "${ckpt}" ]
}

train_pure_sasrec() {
  echo "===== train pure SASRec teacher ====="
  if [ -f "${PURE_CKPT}" ]; then
    echo "[skip] pure teacher exists: ${PURE_CKPT}"
    return 0
  fi

  cd "${ROOT_DIR}/SeqRec/sasrec" || return 1
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset ${DATASET} \
    --batch_size 256 \
    --num_epochs 200 \
    --lr 0.001 \
    --maxlen 128 \
    --num_heads 1 \
    --dropout_rate 0.2 \
    --hidden_units 64 \
    --num_blocks 2 \
    --device 0
  local status=$?
  cd "${ROOT_DIR}" || return 1

  if [ ${status} -ne 0 ]; then
    echo "[warn] pure teacher exited with status ${status}, checking checkpoint..."
  fi
  [ -f "${PURE_CKPT}" ]
}

run_si() {
  local variant="$1"
  local ckpt="$2"
  local save_dir="$3"
  local match_weight="$4"
  local port="$5"
  local result_file="./models/${DATASET}/${save_dir}/${DATASET}_${LLM_NAME}_all_results.txt"

  echo "===== run SI: ${variant} ====="
  if [ -f "${result_file}" ]; then
    echo "[skip] SI result exists: ${result_file}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${port} \
    main.py --train --multi_gpu --world_size ${NPROC_PER_NODE} \
    --recsys sasrec \
    --rec_pre_trained_data ${DATASET} \
    --recsys_ckpt_path "${ckpt}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --hf_local_only \
    --hf_cache_dir "${LLM_PATH}" \
    --batch_size 20 \
    --train_candidate_num ${CAND_NUM} \
    --candidate_chunk_size 80 \
    --batch_size_infer 10 \
    --llm_max_length ${LLM_MAX_LENGTH:-1024} \
    --eval_item_batch ${EVAL_ITEM_BATCH:-32} \
    --eval_max_length ${EVAL_MAX_LENGTH:-1024} \
    --eval_min_length ${EVAL_MIN_LENGTH:-1024} \
    --maxlen 128 \
    --stage2_lr 1e-4 \
    --num_epochs 25 \
    --early_stop_patience 3 \
    --min_epochs_before_early_stop 12 \
    --seed 42 \
    --match_weight "${match_weight}" \
    --save_dir "${save_dir}"

  local status=$?
  if [ ${status} -ne 0 ]; then
    echo "[warn] SI ${variant} exited with status ${status}, checking result file..."
  fi
  [ -f "${result_file}" ]
}

prepare_data || exit 1
invalidate_downstream_if_data_rebuilt || exit 1
prepare_assets || exit 1

echo "===== [2/6] full SA -> SI ====="
train_teacher "${FULL_PREFIX}" 0.1 0.1 29731 || exit 1
run_si "full" "${FULL_CKPT}" "${FULL_SI_SAVE}" 1.0 29741 || exit 1

echo "===== [3/6] pure SASRec -> SI ====="
train_pure_sasrec || exit 1
run_si "pure_sasrec" "${PURE_CKPT}" "${PURE_SI_SAVE}" 1.0 29742 || exit 1

echo "===== [4/6] w/o user-cl -> SI ====="
train_teacher "${WO_USER_PREFIX}" 0.0 0.1 29732 || exit 1
run_si "wo_usercl" "${WO_USER_CKPT}" "${WO_USER_SI_SAVE}" 1.0 29743 || exit 1

echo "===== [5/6] w/o item-cl -> SI ====="
train_teacher "${WO_ITEM_PREFIX}" 0.1 0.0 29733 || exit 1
run_si "wo_itemcl" "${WO_ITEM_CKPT}" "${WO_ITEM_SI_SAVE}" 1.0 29744 || exit 1

echo "===== [6/6] w/o match -> SI ====="
run_si "wo_match" "${FULL_CKPT}" "${WO_MATCH_SI_SAVE}" 0.0 29745 || exit 1

echo "===== done ====="
echo "dataset=${DATASET}, candidate_num=${CAND_NUM}"
echo "./models/${DATASET}/${FULL_SI_SAVE}"
echo "./models/${DATASET}/${PURE_SI_SAVE}"
echo "./models/${DATASET}/${WO_USER_SI_SAVE}"
echo "./models/${DATASET}/${WO_ITEM_SI_SAVE}"
echo "./models/${DATASET}/${WO_MATCH_SI_SAVE}"
