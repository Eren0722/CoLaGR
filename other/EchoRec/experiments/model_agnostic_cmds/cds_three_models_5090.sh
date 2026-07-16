#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
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

DATASET=${DATASET:-CDs_and_Vinyl}
DATA_ROOT=${DATA_ROOT:-./SeqRec}
ASSET_ROOT=${ASSET_ROOT:-./SA_assets}
ASSET_BATCH_SIZE=${ASSET_BATCH_SIZE:-64}
ASSET_NEIGHBOR_K=${ASSET_NEIGHBOR_K:-10}

COMMON_BATCH_SIZE=${COMMON_BATCH_SIZE:-256}
COMMON_TEST_BATCH_SIZE=${COMMON_TEST_BATCH_SIZE:-512}
COMMON_NUM_EPOCHS=${COMMON_NUM_EPOCHS:-100}
COMMON_LR=${COMMON_LR:-1e-3}
COMMON_SEED=${COMMON_SEED:-42}
COMMON_MAXLEN=${COMMON_MAXLEN:-128}
COMMON_EVAL_EVERY=${COMMON_EVAL_EVERY:-1}
COMMON_STAGE1_PATIENCE=${COMMON_STAGE1_PATIENCE:-10}

SI_BATCH_SIZE=${SI_BATCH_SIZE:-20}
SI_BATCH_SIZE_INFER=${SI_BATCH_SIZE_INFER:-10}
SI_TRAIN_CANDIDATE_NUM=${SI_TRAIN_CANDIDATE_NUM:-4}
SI_CANDIDATE_CHUNK_SIZE=${SI_CANDIDATE_CHUNK_SIZE:-80}
SI_NUM_EPOCHS=${SI_NUM_EPOCHS:-25}
SI_LR=${SI_LR:-1e-4}
SI_MATCH_WEIGHT=${SI_MATCH_WEIGHT:-1.0}
SI_EARLY_STOP_PATIENCE=${SI_EARLY_STOP_PATIENCE:-3}
SI_MIN_EPOCHS=${SI_MIN_EPOCHS:-10}
SI_LLM_MAX_LENGTH=${SI_LLM_MAX_LENGTH:-1024}
SI_EVAL_ITEM_BATCH=${SI_EVAL_ITEM_BATCH:-32}
SI_EVAL_MAX_LENGTH=${SI_EVAL_MAX_LENGTH:-1024}
SI_EVAL_MIN_LENGTH=${SI_EVAL_MIN_LENGTH:-1024}

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
echo "[info] using LLM_PATH=${LLM_PATH}"

SASREC_SAVE_DIR=${SASREC_SAVE_DIR:-./SeqRec/sacp_sasrec}
BERT4REC_SAVE_DIR=${BERT4REC_SAVE_DIR:-./SeqRec/bert4rec}
GRU4REC_SAVE_DIR=${GRU4REC_SAVE_DIR:-./SeqRec/gru4rec}

SASREC_BATCH_SIZE=${SASREC_BATCH_SIZE:-256}
SASREC_TEST_BATCH_SIZE=${SASREC_TEST_BATCH_SIZE:-512}
SASREC_NUM_EPOCHS=${SASREC_NUM_EPOCHS:-50}
SASREC_LR=${SASREC_LR:-1e-3}
SASREC_BACKBONE_LR=${SASREC_BACKBONE_LR:-1e-4}
SASREC_ALPHA=${SASREC_ALPHA:-0.1}
SASREC_BETA=${SASREC_BETA:-0.1}
SASREC_MLM_PROB=${SASREC_MLM_PROB:-0.2}
SASREC_TEMPERATURE=${SASREC_TEMPERATURE:-1.0}
SASREC_FULL_PREFIX=${SASREC_FULL_PREFIX:-cds_sasrec_sacp_adapt_ce}
SASREC_WO_PREFIX=${SASREC_WO_PREFIX:-cds_sasrec_sacp_adapt_wo_sacp}
SASREC_FULL_SI_SAVE=${SASREC_FULL_SI_SAVE:-cds_sasrec_model_agnostic_full_si_5090}
SASREC_WO_SI_SAVE=${SASREC_WO_SI_SAVE:-cds_sasrec_model_agnostic_wo_sacp_si_5090}

BERT4REC_FULL_PREFIX=${BERT4REC_FULL_PREFIX:-cds_bertstyle_nextitem_model_agnostic_mean_a01_b01}
BERT4REC_WO_PREFIX=${BERT4REC_WO_PREFIX:-cds_bertstyle_nextitem_model_agnostic_wo_sacp}
BERT4REC_FULL_SI_SAVE=${BERT4REC_FULL_SI_SAVE:-cds_bert4rec_model_agnostic_full_si_5090}
BERT4REC_WO_SI_SAVE=${BERT4REC_WO_SI_SAVE:-cds_bert4rec_model_agnostic_wo_sacp_si_5090}

GRU4REC_FULL_PREFIX=${GRU4REC_FULL_PREFIX:-cds_gru4rec_model_agnostic_mean_a01_b01}
GRU4REC_WO_PREFIX=${GRU4REC_WO_PREFIX:-cds_gru4rec_model_agnostic_wo_sacp}
GRU4REC_FULL_SI_SAVE=${GRU4REC_FULL_SI_SAVE:-cds_gru4rec_model_agnostic_full_si_5090}
GRU4REC_WO_SI_SAVE=${GRU4REC_WO_SI_SAVE:-cds_gru4rec_model_agnostic_wo_sacp_si_5090}

SASREC_FULL_CKPT=${SASREC_SAVE_DIR}/${DATASET}/${SASREC_FULL_PREFIX}/model_metric_best.pth
SASREC_WO_CKPT=${SASREC_SAVE_DIR}/${DATASET}/${SASREC_WO_PREFIX}/model_metric_best.pth
BERT4REC_FULL_CKPT=${BERT4REC_SAVE_DIR}/${DATASET}/${BERT4REC_FULL_PREFIX}/model_metric_best.pth
BERT4REC_WO_CKPT=${BERT4REC_SAVE_DIR}/${DATASET}/${BERT4REC_WO_PREFIX}/model_metric_best.pth
GRU4REC_FULL_CKPT=${GRU4REC_SAVE_DIR}/${DATASET}/${GRU4REC_FULL_PREFIX}/model_metric_best.pth
GRU4REC_WO_CKPT=${GRU4REC_SAVE_DIR}/${DATASET}/${GRU4REC_WO_PREFIX}/model_metric_best.pth

TRAIN_FILE=${DATA_ROOT}/data_${DATASET}/${DATASET}_train.txt
TEXT_FILE=${DATA_ROOT}/data_${DATASET}/text_name_dict.json.gz
ITEM_EMB=${ASSET_ROOT}/${DATASET}/item_semantic_embeddings.pt
USER_EMB=${ASSET_ROOT}/${DATASET}/user_semantic_embeddings.pt
SEQ_KEYS=${ASSET_ROOT}/${DATASET}/seq_keys_to_int.pkl

prepare_data() {
  echo "===== [0/8] prepare dataset ====="
  if [ -f "${TRAIN_FILE}" ] && [ -f "${TEXT_FILE}" ]; then
    echo "[skip] dataset exists"
    return 0
  fi

  cd "${ROOT_DIR}/SeqRec/sasrec" || return 1
  python - <<'PY'
from data_preprocess import preprocess_raw_5core
preprocess_raw_5core("CDs_and_Vinyl")
PY
  local status=$?
  cd "${ROOT_DIR}" || return 1
  return ${status}
}

prepare_assets() {
  echo "===== [1/8] prepare semantic assets ====="
  if [ -f "${ITEM_EMB}" ] && [ -f "${USER_EMB}" ] && [ -f "${SEQ_KEYS}" ]; then
    echo "[skip] semantic assets exist"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python SA/generate_assets.py \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --asset_root "${ASSET_ROOT}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --device cuda:0 \
    --batch_size ${ASSET_BATCH_SIZE} \
    --maxlen ${COMMON_MAXLEN} \
    --max_length 256 \
    --neighbor_k ${ASSET_NEIGHBOR_K}
}

run_sasrec_teacher() {
  local prefix="$1"
  local alpha="$2"
  local beta="$3"
  local port="$4"
  local ckpt="${SASREC_SAVE_DIR}/${DATASET}/${prefix}/model_metric_best.pth"

  echo "===== train SASRec teacher: ${prefix} ====="
  if [ -f "${ckpt}" ]; then
    echo "[skip] checkpoint exists: ${ckpt}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONPATH=$(pwd) \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} --master_port=${port} \
    -m SeqRec.sasrec.train_sacp_sasrec \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --sa_asset_root "${ASSET_ROOT}" \
    --batch_size ${SASREC_BATCH_SIZE} \
    --test_batch_size ${SASREC_TEST_BATCH_SIZE} \
    --num_epochs ${SASREC_NUM_EPOCHS} \
    --learning_rate ${SASREC_LR} \
    --backbone_learning_rate ${SASREC_BACKBONE_LR} \
    --weight_decay 0.0 \
    --l2_emb 0.0 \
    --maxlen ${COMMON_MAXLEN} \
    --hidden_units 64 \
    --num_blocks 2 \
    --num_heads 1 \
    --dropout_rate 0.2 \
    --inner_size 256 \
    --hidden_dropout_prob 0.2 \
    --attn_dropout_prob 0.2 \
    --rec_objective ce \
    --recsys_backbone sasrec \
    --sacp_preset raw_sasrec \
    --sa_similarity cos \
    --sa_repr_mode mean \
    --seed ${COMMON_SEED} \
    --device 0 \
    --save_dir "${SASREC_SAVE_DIR}" \
    --save_prefix "${prefix}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --sa_k_num 10 \
    --sa_mlm_probability ${SASREC_MLM_PROB} \
    --sa_temperature ${SASREC_TEMPERATURE} \
    --eval_every ${COMMON_EVAL_EVERY} \
    --patience ${COMMON_STAGE1_PATIENCE}
}

run_bert4rec_teacher() {
  local prefix="$1"
  local alpha="$2"
  local beta="$3"
  local port="$4"
  local ckpt="${BERT4REC_SAVE_DIR}/${DATASET}/${prefix}/model_metric_best.pth"

  echo "===== train BERT4Rec teacher: ${prefix} ====="
  if [ -f "${ckpt}" ]; then
    echo "[skip] checkpoint exists: ${ckpt}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONPATH=$(pwd) \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} --master_port=${port} \
    -m SeqRec.sasrec.train_sacp_sasrec \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --sa_asset_root "${ASSET_ROOT}" \
    --batch_size ${COMMON_BATCH_SIZE} \
    --test_batch_size ${COMMON_TEST_BATCH_SIZE} \
    --num_epochs ${COMMON_NUM_EPOCHS} \
    --learning_rate ${COMMON_LR} \
    --weight_decay 0.0 \
    --l2_emb 0.0 \
    --maxlen ${COMMON_MAXLEN} \
    --hidden_units 64 \
    --num_blocks 2 \
    --num_heads 2 \
    --dropout_rate 0.2 \
    --inner_size 256 \
    --hidden_dropout_prob 0.2 \
    --attn_dropout_prob 0.2 \
    --hidden_act gelu \
    --bert_mask_prob 0.15 \
    --bert_rec_objective next_item \
    --recsys_backbone bert4rec \
    --sacp_preset raw_sasrec \
    --sa_similarity cos \
    --sa_repr_mode mean \
    --seed ${COMMON_SEED} \
    --device 0 \
    --save_dir "${BERT4REC_SAVE_DIR}" \
    --save_prefix "${prefix}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --sa_mlm_probability 0.05 \
    --sa_temperature 0.2 \
    --eval_every ${COMMON_EVAL_EVERY} \
    --patience ${COMMON_STAGE1_PATIENCE}
}

run_gru4rec_teacher() {
  local prefix="$1"
  local alpha="$2"
  local beta="$3"
  local port="$4"
  local ckpt="${GRU4REC_SAVE_DIR}/${DATASET}/${prefix}/model_metric_best.pth"

  echo "===== train GRU4Rec teacher: ${prefix} ====="
  if [ -f "${ckpt}" ]; then
    echo "[skip] checkpoint exists: ${ckpt}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} PYTHONPATH=$(pwd) \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} --master_port=${port} \
    -m SeqRec.sasrec.train_sacp_sasrec \
    --dataset ${DATASET} \
    --data_root "${DATA_ROOT}" \
    --sa_asset_root "${ASSET_ROOT}" \
    --batch_size ${COMMON_BATCH_SIZE} \
    --test_batch_size ${COMMON_TEST_BATCH_SIZE} \
    --num_epochs ${COMMON_NUM_EPOCHS} \
    --learning_rate ${COMMON_LR} \
    --weight_decay 0.0 \
    --l2_emb 0.0 \
    --maxlen ${COMMON_MAXLEN} \
    --hidden_units 64 \
    --num_blocks 2 \
    --dropout_rate 0.2 \
    --recsys_backbone gru4rec \
    --gru_output_head linear \
    --gru_sacp_layer_norm \
    --gru_enable_user_cl \
    --gru_user_cl_similarity cos \
    --gru_user_cl_temperature 0.2 \
    --gru_user_cl_post_norm \
    --sa_similarity cos \
    --sa_repr_mode mean \
    --seed ${COMMON_SEED} \
    --device 0 \
    --save_dir "${GRU4REC_SAVE_DIR}" \
    --save_prefix "${prefix}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --sa_mlm_probability 0.05 \
    --sa_temperature 0.2 \
    --eval_every ${COMMON_EVAL_EVERY} \
    --patience ${COMMON_STAGE1_PATIENCE}
}

run_si() {
  local variant="$1"
  local recsys="$2"
  local ckpt="$3"
  local save_dir="$4"
  local match_weight="$5"
  local port="$6"
  local result_file="./models/${DATASET}/${save_dir}/${DATASET}_${LLM_NAME}_all_results.txt"

  echo "===== run SI: ${variant} ====="
  if [ -f "${result_file}" ]; then
    echo "[skip] SI result exists: ${result_file}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} \
  torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} --master_port=${port} \
    main.py --train --multi_gpu --world_size ${NPROC_PER_NODE} \
    --recsys "${recsys}" \
    --rec_pre_trained_data ${DATASET} \
    --recsys_ckpt_path "${ckpt}" \
    --llm ${LLM_NAME} \
    --llm_path "${LLM_PATH}" \
    --hf_local_only \
    --hf_cache_dir "${LLM_PATH}" \
    --batch_size ${SI_BATCH_SIZE} \
    --train_candidate_num ${SI_TRAIN_CANDIDATE_NUM} \
    --candidate_chunk_size ${SI_CANDIDATE_CHUNK_SIZE} \
    --batch_size_infer ${SI_BATCH_SIZE_INFER} \
    --llm_max_length ${SI_LLM_MAX_LENGTH} \
    --eval_item_batch ${SI_EVAL_ITEM_BATCH} \
    --eval_max_length ${SI_EVAL_MAX_LENGTH} \
    --eval_min_length ${SI_EVAL_MIN_LENGTH} \
    --maxlen ${COMMON_MAXLEN} \
    --stage2_lr ${SI_LR} \
    --num_epochs ${SI_NUM_EPOCHS} \
    --early_stop_patience ${SI_EARLY_STOP_PATIENCE} \
    --min_epochs_before_early_stop ${SI_MIN_EPOCHS} \
    --seed ${COMMON_SEED} \
    --match_weight "${match_weight}" \
    --save_dir "${save_dir}"

  local status=$?
  if [ ${status} -ne 0 ]; then
    echo "[warn] SI ${variant} exited with status ${status}, checking result file..."
  fi
  [ -f "${result_file}" ]
}

prepare_data || exit 1
prepare_assets || exit 1

echo "===== [2/8] SASRec teacher ====="
run_sasrec_teacher "${SASREC_FULL_PREFIX}" ${SASREC_ALPHA} ${SASREC_BETA} 29611 || exit 1
run_sasrec_teacher "${SASREC_WO_PREFIX}" 0.0 0.0 29612 || exit 1

echo "===== [3/8] SASRec SI ====="
run_si "sasrec_full" "sasrec" "${SASREC_FULL_CKPT}" "${SASREC_FULL_SI_SAVE}" 1.0 29621 || exit 1
run_si "sasrec_wo_sacp" "sasrec" "${SASREC_WO_CKPT}" "${SASREC_WO_SI_SAVE}" 1.0 29622 || exit 1

echo "===== [4/8] BERT4Rec teacher ====="
run_bert4rec_teacher "${BERT4REC_FULL_PREFIX}" 0.1 0.1 29613 || exit 1
run_bert4rec_teacher "${BERT4REC_WO_PREFIX}" 0.0 0.0 29614 || exit 1

echo "===== [5/8] BERT4Rec SI ====="
run_si "bert4rec_full" "bert4rec" "${BERT4REC_FULL_CKPT}" "${BERT4REC_FULL_SI_SAVE}" 1.0 29623 || exit 1
run_si "bert4rec_wo_sacp" "bert4rec" "${BERT4REC_WO_CKPT}" "${BERT4REC_WO_SI_SAVE}" 1.0 29624 || exit 1

echo "===== [6/8] GRU4Rec teacher ====="
run_gru4rec_teacher "${GRU4REC_FULL_PREFIX}" 0.1 0.1 29615 || exit 1
run_gru4rec_teacher "${GRU4REC_WO_PREFIX}" 0.0 0.0 29616 || exit 1

echo "===== [7/8] GRU4Rec SI ====="
run_si "gru4rec_full" "gru4rec" "${GRU4REC_FULL_CKPT}" "${GRU4REC_FULL_SI_SAVE}" 1.0 29625 || exit 1
run_si "gru4rec_wo_sacp" "gru4rec" "${GRU4REC_WO_CKPT}" "${GRU4REC_WO_SI_SAVE}" 1.0 29626 || exit 1

echo "===== [8/8] summary ====="
echo "sasrec_full_ckpt=${SASREC_FULL_CKPT}"
echo "sasrec_wo_ckpt=${SASREC_WO_CKPT}"
echo "bert4rec_full_ckpt=${BERT4REC_FULL_CKPT}"
echo "bert4rec_wo_ckpt=${BERT4REC_WO_CKPT}"
echo "gru4rec_full_ckpt=${GRU4REC_FULL_CKPT}"
echo "gru4rec_wo_ckpt=${GRU4REC_WO_CKPT}"
echo "sasrec_full_si=./models/${DATASET}/${SASREC_FULL_SI_SAVE}"
echo "sasrec_wo_si=./models/${DATASET}/${SASREC_WO_SI_SAVE}"
echo "bert4rec_full_si=./models/${DATASET}/${BERT4REC_FULL_SI_SAVE}"
echo "bert4rec_wo_si=./models/${DATASET}/${BERT4REC_WO_SI_SAVE}"
echo "gru4rec_full_si=./models/${DATASET}/${GRU4REC_FULL_SI_SAVE}"
echo "gru4rec_wo_si=./models/${DATASET}/${GRU4REC_WO_SI_SAVE}"
