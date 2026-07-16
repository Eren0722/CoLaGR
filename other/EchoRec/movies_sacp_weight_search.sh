#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}" || exit 1

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

DATASET="${DATASET:-Movies_and_TV}"
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-./SeqRec}"
ASSET_ROOT="${ASSET_ROOT:-./SA_assets}"
SAVE_DIR="${SAVE_DIR:-./SeqRec/sacp_sasrec}"
LOG_DIR="${LOG_DIR:-./debug_logs/weight_search}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
MAXLEN="${MAXLEN:-128}"
HIDDEN_UNITS="${HIDDEN_UNITS:-64}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
NUM_HEADS="${NUM_HEADS:-1}"
DROPOUT_RATE="${DROPOUT_RATE:-0.2}"
PATIENCE="${PATIENCE:-10}"
EVAL_EVERY="${EVAL_EVERY:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"

PHASE="${1:-phase1}"

mkdir -p "${LOG_DIR}"

run_trial() {
  local tag="$1"
  local alpha="$2"
  local beta="$3"
  local mlm="$4"
  local seed="$5"

  local log_file="${LOG_DIR}/${tag}.log"
  local out_dir="${SAVE_DIR}/${DATASET}/${tag}"

  if [ -f "${log_file}" ] && grep -q "Training finished" "${log_file}"; then
    echo "[skip] completed log exists: ${tag}"
    return 0
  fi

  if [ -f "${out_dir}/model_metric_best.pth" ] && [ -f "${out_dir}/model.pth" ]; then
    echo "[skip] checkpoint exists: ${tag}"
    return 0
  fi

  echo "===== running: ${tag} ====="
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="$(pwd)" python -m SeqRec.sasrec.train_sacp_sasrec \
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
    --num_heads "${NUM_HEADS}" \
    --dropout_rate "${DROPOUT_RATE}" \
    --recsys_backbone sasrec \
    --sacp_preset raw_sasrec \
    --rec_objective bce \
    --bce_mode sampler \
    --seed "${seed}" \
    --device 0 \
    --save_dir "${SAVE_DIR}" \
    --save_prefix "${tag}" \
    --sa_alpha "${alpha}" \
    --sa_beta "${beta}" \
    --sa_mlm_probability "${mlm}" \
    --sa_temperature "${TEMPERATURE}" \
    --eval_every "${EVAL_EVERY}" \
    --patience "${PATIENCE}" 2>&1 | tee "${log_file}"
}

phase1() {
  # Coarse search: lock mlm and sweep the two loss weights.
  local mlm="0.05"
  local seed="42"
  local alphas=(0.0025 0.005 0.0075 0.01 0.015)
  local betas=(0.03 0.04 0.045 0.05 0.06)

  for alpha in "${alphas[@]}"; do
    for beta in "${betas[@]}"; do
      local tag
      tag=$(printf "movies_sacp_wsearch_p1_a%s_b%s_m%s_s%s" "${alpha}" "${beta}" "${mlm}" "${seed}" | tr '.' 'p')
      run_trial "${tag}" "${alpha}" "${beta}" "${mlm}" "${seed}"
    done
  done
}

phase2() {
  # Fine search around the empirically strongest band observed so far.
  local seed="42"
  local configs=(
    "0.004 0.040 0.05"
    "0.005 0.040 0.05"
    "0.006 0.040 0.05"
    "0.0075 0.040 0.05"
    "0.005 0.0425 0.05"
    "0.0075 0.0425 0.05"
    "0.010 0.0425 0.05"
    "0.005 0.045 0.05"
    "0.0075 0.045 0.05"
    "0.010 0.045 0.05"
    "0.005 0.050 0.05"
    "0.0075 0.050 0.05"
  )

  for cfg in "${configs[@]}"; do
    read -r alpha beta mlm <<< "${cfg}"
    local tag
    tag=$(printf "movies_sacp_wsearch_p2_a%s_b%s_m%s_s%s" "${alpha}" "${beta}" "${mlm}" "${seed}" | tr '.' 'p')
    run_trial "${tag}" "${alpha}" "${beta}" "${mlm}" "${seed}"
  done
}

phase3() {
  # After alpha/beta are narrowed, check whether mlm is helping or hurting.
  local seed="42"
  local configs=(
    "0.005 0.040 0.03"
    "0.005 0.040 0.05"
    "0.005 0.040 0.07"
    "0.0075 0.0425 0.03"
    "0.0075 0.0425 0.05"
    "0.0075 0.0425 0.07"
    "0.010 0.045 0.03"
    "0.010 0.045 0.05"
    "0.010 0.045 0.07"
  )

  for cfg in "${configs[@]}"; do
    read -r alpha beta mlm <<< "${cfg}"
    local tag
    tag=$(printf "movies_sacp_wsearch_p3_a%s_b%s_m%s_s%s" "${alpha}" "${beta}" "${mlm}" "${seed}" | tr '.' 'p')
    run_trial "${tag}" "${alpha}" "${beta}" "${mlm}" "${seed}"
  done
}

phase4() {
  # Stability check: do not trust a single seed when gaps are small.
  local configs=(
    "0.005 0.040 0.05"
    "0.0075 0.0425 0.05"
    "0.010 0.045 0.05"
  )
  local seeds=(42 2024 3407)

  for cfg in "${configs[@]}"; do
    read -r alpha beta mlm <<< "${cfg}"
    for seed in "${seeds[@]}"; do
      local tag
      tag=$(printf "movies_sacp_wsearch_p4_a%s_b%s_m%s_s%s" "${alpha}" "${beta}" "${mlm}" "${seed}" | tr '.' 'p')
      run_trial "${tag}" "${alpha}" "${beta}" "${mlm}" "${seed}"
    done
  done
}

case "${PHASE}" in
  phase1) phase1 ;;
  phase2) phase2 ;;
  phase3) phase3 ;;
  phase4) phase4 ;;
  all)
    phase1
    phase2
    phase3
    phase4
    ;;
  *)
    echo "Unknown phase: ${PHASE}"
    echo "Usage: bash movies_sacp_weight_search.sh [phase1|phase2|phase3|phase4|all]"
    exit 1
    ;;
esac
