#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}" || exit 1

ENV_NAME=${ENV_NAME:-llmsrec6000ada}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
CUDA_DEVICE=${CUDA_DEVICE:-0}
LLM_NAME=${LLM_NAME:-llama-3b}
LLM_PATH=${LLM_PATH:-/home/cyx/models/llama3_3b}
CREATE_ENV=${CREATE_ENV:-1}
INSTALL_DEPS=${INSTALL_DEPS:-1}
RUN_SASREC=${RUN_SASREC:-1}
RUN_LLM=${RUN_LLM:-1}

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found"
  exit 1
fi

eval "$(conda shell.bash hook)"

if [ "${CREATE_ENV}" = "1" ]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" pip
  fi
fi

conda activate "${ENV_NAME}"

if [ "${INSTALL_DEPS}" = "1" ]; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install sentencepiece
fi

if [ ! -d "${LLM_PATH}" ]; then
  echo "[error] local model path not found: ${LLM_PATH}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONPATH="${ROOT_DIR}"
export HF_DATASETS_TRUST_REMOTE_CODE=1

run_sasrec() {
  local dataset="$1"
  echo "===== SASRec: ${dataset} ====="
  cd "${ROOT_DIR}/SeqRec/sasrec" || exit 1
  python main.py --device 0 --dataset "${dataset}"
  cd "${ROOT_DIR}" || exit 1
}

run_llmsrec() {
  local dataset="$1"
  local save_dir="$2"
  local batch_size="$3"
  echo "===== LLM-SRec: ${dataset} ====="
  cd "${ROOT_DIR}" || exit 1
  python main.py \
    --device 0 \
    --train \
    --rec_pre_trained_data "${dataset}" \
    --save_dir "${save_dir}" \
    --batch_size "${batch_size}" \
    --llm "${LLM_NAME}" \
    --llm_path "${LLM_PATH}"
}

if [ "${RUN_SASREC}" = "1" ]; then
  run_sasrec "Electronics"
  run_sasrec "Industrial_and_Scientific"
fi

if [ "${RUN_LLM}" = "1" ]; then
  run_llmsrec "Electronics" "electronics_pure_llmsrec" 16
  run_llmsrec "Industrial_and_Scientific" "scientific_pure_llmsrec" 20
fi
