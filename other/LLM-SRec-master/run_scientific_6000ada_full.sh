#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CYX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ECHOREC_DIR="${ECHOREC_DIR:-${CYX_ROOT}/EchoRec}"
ENV_YML="${ENV_YML:-${ECHOREC_DIR}/eren-env.yml}"

ENV_NAME="${ENV_NAME:-eren}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET="${DATASET:-Industrial_and_Scientific}"
LLM_NAME="${LLM_NAME:-llama-3b}"
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.2-3B-Instruct}"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/llama3_3b}"
SAVE_DIR="${SAVE_DIR:-scientific_pure_llmsrec_6000ada}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
WORK_DIR="${WORK_DIR:-${HOME}/.cache/llmsrec_6000ada}"
RUN_ENV_SETUP="${RUN_ENV_SETUP:-1}"
RUN_MODEL_DOWNLOAD="${RUN_MODEL_DOWNLOAD:-1}"
RUN_SASREC="${RUN_SASREC:-1}"
RUN_LLM="${RUN_LLM:-1}"

status_ok=1

log() {
  echo "[info] $*"
}

warn() {
  echo "[warn] $*"
}

fail() {
  echo "[error] $*"
  status_ok=0
}

if ! command -v conda >/dev/null 2>&1; then
  fail "conda not found"
fi

if ! eval "$(conda shell.bash hook)" 2>/dev/null; then
  fail "failed to initialize conda shell hook"
fi

export HF_ENDPOINT
export HF_HOME
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-1800}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "${WORK_DIR}" 2>/dev/null || true

if [ -z "${HF_TOKEN:-}" ]; then
  read -rsp "Enter Hugging Face token: " HF_TOKEN
  echo
fi

if [ -z "${HF_TOKEN:-}" ]; then
  fail "HF token is empty"
fi

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

check_repo_writable() {
  local paths=(
    "${SCRIPT_DIR}"
    "${SCRIPT_DIR}/SeqRec"
    "${SCRIPT_DIR}/SeqRec/sasrec"
    "${SCRIPT_DIR}/models"
  )
  for path in "${paths[@]}"; do
    if [ -e "${path}" ] && [ ! -w "${path}" ]; then
      fail "repo path is not writable: ${path}"
      echo "[hint] copy the repo to a writable directory or run: chmod -R u+w '${SCRIPT_DIR}'"
      return 1
    fi
  done
  return 0
}

sanitize_env_file() {
  local src="$1"
  local dst="$2"
  SANITIZE_SRC="${src}" SANITIZE_DST="${dst}" python - <<'PY'
import os
from pathlib import Path

src = Path(os.environ["SANITIZE_SRC"])
dst = Path(os.environ["SANITIZE_DST"])

drop_prefixes = (
    "cuda-",
    "libcublas",
    "libcufft",
    "libcufile",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libnpp",
    "libnvfatbin",
    "libnvjitlink",
    "libnvjpeg",
    "nsight-compute",
    "gds-tools",
)

lines = src.read_text(encoding="utf-8").splitlines()
kept = []
for line in lines:
    if line.startswith("  - ") and not line.startswith("      - "):
        pkg = line[4:].strip()
        name = pkg.split("=", 1)[0].strip()
        if any(name.startswith(prefix) for prefix in drop_prefixes):
            continue
    kept.append(line)

dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(dst)
PY
}

setup_env() {
  if [ "${RUN_ENV_SETUP}" != "1" ]; then
    log "skip env setup"
    return 0
  fi
  if [ ! -f "${ENV_YML}" ]; then
    fail "environment file not found: ${ENV_YML}"
    return 1
  fi
  log "updating conda env '${ENV_NAME}' from ${ENV_YML}"
  if ! conda env update -n "${ENV_NAME}" -f "${ENV_YML}" --prune; then
    local sanitized_yml="${WORK_DIR}/eren-env.6000ada.sanitized.yml"
    warn "original env solve failed, retrying with sanitized env file"
    sanitize_env_file "${ENV_YML}" "${sanitized_yml}" || return 1
    log "updating conda env '${ENV_NAME}' from ${sanitized_yml}"
    conda env update -n "${ENV_NAME}" -f "${sanitized_yml}" --prune || return 1
  fi
  conda activate "${ENV_NAME}" || return 1
  python -m pip install --upgrade pip || return 1
  python -m pip install -r "${SCRIPT_DIR}/requirements.txt" || return 1
  python -m pip install sentencepiece || return 1
}

download_model() {
  if [ "${RUN_MODEL_DOWNLOAD}" != "1" ]; then
    log "skip model download"
    return 0
  fi
  conda activate "${ENV_NAME}" || return 1
  mkdir -p "${MODEL_DIR}" || return 1
  log "downloading ${MODEL_REPO} from ${HF_ENDPOINT} to ${MODEL_DIR}"
  MODEL_REPO="${MODEL_REPO}" MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
token = os.environ["HF_TOKEN"]
endpoint = os.environ["HF_ENDPOINT"]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
    endpoint=endpoint,
    local_dir_use_symlinks=False,
)
print(local_dir)
PY
}

run_sasrec() {
  if [ "${RUN_SASREC}" != "1" ]; then
    log "skip SASRec"
    return 0
  fi
  conda activate "${ENV_NAME}" || return 1
  cd "${SCRIPT_DIR}/SeqRec/sasrec" || return 1
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  log "running SASRec teacher for ${DATASET}"
  python main.py --device 0 --dataset "${DATASET}"
}

run_llm() {
  if [ "${RUN_LLM}" != "1" ]; then
    log "skip LLM-SRec"
    return 0
  fi
  conda activate "${ENV_NAME}" || return 1
  cd "${SCRIPT_DIR}" || return 1
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
  log "running LLM-SRec for ${DATASET}"
  python main.py \
    --device 0 \
    --train \
    --rec_pre_trained_data "${DATASET}" \
    --save_dir "${SAVE_DIR}" \
    --batch_size 20 \
    --llm "${LLM_NAME}" \
    --llm_path "${MODEL_DIR}"
}

setup_env || fail "environment setup failed"
check_repo_writable || fail "repo is not writable for preprocessing/checkpoints"
download_model || fail "model download failed"
run_sasrec || fail "SASRec stage failed"
run_llm || fail "LLM-SRec stage failed"

if [ "${status_ok}" = "1" ]; then
  echo "[done] scientific pipeline finished"
else
  echo "[done] scientific pipeline finished with errors"
fi
