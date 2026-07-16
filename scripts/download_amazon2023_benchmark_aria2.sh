#!/usr/bin/env bash

set -euo pipefail

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c not found. Please install aria2 first." >&2
  echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y aria2" >&2
  exit 1
fi

CATEGORY="${1:-Industrial_and_Scientific}"
KCORE="${2:-5core}"
SPLIT="${3:-last_out_w_his}"
OUTPUT_ROOT="${4:-benchmark}"

BASE_URL="https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/${KCORE}/${SPLIT}"
OUTPUT_DIR="${OUTPUT_ROOT}/${KCORE}/${SPLIT}"

mkdir -p "${OUTPUT_DIR}"

download_one() {
  local split_name="$1"
  local gz_name="${CATEGORY}.${split_name}.csv.gz"
  local csv_name="${CATEGORY}.${split_name}.csv"
  local url="${BASE_URL}/${gz_name}"

  echo "[download] ${url}"
  rm -f "${OUTPUT_DIR}/${csv_name}"
  aria2c -x 8 -s 8 -k 1M \
    "${url}" \
    -d "${OUTPUT_DIR}" \
    -o "${gz_name}"

  echo "[extract] ${OUTPUT_DIR}/${gz_name}"
  gunzip -f "${OUTPUT_DIR}/${gz_name}"

  if [[ ! -s "${OUTPUT_DIR}/${csv_name}" ]]; then
    echo "Downloaded file is missing or empty: ${OUTPUT_DIR}/${csv_name}" >&2
    exit 1
  fi
}

download_one "train"
download_one "valid"
download_one "test"

echo "[done] Downloaded benchmark files:"
ls -lh "${OUTPUT_DIR}/${CATEGORY}."*
