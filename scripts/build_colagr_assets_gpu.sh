#!/usr/bin/env bash
set -euo pipefail

# Canonical asset builder for the stepwise-CoPref mainline.
CATEGORY="${1:?usage: $0 CATEGORY [GPU]}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARTIFACTS="$ROOT/artifacts/$CATEGORY"
SID_DIR="$ARTIFACTS/rqkmeans"
OUT_DIR="$ROOT/artifacts/collaborative/$CATEGORY"
PYTHON="${PYTHON:-/home/cyx/miniconda3/envs/cola/bin/python}"

mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" -m colagr.copref.build_copref_latte \
  --artifacts_dir "$SID_DIR" \
  --teacher_dir "$ARTIFACTS/teacher" \
  --output_dir "$OUT_DIR" \
  --device cuda --batch_size 8192 \
  --output_format tensor --quiet_progress

echo "GPU assets ready: $CATEGORY -> $OUT_DIR"
