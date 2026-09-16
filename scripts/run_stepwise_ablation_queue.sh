#!/usr/bin/env bash
# Keep three jobs resident on one GPU while the six-card worker is unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/stepwise_ablation}"

run_worker() {
  local worker="$1"
  shift
  for task in "$@"; do
    local category="${task%%:*}"
    local variant="${task##*:}"
    printf '[%s] worker=%s start %s %s\n' "$(date --iso-8601=seconds)" "$worker" "$category" "$variant" \
      | tee -a "$RESULT_ROOT/queue.log"
    GPU="$GPU" CATEGORY="$category" VARIANT="$variant" PYTHON="$PYTHON" \
      RESULT_ROOT="$RESULT_ROOT" bash "$ROOT/scripts/run_stepwise_ablation.sh" \
      >> "$RESULT_ROOT/queue.log" 2>&1
  done
}

mkdir -p "$RESULT_ROOT"

# Each worker interleaves domains so a single failed domain cannot delay all
# evidence for a mechanism. The five variants form the revised main ablation.
run_worker A \
  Industrial_and_Scientific:base \
  Industrial_and_Scientific:one_shot \
  Industrial_and_Scientific:stepwise \
  Musical_Instruments:latent_only \
  Musical_Instruments:direct &
run_worker B \
  Industrial_and_Scientific:latent_only \
  Industrial_and_Scientific:direct \
  Musical_Instruments:base \
  Musical_Instruments:one_shot \
  Musical_Instruments:stepwise &
run_worker C \
  Video_Games:base \
  Video_Games:latent_only \
  Video_Games:one_shot \
  Video_Games:direct \
  Video_Games:stepwise &
wait
