#!/usr/bin/env bash
# Run one auditable generator-side ablation from the revised CoLaGR story.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CATEGORY="${CATEGORY:?Set CATEGORY to an AmazonReviews2023 domain.}"
VARIANT="${VARIANT:?Set VARIANT to base, latent_only, one_shot, direct, or stepwise.}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-150}"
EVAL_INTERVAL="${EVAL_INTERVAL:-3}"
SEED="${SEED:-2024}"
PYTHON="${PYTHON:-python}"
ASSET_ROOT="${ASSET_ROOT:-$ROOT/artifacts/collaborative}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/stepwise_ablation}"

case "$VARIANT" in
  base)
    FLAGS=(--use_copref_module=false --use_copref_loss=false --grounding_anchor=action --direct_copref_alignment=false --copref_reasoning_mode=per_level)
    ;;
  latent_only)
    FLAGS=(--use_copref_module=true --use_copref_loss=false --grounding_anchor=cp --direct_copref_alignment=false --copref_reasoning_mode=per_level)
    ;;
  one_shot)
    FLAGS=(--use_copref_module=true --use_copref_loss=true --grounding_anchor=cp --direct_copref_alignment=false --copref_reasoning_mode=one_shot)
    ;;
  direct)
    FLAGS=(--use_copref_module=false --use_copref_loss=true --grounding_anchor=action --direct_copref_alignment=true --copref_reasoning_mode=per_level)
    ;;
  stepwise)
    FLAGS=(--use_copref_module=true --use_copref_loss=true --grounding_anchor=cp --direct_copref_alignment=false --copref_reasoning_mode=per_level)
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

for path in \
  "artifacts/$CATEGORY/rqkmeans/level_token_ids.pt" \
  "artifacts/$CATEGORY/rqkmeans/valid_prefix_trie.json" \
  "$ASSET_ROOT/$CATEGORY/copref_train.pt" \
  "$ASSET_ROOT/$CATEGORY/copref_val.pt" \
  "$ASSET_ROOT/$CATEGORY/copref_test.pt"; do
  test -f "$path" || { echo "Missing required asset: $path" >&2; exit 1; }
done

LOG_DIR="$RESULT_ROOT/logs/$CATEGORY"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${VARIANT}_seed${SEED}.log"
RUN_ID="stepwise_v1_${CATEGORY}_${VARIANT}_e${EPOCHS}_s${SEED}"

# A manual capacity fill may overlap the sequential queue.  The per-run lock
# makes the later queue invocation a no-op instead of training a duplicate.
exec 9>"${LOG}.lock"
if ! flock -n 9; then
  echo "Already running elsewhere: $CATEGORY $VARIANT" >&2
  exit 0
fi
if grep -q 'Test Results' "$LOG" 2>/dev/null; then
  echo "Already completed: $CATEGORY $VARIANT" >&2
  exit 0
fi
if grep -q 'OFFLOADED_TO_SERVER2' "$LOG" 2>/dev/null; then
  echo "Offloaded to the second server: $CATEGORY $VARIANT" >&2
  exit 0
fi

printf '%s\t%s\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "$CATEGORY" "$VARIANT" "$ASSET_ROOT" "$RUN_ID" \
  >> "$RESULT_ROOT/run_manifest.tsv"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u main.py \
  --model=CoLaGR \
  --dataset=AmazonReviews2023 \
  --category="$CATEGORY" \
  --config_file=genrec/models/CoLaGR/config.yaml \
  --vq_method=rqkmeans \
  --level_token_ids_path="artifacts/$CATEGORY/rqkmeans/level_token_ids.pt" \
  --valid_prefix_trie_path="artifacts/$CATEGORY/rqkmeans/valid_prefix_trie.json" \
  --copref_train_path="$ASSET_ROOT/$CATEGORY/copref_train.pt" \
  --copref_val_path="$ASSET_ROOT/$CATEGORY/copref_val.pt" \
  --copref_test_path="$ASSET_ROOT/$CATEGORY/copref_test.pt" \
  --num_train_epochs="$EPOCHS" \
  --eval_interval="$EVAL_INTERVAL" \
  --patience=999 \
  --train_batch_size=256 \
  --eval_batch_size=128 \
  --dataloader_num_workers=4 \
  --preload_copref_to_device=true \
  --lambda_pref=2.5 \
  --use_coleaf=false \
  --rand_seed="$SEED" \
  --lr=0.003 \
  --topk='[5,10]' \
  --run_id="$RUN_ID" \
  "${FLAGS[@]}" \
  2>&1 | tee "$LOG"
