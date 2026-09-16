#!/usr/bin/env bash
# Evaluate CoLeaf only on candidates exported by the completed stepwise model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CATEGORY="${CATEGORY:?Set CATEGORY to an AmazonReviews2023 domain.}"
GPU="${GPU:-0}"
SEED="${SEED:-2024}"
PYTHON="${PYTHON:-python}"
ASSET_ROOT="${ASSET_ROOT:-$ROOT/artifacts/collaborative}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/stepwise_ablation}"
SOURCE_LOG="$RESULT_ROOT/logs/$CATEGORY/stepwise_seed${SEED}.log"
LOG="$RESULT_ROOT/logs/$CATEGORY/coleaf_seed${SEED}.log"
CANDIDATES="$RESULT_ROOT/candidates/${CATEGORY}_stepwise_seed${SEED}_test.pt"

for path in \
  "$SOURCE_LOG" \
  "artifacts/$CATEGORY/rqkmeans/level_token_ids.pt" \
  "artifacts/$CATEGORY/rqkmeans/valid_prefix_trie.json" \
  "$ASSET_ROOT/$CATEGORY/copref_train.pt" \
  "$ASSET_ROOT/$CATEGORY/copref_val.pt" \
  "$ASSET_ROOT/$CATEGORY/copref_test.pt"; do
  test -f "$path" || { echo "Missing required input: $path" >&2; exit 1; }
done

CHECKPOINT="$(grep -oE 'Saved model checkpoint to [^[:space:]]+' "$SOURCE_LOG" | tail -n 1 | sed 's/^Saved model checkpoint to //')"
test -n "$CHECKPOINT" && test -f "$CHECKPOINT" || {
  echo "No usable stepwise checkpoint recorded in $SOURCE_LOG" >&2
  exit 1
}

mkdir -p "$(dirname "$LOG")" "$(dirname "$CANDIDATES")"
printf '%s\t%s\tcoleaf\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "$CATEGORY" "$ASSET_ROOT" "$CHECKPOINT" \
  >> "$RESULT_ROOT/run_manifest.tsv"

# Candidate generation must reconstruct the same per-level CoPref state used
# for the source checkpoint; CoLeaf then reranks this fixed candidate set.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u colagr/eval/export_beam_candidates.py \
  --checkpoint="$CHECKPOINT" \
  --output="$CANDIDATES" \
  --split=test \
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
  --eval_batch_size=128 \
  --dataloader_num_workers=4 \
  --num_beams=50 \
  --topk='[5,10,50]' \
  --lambda_pref=2.5 \
  --use_copref_module=true \
  --use_copref_loss=true \
  --grounding_anchor=cp \
  --direct_copref_alignment=false \
  --copref_reasoning_mode=per_level \
  > "$LOG" 2>&1

PYTHONPATH=. "$PYTHON" -u colagr/eval/collab_memory_fusion_diagnostic.py \
  --category="$CATEGORY" \
  --vq_method=rqkmeans \
  --base_candidates="$CANDIDATES" \
  --split=test \
  --window=5 \
  --neighbors_per_item=200 \
  --history_lengths=10 \
  --memory_ks=50 \
  --alphas=0.10 \
  --candidate_mode=base_only \
  --fusion_mode=score \
  >> "$LOG" 2>&1
