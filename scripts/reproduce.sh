#!/usr/bin/env bash
# Export scored beams from the three released checkpoints and rebuild the main row.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ALPHA_GRID='0,0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.6,0.8,1,1.5,2,3,4,5,6,7,8'
DOMAINS=("$@")
if [[ $# -eq 0 ]]; then
  DOMAINS=(industrial musical video)
fi

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('A CUDA-enabled PyTorch installation and GPU are required for scored-beam export.')
PY

for slug in "${DOMAINS[@]}"; do
  case "$slug" in
    industrial) category=Industrial_and_Scientific; lambda_pref=1.75 ;;
    musical) category=Musical_Instruments; lambda_pref=1.0 ;;
    video) category=Video_Games; lambda_pref=0.75 ;;
    *) echo "Unknown domain: $slug (use industrial, musical, or video)" >&2; exit 2 ;;
  esac

  bundle="release/$slug"
  processed="cache/AmazonReviews2023/$category/processed"
  candidates="results/$slug/candidates"
  out="results/$slug/coleaf"
  test -s "$bundle/checkpoint.pth" || { echo "Missing $bundle/checkpoint.pth; run git lfs pull." >&2; exit 2; }
  test -s "$bundle/sid/level_token_ids.pt" || { echo "Missing $slug SID assets." >&2; exit 2; }
  test -s "$bundle/sid/valid_prefix_trie.json" || { echo "Missing $slug prefix trie." >&2; exit 2; }

  if [[ ! -s "$processed/all_item_seqs.json" || ! -s "$processed/id_mapping.json" || ! -s "$processed/metadata.sentence.json" ]]; then
    raw="cache/AmazonReviews2023/$category/raw/benchmark/5core/last_out_w_his"
    if [[ ! -s "$raw/$category.test.csv" ]]; then
      bash scripts/download_amazon2023_benchmark_aria2.sh "$category" 5core last_out_w_his "cache/AmazonReviews2023/$category/raw/benchmark"
    fi
    "$PYTHON" -m scripts.prepare_dataset "$category"
  fi

  mkdir -p "$processed" "$candidates" "$out"
  cp "$bundle/processed_sid.sem_ids" "$processed/sentence-t5-base_rqkmeans_3x256_psid.sem_ids"
  "$PYTHON" -m scripts.check_bundle data "$slug"

  for split in val test; do
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m colagr.eval.export_beam_candidates \
      --checkpoint="$bundle/checkpoint.pth" \
      --output="$candidates/$split.pt" --split="$split" \
      --category="$category" --vq_method=rqkmeans \
      --level_token_ids_path="$bundle/sid/level_token_ids.pt" \
      --valid_prefix_trie_path="$bundle/sid/valid_prefix_trie.json" \
      --lambda_pref="$lambda_pref" --num_beams=50 --eval_batch_size="$BATCH_SIZE" \
      --dataloader_num_workers=0 --with_scores
  done

  CATEGORY="$category" CANDIDATES="$candidates" OUT="$out" \
    ALPHA_GRID="$ALPHA_GRID" bash scripts/run_coleaf.sh
  "$PYTHON" -m scripts.check_bundle results "$slug" "$out/validation.log" "$out/test.log"
done

"$PYTHON" -m scripts.check_bundle table "${DOMAINS[@]}"
