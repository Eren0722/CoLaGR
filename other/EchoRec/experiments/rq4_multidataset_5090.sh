#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_PATH="${LLM_PATH:-/home/cyx/models/llama3_3b}"
DEVICE="${DEVICE:-0}"

COMMON_ARGS=(
  --device "${DEVICE}"
  --llm llama-3b
  --llm_path "${LLM_PATH}"
  --hf_local_only
  --hf_cache_dir "${LLM_PATH}"
  --batch_size_infer 8
  --eval_item_batch 32
  --eval_max_length 1024
  --eval_min_length 1024
  --llm_max_length 1024
  --inference_chunk_size 8
  --sample_size 2000
  --seeds 42 52 62
  --neighbor_ks 10 20 30 40 50
  --transfer_k 20
)

run_one() {
  local display_name="$1"
  local dataset="$2"
  local pure_save="$3"
  local echo_save="$4"
  local echo_teacher="$5"
  local out_dir="$6"

  python experiments/cds_sshg_full_table.py \
    --dataset "${dataset}" \
    --output_dir "${out_dir}" \
    --pure_save_dir "${pure_save}" \
    --pure_teacher_ckpt "./SeqRec/sasrec/${dataset}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth" \
    --echo_save_dir "${echo_save}" \
    --echo_teacher_ckpt "${echo_teacher}" \
    "${COMMON_ARGS[@]}"
}

run_one "Movies" "Movies_and_TV" \
  "movies_pure_sasrec_si_5090" \
  "movies_si_5090" \
  "./SeqRec/sasrec/Movies_and_TV/movies_sa_teacher/model_metric_best.pth" \
  "./analysis/rq4_movies_full"

run_one "Scientific" "Industrial_and_Scientific" \
  "scientific_pure_sasrec_si_5090" \
  "scientific_si_5090" \
  "./SeqRec/sasrec/Industrial_and_Scientific/scientific_sa_teacher/model_metric_best.pth" \
  "./analysis/rq4_scientific_full"

run_one "Electronics" "Electronics" \
  "electronics_pure_sasrec_si_5090" \
  "electronics_si_5090" \
  "./SeqRec/sasrec/Electronics/electronics_sa_teacher/model_metric_best.pth" \
  "./analysis/rq4_electronics_full"

run_one "CDs" "CDs_and_Vinyl" \
  "cds_pure_sasrec_si_cand4_5090" \
  "cds_si_cand4_5090" \
  "./SeqRec/sasrec/CDs_and_Vinyl/cds_sa_teacher/model_metric_best.pth" \
  "./analysis/rq4_cds_full"

python experiments/build_multidataset_sshg_paper_assets.py \
  --summary_inputs \
    "Movies=./analysis/rq4_movies_full" \
    "Scientific=./analysis/rq4_scientific_full" \
    "Electronics=./analysis/rq4_electronics_full" \
    "CDs=./analysis/rq4_cds_full" \
  --curve_inputs \
    "Movies=./analysis/rq4_movies_full" \
    "Scientific=./analysis/rq4_scientific_full" \
    "Electronics=./analysis/rq4_electronics_full" \
    "CDs=./analysis/rq4_cds_full" \
  --out_dir "./analysis/multidataset_sshg" \
  --paper_figure_dir "./paper/figure" \
  --figure_name "multidataset_sshg_curves"

echo "[done] RQ4 multi-dataset outputs are in ./analysis/multidataset_sshg and ./paper/figure"
