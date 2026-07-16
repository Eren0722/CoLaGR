#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_PATH=${LLM_PATH:-/home/cyx/models/llama3_3b}
DEVICE=${DEVICE:-0}

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

run_full_table() {
  local short="$1"
  local dataset=""
  local pure_save=""
  local echo_save=""
  local pure_ckpt=""
  local echo_ckpt=""
  local out=""

  case "${short}" in
    movies)
      dataset=Movies_and_TV
      pure_save=movies_pure_sasrec_si_5090
      echo_save=movies_si_5090
      pure_ckpt=./SeqRec/sasrec/${dataset}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth
      echo_ckpt=./SeqRec/sasrec/${dataset}/movies_sa_teacher/model_metric_best.pth
      out=./analysis/rq4_movies_full
      ;;
    scientific)
      dataset=Industrial_and_Scientific
      pure_save=scientific_pure_sasrec_si_5090
      echo_save=scientific_si_5090
      pure_ckpt=./SeqRec/sasrec/${dataset}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth
      echo_ckpt=./SeqRec/sasrec/${dataset}/scientific_sa_teacher/model_metric_best.pth
      out=./analysis/rq4_scientific_full
      ;;
    electronics)
      dataset=Electronics
      pure_save=electronics_pure_sasrec_si_5090
      echo_save=electronics_si_5090
      pure_ckpt=./SeqRec/sasrec/${dataset}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth
      echo_ckpt=./SeqRec/sasrec/${dataset}/electronics_sa_teacher/model_metric_best.pth
      out=./analysis/rq4_electronics_full
      ;;
    cds)
      dataset=CDs_and_Vinyl
      pure_save=cds_pure_sasrec_si_cand4_5090
      echo_save=cds_si_cand4_5090
      pure_ckpt=./SeqRec/sasrec/${dataset}/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth
      echo_ckpt=./SeqRec/sasrec/${dataset}/cds_sa_teacher/model_metric_best.pth
      out=./analysis/rq4_cds_full
      ;;
    *)
      echo "Unknown dataset key: ${short}" >&2
      exit 1
      ;;
  esac

  python experiments/cds_sshg_full_table.py \
    --dataset "${dataset}" \
    --output_dir "${out}" \
    --pure_save_dir "${pure_save}" \
    --pure_teacher_ckpt "${pure_ckpt}" \
    --echo_save_dir "${echo_save}" \
    --echo_teacher_ckpt "${echo_ckpt}" \
    "${COMMON_ARGS[@]}"
}

for short in movies scientific electronics cds; do
  run_full_table "${short}"
done

python experiments/build_multidataset_sshg_paper_assets.py \
  --summary_inputs \
    Movies=./analysis/rq4_movies_full \
    Scientific=./analysis/rq4_scientific_full \
    Electronics=./analysis/rq4_electronics_full \
    CDs=./analysis/rq4_cds_full \
  --curve_inputs \
    CDs=./analysis/rq4_cds_full \
    Scientific=./analysis/rq4_scientific_full \
    Movies=./analysis/rq4_movies_full \
    Electronics=./analysis/rq4_electronics_full \
  --out_dir ./analysis/multidataset_sshg \
  --paper_figure_dir ./paper/figure \
  --figure_name multidataset_sshg_curves

echo "[done] RQ4 main-text + appendix SSHG figures written to ./paper/figure"
