#!/usr/bin/env bash
# Main CoLaGR-G and CoLeaf protocol; artifacts stay outside Git.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAGE="${1:?Choose prepare, train, beam, or coleaf}"
CATEGORY="${CATEGORY:?Set CATEGORY to one of the three benchmark domains}"
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-AmazonReviews2023}"
SEED="${SEED:-2024}"
EPOCHS="${EPOCHS:-150}"
GPU="${GPU:-0}"
TOP_M="${TOP_M:-200}"
ASSETS="${ASSETS:-artifacts/$CATEGORY}"
SID="$ASSETS/rqkmeans"
TEACHER="$ASSETS/teacher"
COPREF="$ASSETS/copref_global"
CANDIDATES="${CANDIDATES:-results/$CATEGORY/candidates}"
mkdir -p "$CANDIDATES"

case "$STAGE" in
  prepare)
    : "${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to a trained SASRec .pth file}"
    "$PYTHON" colagr/copref/export_sid_artifacts_latte.py \
      --model=CoLaGR --dataset="$DATASET" --category="$CATEGORY" --vq_method=rqkmeans --output_dir="$SID"
    "$PYTHON" colagr/teacher/export_topm_sasrec_latte.py \
      --dataset="$DATASET" --category="$CATEGORY" --checkpoint="$TEACHER_CHECKPOINT" \
      --output_dir="$TEACHER" --top_m="$TOP_M" --splits=train,val,test --device="cuda:$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" colagr/copref/build_copref_latte.py \
      --artifacts_dir="$SID" --teacher_dir="$TEACHER" --output_dir="$COPREF" --device=cuda
    ;;
  train)
    : "${LAMBDA_PREF:?Set validation-selected LAMBDA_PREF for this domain}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" main.py \
      --model=CoLaGR --dataset="$DATASET" --category="$CATEGORY" \
      --vq_method=rqkmeans \
      --level_token_ids_path="$SID/level_token_ids.pt" \
      --valid_prefix_trie_path="$SID/valid_prefix_trie.json" \
      --copref_train_path="$COPREF/copref_train.pt" \
      --copref_val_path="$COPREF/copref_val.pt" \
      --copref_test_path="$COPREF/copref_test.pt" \
      --num_train_epochs="$EPOCHS" --eval_interval=3 --patience=999 \
      --train_batch_size=256 --eval_batch_size=128 --dataloader_num_workers=4 \
      --preload_copref_to_device=true --lambda_pref="$LAMBDA_PREF" \
      --lambda_reason=0.0 --use_coreason=false --use_coleaf=false \
      --use_copref_module=true --use_copref_loss=true --grounding_anchor=cp \
      --direct_copref_alignment=false --copref_reasoning_mode=per_level \
      --rand_seed="$SEED" --lr=0.003 --topk='[5,10]' \
      --run_id="colagr_g_${CATEGORY}_seed${SEED}"
    ;;
  beam)
    : "${LAMBDA_PREF:?Set LAMBDA_PREF used to train the checkpoint}"
    : "${GENERATOR_CHECKPOINT:?Set GENERATOR_CHECKPOINT to the validation-selected .pth file}"
    for split in val test; do
      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" colagr/eval/export_beam_candidates.py \
        --checkpoint="$GENERATOR_CHECKPOINT" --output="$CANDIDATES/$split.pt" \
        --split="$split" --dataset="$DATASET" --category="$CATEGORY" \
        --level_token_ids_path="$SID/level_token_ids.pt" \
        --valid_prefix_trie_path="$SID/valid_prefix_trie.json" \
        --copref_train_path="$COPREF/copref_train.pt" \
        --copref_val_path="$COPREF/copref_val.pt" \
        --copref_test_path="$COPREF/copref_test.pt" \
        --lambda_pref="$LAMBDA_PREF" --num_beams=50 --with_scores
    done
    ;;
  coleaf)
    bash scripts/run_coleaf.sh
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
