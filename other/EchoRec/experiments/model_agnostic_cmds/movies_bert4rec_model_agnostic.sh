#!/usr/bin/env bash
set -euo pipefail

cd ~/EchoRec/EchoRec

export PYTHONPATH=$(pwd)
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-1800}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-1800}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TRAIN_DEVICES=${TRAIN_DEVICES:-0,1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export LLM_NAME=${LLM_NAME:-llama-3b}
export LLM_PATH=${LLM_PATH:-/home/cyx/models/llama3_3b}

echo "===== [1/4] BERT4Rec full teacher ====="
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=29613 \
  -m SeqRec.sasrec.train_sacp_sasrec \
  --dataset Movies_and_TV \
  --data_root ./SeqRec \
  --sa_asset_root ./SA_assets \
  --batch_size 256 \
  --test_batch_size 512 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --l2_emb 0.0 \
  --maxlen 128 \
  --hidden_units 64 \
  --num_blocks 2 \
  --num_heads 2 \
  --dropout_rate 0.2 \
  --inner_size 256 \
  --hidden_dropout_prob 0.2 \
  --attn_dropout_prob 0.2 \
  --hidden_act gelu \
  --bert_mask_prob 0.15 \
  --bert_rec_objective next_item \
  --recsys_backbone bert4rec \
  --sacp_preset raw_sasrec \
  --sa_similarity cos \
  --sa_repr_mode mean \
  --seed 42 \
  --device 0 \
  --save_dir ./SeqRec/bert4rec \
  --save_prefix movies_bertstyle_nextitem_model_agnostic_mean_a01_b01 \
  --sa_alpha 0.1 \
  --sa_beta 0.1 \
  --sa_mlm_probability 0.05 \
  --sa_temperature 0.2 \
  --eval_every 1 \
  --patience 10

echo "===== [2/4] BERT4Rec wo SACP teacher ====="
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=29614 \
  -m SeqRec.sasrec.train_sacp_sasrec \
  --dataset Movies_and_TV \
  --data_root ./SeqRec \
  --sa_asset_root ./SA_assets \
  --batch_size 256 \
  --test_batch_size 512 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --l2_emb 0.0 \
  --maxlen 128 \
  --hidden_units 64 \
  --num_blocks 2 \
  --num_heads 2 \
  --dropout_rate 0.2 \
  --inner_size 256 \
  --hidden_dropout_prob 0.2 \
  --attn_dropout_prob 0.2 \
  --hidden_act gelu \
  --bert_mask_prob 0.15 \
  --bert_rec_objective next_item \
  --recsys_backbone bert4rec \
  --sacp_preset raw_sasrec \
  --sa_similarity cos \
  --sa_repr_mode mean \
  --seed 42 \
  --device 0 \
  --save_dir ./SeqRec/bert4rec \
  --save_prefix movies_bertstyle_nextitem_model_agnostic_wo_sacp \
  --sa_alpha 0.0 \
  --sa_beta 0.0 \
  --sa_mlm_probability 0.05 \
  --sa_temperature 0.2 \
  --eval_every 1 \
  --patience 10

echo "===== [3/4] BERT4Rec full -> SI ====="
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=29623 \
  main.py --train --multi_gpu --world_size ${NPROC_PER_NODE} \
  --recsys bert4rec \
  --rec_pre_trained_data Movies_and_TV \
  --recsys_ckpt_path ./SeqRec/bert4rec/Movies_and_TV/movies_bertstyle_nextitem_model_agnostic_mean_a01_b01/model_metric_best.pth \
  --llm ${LLM_NAME} \
  --llm_path "${LLM_PATH}" \
  --hf_local_only \
  --hf_cache_dir "${LLM_PATH}" \
  --batch_size 20 \
  --train_candidate_num 4 \
  --candidate_chunk_size 80 \
  --batch_size_infer 10 \
  --llm_max_length 1024 \
  --eval_item_batch 32 \
  --eval_max_length 1024 \
  --eval_min_length 1024 \
  --maxlen 128 \
  --stage2_lr 1e-4 \
  --num_epochs 25 \
  --early_stop_patience 3 \
  --min_epochs_before_early_stop 10 \
  --seed 42 \
  --match_weight 1.0 \
  --save_dir movies_bert4rec_model_agnostic_full_si_5090

echo "===== [4/4] BERT4Rec wo SACP -> SI ====="
CUDA_VISIBLE_DEVICES=${TRAIN_DEVICES} torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=29624 \
  main.py --train --multi_gpu --world_size ${NPROC_PER_NODE} \
  --recsys bert4rec \
  --rec_pre_trained_data Movies_and_TV \
  --recsys_ckpt_path ./SeqRec/bert4rec/Movies_and_TV/movies_bertstyle_nextitem_model_agnostic_wo_sacp/model_metric_best.pth \
  --llm ${LLM_NAME} \
  --llm_path "${LLM_PATH}" \
  --hf_local_only \
  --hf_cache_dir "${LLM_PATH}" \
  --batch_size 20 \
  --train_candidate_num 4 \
  --candidate_chunk_size 80 \
  --batch_size_infer 10 \
  --llm_max_length 1024 \
  --eval_item_batch 32 \
  --eval_max_length 1024 \
  --eval_min_length 1024 \
  --maxlen 128 \
  --stage2_lr 1e-4 \
  --num_epochs 25 \
  --early_stop_patience 3 \
  --min_epochs_before_early_stop 10 \
  --seed 42 \
  --match_weight 1.0 \
  --save_dir movies_bert4rec_model_agnostic_wo_sacp_si_5090
