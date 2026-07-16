#!/bin/bash
# Latte Model Quick Start Training Script
# 
# This script runs a single training for Latte model with fixed hyperparameters,
# followed by inference with num_beams=500.
#
# Usage:
#   bash train_latte.sh [category] [gpu_id] [n_latent_tokens] [vq_method] [aggregation_method] [lr]
#
# Arguments:
#   category: Dataset category (default: Industrial_and_Scientific)
#   gpu_id: GPU ID to use (default: 0)
#   n_latent_tokens: Number of latent tokens (default: 8)
#   vq_method: VQ method to use (default: rqkmeans)
#   aggregation_method: Aggregation method (default: agg_max)
#   lr: Learning rate (default: 3e-3)
#
# Examples:
#   bash train_latte.sh Industrial_and_Scientific 0 8 rqkmeans agg_max 3e-3
#   bash train_latte.sh Video_Games 1 16 rqkmeans agg_sum 3e-3

category=${1:-Industrial_and_Scientific}
gpu_id=${2:-0}
n_latent_tokens=${3:-8}
vq_method=${4:-rqkmeans}
aggregation_method=${5:-agg_max}
lr=${6:-3e-3}

echo "=========================================="
echo "Latte Model Quick Start Training"
echo "=========================================="
echo "Category: ${category}"
echo "GPU ID: ${gpu_id}"
echo "Learning Rate: ${lr}"
echo "n_latent_tokens: ${n_latent_tokens}"
echo "vq_method: ${vq_method}"
echo "aggregation_method: ${aggregation_method}"
echo "=========================================="
echo ""

# Run the Latte training script
uv run python train_latte.py \
    --category=${category} \
    --gpu_id=${gpu_id} \
    --n_latent_tokens=${n_latent_tokens} \
    --vq_method=${vq_method} \
    --aggregation_method=${aggregation_method} \
    --lr=${lr}
