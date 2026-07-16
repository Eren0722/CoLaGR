# APAO: Adaptive Prefix-Aware Optimization for Generative Recommendation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20439389.svg)](https://doi.org/10.5281/zenodo.20439389)

This repository provides the official implementation for our KDD'26 paper, **Adaptive Prefix-Aware Optimization (APAO)**, a training-side framework for generative sequential recommendation under beam-search
decoding.

APAO addresses the training-inference inconsistency caused by teacher-forcing training and pruning-based beam-search inference by introducing prefix-level optimization objectives that
better align model training with the progressive pruning behavior of beam search.

The codebase includes implementations of APAO-Pointwise and APAO-Pairwise, representative T5-based encoder-decoder and Llama-based decoder-only backbones, processed benchmark datasets, tokenizer
  files, and scripts for reproducing the reported experimental results on Office, Grocery, Beauty, and Yelp.

## Environment Setup

1. Create a new Conda environment:

```bash
conda create -n apao python=3.12.11
```

2. Activate the environment:

```bash
conda activate apao
```

3. Install [Pytorch](https://pytorch.org/get-started/locally/) and other required dependencies via `pip`:

```bash
# Take CUDA 12.6 as an example, you can change it to your desired version
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
# Install other dependencies
pip install -r requirements.txt
```
**Note**: Ensure that the version of GCC/G++ is >= 9.0.0.

## Training and Evaluation with APAO

Run the scripts

```bash
bash scripts/run_apao_pointwise.sh
bash scripts/run_apao_pairwise.sh
```

Each script includes two parts: a hyperparameter search block and a reproduction block with the optimal hyperparameters used in the paper.

Example Command:

```bash
export PYTHONPATH=.:./src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET=Office  # Office, Grocery, Beauty, Yelp
method_name=T5_APAO_Pointwise  # T5_APAO_Pointwise, Llama_APAO_Pointwise
SEED=42
beam_size=20
test_bsz=512

beta=0.3
eta=1e-4

export WANDB_PROJECT="${DATASET}_${method_name}"
export WANDB_RUN_NAME="${method_name}_seed${SEED}_beta${beta}_eta${eta}"
OUTPUT_DIR=./ckpt/${DATASET}_${method_name}/${WANDB_RUN_NAME}/
RESULTS_FILE=./results/${DATASET}_${method_name}/$WANDB_RUN_NAME/

# Training
python src/train/train_apao.py \
    --output_dir $OUTPUT_DIR \
    --seed $SEED \
    --dataset $DATASET \
    --wandb_run_name $WANDB_RUN_NAME \
    --method_name $method_name \
    --beta $beta \
    --eta_adapt $eta

# Testing
python src/test/test.py \
    --ckpt_path $OUTPUT_DIR \
    --seed $SEED \
    --dataset $DATASET \
    --method_name $method_name \
    --results_file $RESULTS_FILE \
    --test_batch_size $test_bsz \
    --num_beams $beam_size \
    --is_constrained_beam_search
```


## Training and Evaluation with Baselines
Run the scripts

```bash
bash scripts/run_ce.sh
bash scripts/run_msl.sh
```

The baseline scripts follow the same structure: the first block can be used for hyperparameter tuning, and the reproduction block lists the optimal settings for the reported results.

## Citation

The archived KDD26 release of this repository is available on Zenodo:
```bibtex
@software{yuanqing_yu_2026_20439389,
  author       = {Yuanqing Yu},
  title        = {yuyq18/APAO: KDD26},
  month        = may,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {KDD26},
  doi          = {10.5281/zenodo.20439389},
  url          = {https://doi.org/10.5281/zenodo.20439389}
}
```
