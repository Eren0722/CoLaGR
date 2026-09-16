# CoLaGR

CoLaGR is a sequential recommendation framework built on clean hierarchical
Semantic IDs (SIDs). The submitted mainline has two stages:

- **Stepwise CoPref.** Before each SID decision, a collaborative preference
  state is aligned to a level-matched, prefix-conditioned distribution derived
  from a frozen sequential collaborative-filtering teacher.
- **CoLeaf.** After teacher-free beam decoding, training-only transition memory
  reranks the same generated candidate set at the leaf/item level.

The generator input is `[BOS, CP_1, SID_1, ..., CP_L, SID_L]`. CoPref is a
training-only signal and is never supplied during generation. `one_shot` and
`direct_copref_alignment` are controlled ablations of the stepwise mechanism.

## Repository Layout

- `genrec/`: PSID, Latte baselines, and the CoLaGR generator
- `colagr/copref/`: SID and prefix-conditioned CoPref asset construction
- `colagr/teacher/`: frozen SASRec teacher utilities and Top-M export
- `colagr/eval/`: candidate export, CoLeaf, and protocol diagnostics
- `scripts/`: reproducible asset and ablation entry points
- `tests/`: CoLeaf unit tests

Generated datasets, teacher checkpoints, CoPref tensors, logs, candidates, and
paper artifacts are deliberately excluded from version control.

## Installation

Python 3.12 and a CUDA-compatible PyTorch build are recommended. For RTX 5090,
use CUDA 12.8 wheels or newer.

```bash
conda create -n cola python=3.12 -y
conda activate cola
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==2.2.6
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

## Quick Start

Download benchmark CSVs, export SID artifacts and teacher Top-M records, then
build CoPref assets. Detailed commands are in [colagr/README.md](colagr/README.md).

```bash
bash scripts/download_amazon2023_benchmark_aria2.sh Industrial_and_Scientific
bash scripts/build_colagr_assets_gpu.sh Industrial_and_Scientific 0
```

Run a generator-side ablation:

```bash
CATEGORY=Industrial_and_Scientific VARIANT=stepwise GPU=0 \
  bash scripts/run_stepwise_ablation.sh
```

Run checks:

```bash
python -m py_compile genrec/models/CoLaGR/{tokenizer,model,trainer}.py \
  colagr/copref/{export_sid_artifacts_latte,build_copref_latte}.py \
  colagr/eval/{export_beam_candidates,collab_memory_fusion_diagnostic}.py
python -m pytest tests -q
python -m colagr.eval.protocol_checks
```
