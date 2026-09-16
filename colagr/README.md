# CoLaGR Pipeline

The active method is **stepwise CoPref + CoLeaf**, implemented independently
from Latte's latent-token labels. CoLaGR uses clean PSID labels
`[sid_1, ..., sid_L, eos]`.

## Environment

Run commands from the repository root in a Python 3.12 `cola` environment.
For Blackwell GPUs, install a CUDA 12.8-compatible PyTorch build.

```bash
conda activate cola
python -m pip install numpy==2.2.6
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

## Build Assets

1. Download the benchmark CSVs.

```bash
bash scripts/download_amazon2023_benchmark_aria2.sh Industrial_and_Scientific
```

2. Export RQ-KMeans SID artifacts.

```bash
python -m colagr.copref.export_sid_artifacts_latte \
  --model=CoLaGR --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific --vq_method=rqkmeans \
  --output_dir=artifacts/Industrial_and_Scientific/rqkmeans
```

3. Train or provide a frozen SASRec teacher, then export its Top-M records.

```bash
python -m colagr.teacher.export_topm_sasrec_latte \
  --dataset=AmazonReviews2023 --category=Industrial_and_Scientific \
  --checkpoint=<sasrec_checkpoint> \
  --output_dir=artifacts/Industrial_and_Scientific/teacher \
  --top_m=200 --splits=train,val,test --device=cuda
```

4. Build the prefix-conditioned CoPref tensors.

```bash
python -m colagr.copref.build_copref_latte \
  --artifacts_dir=artifacts/Industrial_and_Scientific/rqkmeans \
  --teacher_dir=artifacts/Industrial_and_Scientific/teacher \
  --output_dir=artifacts/collaborative/Industrial_and_Scientific \
  --temp=2.0 --device=cuda --batch_size=8192 --output_format=tensor
```

The output supports `colagr_copref_tensor_v1` and
`colagr_copref_prefix_tensor_v2`. It stores one normalized branch distribution
per SID level, plus `prefix_count`, entropy, and target-code rank diagnostics.

## Generator Ablations

`run_stepwise_ablation.sh` accepts these variants:

- `base`: clean PSID decoding without CP states or collaborative KL.
- `latent_only`: CP states without collaborative KL.
- `one_shot`: one CP state for all SID levels, supervised by every level target.
- `direct`: direct KL on SID logits, without CP states.
- `stepwise`: one KL-supervised CP state before each SID action.

```bash
CATEGORY=Industrial_and_Scientific VARIANT=stepwise GPU=0 \
  bash scripts/run_stepwise_ablation.sh
```

The script expects CoPref tensors at
`artifacts/collaborative/<CATEGORY>/copref_{train,val,test}.pt`. Override
`ASSET_ROOT` or `RESULT_ROOT` to use a different location.

## CoLeaf

After a completed stepwise run, export beam candidates and rerank only that
fixed candidate set:

```bash
CATEGORY=Industrial_and_Scientific GPU=0 \
  bash scripts/run_stepwise_coleaf_eval.sh
```

The implementation uses training-only transition memory. `candidate_mode=base_only`
is required for candidate-set-preserving evaluation.

## Verification

```bash
python -m py_compile genrec/models/CoLaGR/{tokenizer,model,trainer}.py \
  colagr/copref/{export_sid_artifacts_latte,build_copref_latte}.py \
  colagr/eval/{export_beam_candidates,collab_memory_fusion_diagnostic}.py
python -m pytest tests -q
python -m colagr.eval.protocol_checks
```
