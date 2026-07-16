# CoLaGR README

This document is the `conda cola` workflow for the CoLaGR implementation inside the Latte repo.

CoLaGR is intended to be:

- `CoPref`: collaborative prefix-conditioned posterior
- `CoReason`: level-wise collaborative-grounded decoder state
- `CoDecode`: generation-time fusion of collaborative prior into next-code logits

It is not supposed to become `Latte++`. In this repo, CoLaGR is implemented independently from Latte's latent-token label path.

## 1. Enter The Correct Python Environment

Run everything from the repo root:

```bash
cd ~/CoLaGR/Latte-main/Latte-main
```

Force the shell onto conda `cola`:

```bash
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate cola
hash -r
which python
```

Expected:

```text
/home/cyx/miniconda3/envs/cola/bin/python
```

If you see `.venv/bin/python`, the shell is still shadowing conda. Re-activate `cola` and check again.

## 2. Install Dependencies In `cola`

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==2.2.6
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

Verify:

```bash
python -c "import sys, torch, numpy; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(numpy.__version__)"
```

If `torch` prints a warning about `Failed to initialize NumPy`, install `numpy` first as above and rerun the check.

## 2.5 Download Benchmark CSV With `aria2c`

If Hugging Face access is slow or blocked, pre-download the benchmark CSV files
from the official McAuley Lab source:

```bash
bash scripts/download_amazon2023_benchmark_aria2.sh Industrial_and_Scientific
```

General form:

```bash
bash scripts/download_amazon2023_benchmark_aria2.sh <CATEGORY> [KCORE] [SPLIT] [OUTPUT_ROOT]
```

Defaults:

- `CATEGORY=Industrial_and_Scientific`
- `KCORE=5core`
- `SPLIT=last_out_w_his`
- `OUTPUT_ROOT=benchmark`

This creates:

```text
benchmark/5core/last_out_w_his/<CATEGORY>.train.csv
benchmark/5core/last_out_w_his/<CATEGORY>.valid.csv
benchmark/5core/last_out_w_his/<CATEGORY>.test.csv
```

Use this script before running `export_sid_artifacts_latte` if the repo is
otherwise falling back to empty or incomplete benchmark files.

## 3. What Is Already Implemented

From code inspection, the current repo status is:

- implemented: `genrec/models/CoLaGR/tokenizer.py`
- implemented: `genrec/models/CoLaGR/model.py`
- implemented: `genrec/models/CoLaGR/trainer.py`
- implemented: `colagr/copref/export_sid_artifacts_latte.py`
- implemented: `colagr/teacher/export_topm_sasrec_latte.py`
- implemented: `colagr/copref/build_copref_latte.py`
- implemented: `colagr/eval/diagnostics_latte.py`
- implemented: optional `valid_prefix_trie` masking in generation
- not implemented yet: `Output-CoPref` baseline
- not implemented yet: `RandomPref / GlobalPref / Prefix-aware` ablation switch
- not implemented yet: `shared CoReason` ablation
- not implemented yet: full `CDG / CRU / intervention` diagnostics

Important code-level conclusions:

- CoLaGR tokenizer inherits from `PSIDTokenizer`, not `LatteTokenizer`
- CoLaGR labels remain clean PSID-style `[sid_1, ..., sid_m, eos]`
- CoReason tokens are appended after the PSID vocab and are not placed into labels
- decoder input is constructed as `[BOS, CoR1, y1, CoR2, y2, ...]`
- level loss is computed from hidden state at `CoR_l`, not at `y_l`
- CoDecode restricts base logits to `level_token_ids[l]` before fusion
- inference is teacher-free and rejects `coprefs` during `generate()`
- PSID / Latte original source files were not modified

## 4. Static Checks

```bash
python -m py_compile \
  genrec/models/CoLaGR/tokenizer.py \
  genrec/models/CoLaGR/model.py \
  genrec/models/CoLaGR/trainer.py \
  colagr/copref/export_sid_artifacts_latte.py \
  colagr/teacher/export_latte_sasrec_data.py \
  colagr/teacher/export_topm_sasrec_latte.py \
  colagr/copref/build_copref_latte.py \
  colagr/eval/diagnostics_latte.py \
  colagr/eval/protocol_checks.py \
  run_colagr_pipeline.py
```

```bash
python -c "from genrec.models.CoLaGR.model import CoLaGR; print('ok')"
```

```bash
python -m colagr.eval.protocol_checks
```

## 5. Baseline Sanity Commands

These are baseline-only checks. They should not route through CoLaGR logic.

PSID:

```bash
python main.py \
  --model=PSID \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans
```

Latte:

```bash
python main.py \
  --model=Latte \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans
```

## 6. Export SID Artifacts

Before this step, make sure the benchmark CSV files exist locally. On servers
with unstable Hugging Face access, run:

```bash
bash scripts/download_amazon2023_benchmark_aria2.sh Industrial_and_Scientific
```

```bash
python -m colagr.copref.export_sid_artifacts_latte \
  --model=CoLaGR \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans \
  --output_dir=artifacts/Industrial_and_Scientific/rqkmeans
```

Expected files:

```text
artifacts/Industrial_and_Scientific/rqkmeans/item2sid.json
artifacts/Industrial_and_Scientific/rqkmeans/sid2item.json
artifacts/Industrial_and_Scientific/rqkmeans/level_token_ids.pt
artifacts/Industrial_and_Scientific/rqkmeans/valid_prefix_trie.json
artifacts/Industrial_and_Scientific/rqkmeans/tokenizer_meta.json
```

## 7. Export SASRec Data

```bash
python -m colagr.teacher.export_latte_sasrec_data \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --sasrec_dataset_name=Industrial_and_Scientific \
  --output_dir=colagr/teacher
```

## 8. Train SASRec Teacher

```bash
cd colagr/teacher/llmsrec_sasrec
python main.py \
  --dataset=Industrial_and_Scientific \
  --device=0 \
  --num_epochs=200 \
  --maxlen=128 \
  --hidden_units=64 \
  --num_blocks=2 \
  --num_heads=1
cd ../../..
```

## 9. Export Teacher Top-M

```bash
python -m colagr.teacher.export_topm_sasrec_latte \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --checkpoint=<sasrec_ckpt> \
  --output_dir=artifacts/Industrial_and_Scientific/teacher \
  --top_m=200 \
  --splits=train,val,test \
  --limit_samples=1000
```

For full export on the second GPU, use the batched full-item scorer. This is
much faster than the legacy per-sample path and uses GPU memory more effectively:

```bash
cd ~/CoLaGR/Latte-main/Latte-main
conda activate cola
hash -r

CUDA_VISIBLE_DEVICES=1 python -u -m colagr.teacher.export_topm_sasrec_latte \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --checkpoint=colagr/teacher/llmsrec_sasrec/Industrial_and_Scientific/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth \
  --output_dir=artifacts/Industrial_and_Scientific/teacher \
  --top_m=200 \
  --splits=train \
  --device=cuda:0 \
  --export_batch_size=512
```

When `CUDA_VISIBLE_DEVICES=1` is set, the physical second GPU is exposed to the
process as `cuda:0`, so keep `--device=cuda:0`.

To run validation and test after train:

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m colagr.teacher.export_topm_sasrec_latte \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --checkpoint=colagr/teacher/llmsrec_sasrec/Industrial_and_Scientific/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth \
  --output_dir=artifacts/Industrial_and_Scientific/teacher \
  --top_m=200 \
  --splits=val,test \
  --device=cuda:0 \
  --export_batch_size=512
```

## 10. Build CoPref

The default output is the fast structured tensor cache:

```python
{
  "format": "colagr_copref_tensor_v1",
  "sample_id": Tensor[N],
  "target_sid": Tensor[N, L],
  "copref": [Tensor[N, K_1], ..., Tensor[N, K_L]],
  "prefix_count": Tensor[N, L],
  "entropy": Tensor[N, L],
  "target_code_rank": Tensor[N, L],
}
```

This avoids the old `list[dict]` cache with many small tensors and lets the
trainer fetch CoPref by `sample_id` with one `index_select` per level.

```bash
CUDA_VISIBLE_DEVICES=1 python -u -m colagr.copref.build_copref_latte \
  --artifacts_dir=artifacts/Industrial_and_Scientific/rqkmeans \
  --teacher_dir=artifacts/Industrial_and_Scientific/teacher \
  --output_dir=artifacts/Industrial_and_Scientific/copref \
  --temp=2.0 \
  --tau=5.0 \
  --min_sid_coverage=0.99 \
  --device=cuda \
  --batch_size=8192 \
  --output_format=tensor
```

If GPU memory is still loose, try `--batch_size=16384`. If you need the old
cache layout for debugging, add `--output_format=records`.

## 11. CoPref Diagnostics

```bash
python -m colagr.eval.diagnostics_latte \
  --copref_path=artifacts/Industrial_and_Scientific/copref/copref_train.pt \
  --sid_artifacts_dir=artifacts/Industrial_and_Scientific/rqkmeans \
  --limit=1000
```

You want to see:

- `copref_sum_*` close to `1.0`
- `nan_count = 0`
- `negative_count = 0`
- `target_code_mean_rank < random_rank_baseline`

## 12. CoLaGR Smoke Commands

CoReason-only:

```bash
python main.py \
  --model=CoLaGR \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans \
  --level_token_ids_path=artifacts/Industrial_and_Scientific/rqkmeans/level_token_ids.pt \
  --valid_prefix_trie_path=artifacts/Industrial_and_Scientific/rqkmeans/valid_prefix_trie.json \
  --copref_train_path=artifacts/Industrial_and_Scientific/copref/copref_train.pt \
  --copref_val_path=artifacts/Industrial_and_Scientific/copref/copref_val.pt \
  --copref_test_path=artifacts/Industrial_and_Scientific/copref/copref_test.pt \
  --use_coreason=True \
  --use_copref_loss=False \
  --use_cofuse=False \
  --lambda_c=0.0 \
  --num_beams=1 \
  --steps=100
```

AuxOnly:

```bash
python main.py \
  --model=CoLaGR \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans \
  --level_token_ids_path=artifacts/Industrial_and_Scientific/rqkmeans/level_token_ids.pt \
  --valid_prefix_trie_path=artifacts/Industrial_and_Scientific/rqkmeans/valid_prefix_trie.json \
  --copref_train_path=artifacts/Industrial_and_Scientific/copref/copref_train.pt \
  --copref_val_path=artifacts/Industrial_and_Scientific/copref/copref_val.pt \
  --copref_test_path=artifacts/Industrial_and_Scientific/copref/copref_test.pt \
  --use_coreason=True \
  --use_copref_loss=True \
  --use_cofuse=False \
  --lambda_c=0.05 \
  --num_beams=1 \
  --steps=100
```

FuseOnly:

```bash
python main.py \
  --model=CoLaGR \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans \
  --level_token_ids_path=artifacts/Industrial_and_Scientific/rqkmeans/level_token_ids.pt \
  --valid_prefix_trie_path=artifacts/Industrial_and_Scientific/rqkmeans/valid_prefix_trie.json \
  --copref_train_path=artifacts/Industrial_and_Scientific/copref/copref_train.pt \
  --copref_val_path=artifacts/Industrial_and_Scientific/copref/copref_val.pt \
  --copref_test_path=artifacts/Industrial_and_Scientific/copref/copref_test.pt \
  --use_coreason=True \
  --use_copref_loss=False \
  --use_cofuse=True \
  --lambda_c=0.0 \
  --fuse_gate_init=-2.0 \
  --num_beams=1 \
  --steps=100
```

Full CoLaGR:

```bash
python main.py \
  --model=CoLaGR \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans \
  --level_token_ids_path=artifacts/Industrial_and_Scientific/rqkmeans/level_token_ids.pt \
  --valid_prefix_trie_path=artifacts/Industrial_and_Scientific/rqkmeans/valid_prefix_trie.json \
  --copref_train_path=artifacts/Industrial_and_Scientific/copref/copref_train.pt \
  --copref_val_path=artifacts/Industrial_and_Scientific/copref/copref_val.pt \
  --copref_test_path=artifacts/Industrial_and_Scientific/copref/copref_test.pt \
  --use_coreason=True \
  --use_copref_loss=True \
  --use_cofuse=True \
  --lambda_c=0.05 \
  --fuse_gate_init=-2.0 \
  --use_prefix_trie=True \
  --num_beams=1 \
  --steps=100
```

## 13. One-Command Wrapper

```bash
python run_colagr_pipeline.py \
  --dataset=AmazonReviews2023 \
  --category=Industrial_and_Scientific \
  --vq_method=rqkmeans \
  --sasrec_checkpoint=<sasrec_ckpt> \
  --top_m=200 \
  --sasrec_device=0 \
  --steps=100 \
  --num_beams=1
```

## 14. Current Gaps

These are the main things still missing from the codebase:

- `use_coreason` is present in config, but the model currently always uses CoReason-style decoding and does not branch on that flag
- `copref_val_path` and `copref_test_path` are not loaded by default for ordinary evaluation; they are only loaded when `load_eval_copref_diagnostics=True`
- `Output-CoPref` / `TCA-style` baseline is missing
- `RandomPref / GlobalPref` ablation modes are missing
- `shared CoReason` ablation is missing
- full `CDG / CRU / intervention / trajectory` diagnostics are missing

## 15. Hard Method Boundary

Use this repo with the following interpretation:

- `PSID` is the clean SID backbone
- `Latte` is a separate latent-token baseline
- `CoLaGR` is collaborative grounding on top of clean PSID decoding
- Latte latent-token label mechanisms should not be described as CoLaGR components
