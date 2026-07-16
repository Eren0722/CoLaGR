# Latte / PSID / CoLaGR

This repository contains:

- `PSID`: purely semantic ID baseline
- `Latte`: latent-token baseline from the Latte paper
- `CoLaGR`: collaborative decoding grounding built on clean PSID labels

`CoLaGR` is implemented as an independent model under `genrec/models/CoLaGR/`. It does not reuse Latte's latent-token labels or latent-token aggregation as its main method.

## Environment

This repo can be used with `uv`, but the recommended setup here is the conda environment `cola`.

```bash
cd ~/CoLaGR/Latte-main/Latte-main

deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate cola
hash -r

which python
# expected: /home/<user>/miniconda3/envs/cola/bin/python
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==2.2.6
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Verify:

```bash
python -c "import sys, torch, numpy; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(numpy.__version__)"
```

The explicit `numpy` install is important for fresh `cola` environments. Without it, PyTorch may import with a warning like `Failed to initialize NumPy`.

For RTX 5090 / Blackwell cards, use a CUDA 12.8 PyTorch build such as
`torch==2.7.1+cu128`; older CUDA 12.1 wheels can import but fail at runtime with
`no kernel image is available for execution on the device`.

## Benchmark CSV Download

For servers where Hugging Face benchmark downloads are slow or blocked, use the
official McAuley Lab source with `aria2c` before running PSID / Latte / CoLaGR:

```bash
bash scripts/download_amazon2023_benchmark_aria2.sh Industrial_and_Scientific
```

This writes:

```text
benchmark/5core/last_out_w_his/Industrial_and_Scientific.train.csv
benchmark/5core/last_out_w_his/Industrial_and_Scientific.valid.csv
benchmark/5core/last_out_w_his/Industrial_and_Scientific.test.csv
```

## Config Order

Config files are merged in this order:

1. `genrec/default.yaml`
2. `genrec/datasets/<DATASET>/config.yaml`
3. `genrec/models/<MODEL>/config.yaml`
4. command line args

## Baselines

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

Latte-specific notes:

- Latte prepends a latent token to training labels.
- Latte inference uses latent-token-aware aggregation such as `agg_max` / `agg_sum`.
- These mechanisms belong to the Latte baseline only.

## CoLaGR

CoLaGR commands and the full collaborative pipeline are documented in [colagr/README.md](colagr/README.md).

The CoPref cache defaults to a structured tensor format
`colagr_copref_tensor_v1`, which is faster to save/load than the old
`list[dict]` cache and is read directly by `CoLaGRTrainer`.

Method boundary:

- CoLaGR labels stay `[sid_1, ..., sid_m, eos]`
- CoReason tokens are inserted only inside `forward()` / `generate()`
- CoLaGR inherits tokenizer behavior from `PSIDTokenizer`, not `LatteTokenizer`
- CoLaGR does not use Latte random latent-token labels
- CoLaGR does not use Latte latent-token aggregation as a method component

## Code Status

From code inspection, the current implementation status is:

- `CoReason`: implemented
- `CoDecode / CoFuse`: implemented
- `CoPref` builder and diagnostics: implemented
- `valid-prefix trie` masking: implemented, but optional by config
- `Output-CoPref` baseline: not implemented yet
- `RandomPref / GlobalPref / shared-CoReason` ablations: not implemented yet
- deeper paper diagnostics such as `CDG / CRU / intervention`: not implemented yet

Static checks:

```bash
python -m py_compile \
  genrec/models/CoLaGR/tokenizer.py \
  genrec/models/CoLaGR/model.py \
  genrec/models/CoLaGR/trainer.py \
  colagr/copref/export_sid_artifacts_latte.py \
  colagr/teacher/export_topm_sasrec_latte.py \
  colagr/copref/build_copref_latte.py \
  colagr/eval/diagnostics_latte.py \
  colagr/eval/protocol_checks.py

python -c "from genrec.models.CoLaGR.model import CoLaGR; print('ok')"
python -m colagr.eval.protocol_checks
```
