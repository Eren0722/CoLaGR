import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def env_value(name, default):
    return os.environ.get(name, default)


DATASET = env_value("DATASET", "CDs_and_Vinyl")
LLM_NAME = env_value("LLM_NAME", "llama-3b")
LLM_PATH = env_value("LLM_PATH", "")
TRAIN_DEVICES = env_value("TRAIN_DEVICES", "0,1")
NPROC_PER_NODE = env_value("NPROC_PER_NODE", "2")
DATA_ROOT = env_value("DATA_ROOT", "./datasets")
ASSET_ROOT = env_value("ASSET_ROOT", "./SA_assets")
TEACHER_BACKBONE = env_value("TEACHER_BACKBONE", "bert4rec")
TEACHER_PREFIX = env_value("TEACHER_PREFIX", "cds_bertstyle_nextitem_full")
SI_SAVE_DIR = env_value("SI_SAVE_DIR", "cds_bert4rec_full_si_5090")


def merged_env():
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else str(ROOT) + os.pathsep + existing_pythonpath
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")
    env.setdefault("HF_HUB_ETAG_TIMEOUT", "1800")
    env.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("NCCL_TIMEOUT", "7200")
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env["CUDA_VISIBLE_DEVICES"] = TRAIN_DEVICES
    env["ECHOREC_DATA_ROOT"] = str(ROOT / DATA_ROOT)
    return env


def run_step(name, command, logfile):
    log_path = ROOT / "logs" / logfile
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n◆ {name}")
    print(f"  log: {log_path}")
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=merged_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
        status = process.wait()
    if status != 0:
        raise SystemExit(f"{name} failed with exit code {status}. See {log_path}")


def dataset_ready():
    base = ROOT / DATA_ROOT / DATASET
    required = [
        base / f"{DATASET}_train.txt",
        base / f"{DATASET}_valid.txt",
        base / f"{DATASET}_test.txt",
        base / "text_name_dict.json.gz",
    ]
    return all(path.exists() for path in required)


def semantic_assets_ready():
    base = ROOT / ASSET_ROOT / DATASET
    required = [
        base / "item_semantic_embeddings.pt",
        base / "user_semantic_embeddings.pt",
        base / "seq_keys_to_int.pkl",
    ]
    return all(path.exists() for path in required)


def teacher_checkpoint():
    return ROOT / "SeqRec" / TEACHER_BACKBONE / DATASET / TEACHER_PREFIX / "model_metric_best.pth"


def main():
    if not LLM_PATH:
        raise SystemExit("Set LLM_PATH to a local LLaMA-3B-compatible checkpoint.")

    print("EchoRec release runner")
    print(f"dataset={DATASET}")
    print(f"llm={LLM_NAME}")
    print(f"devices={TRAIN_DEVICES}")
    print(f"teacher={TEACHER_BACKBONE}:{TEACHER_PREFIX}")
    print(f"output=models/{DATASET}/{SI_SAVE_DIR}")

    if not dataset_ready():
        run_step(
            "Preparing benchmark data",
            [
                sys.executable,
                "-c",
                f"from SeqRec.bert4rec.data_preprocess import preprocess_raw_5core; preprocess_raw_5core('{DATASET}')",
            ],
            "prepare_data.log",
        )

    if not semantic_assets_ready():
        run_step(
            "Constructing semantic assets",
            [
                sys.executable,
                "SA/generate_assets.py",
                "--dataset",
                DATASET,
                "--data_root",
                DATA_ROOT,
                "--asset_root",
                ASSET_ROOT,
                "--llm",
                LLM_NAME,
                "--llm_path",
                LLM_PATH,
                "--device",
                "cuda:0",
                "--batch_size",
                "64",
                "--maxlen",
                "128",
                "--max_length",
                "256",
                "--neighbor_k",
                "10",
            ],
            "semantic_assets.log",
        )

    if not teacher_checkpoint().exists():
        run_step(
            "Training compact sequential teacher",
            [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={NPROC_PER_NODE}",
                "--master_port=29610",
                "-m",
                "SeqRec.bert4rec.train_small_model",
                "--dataset",
                DATASET,
                "--data_root",
                DATA_ROOT,
                "--sa_asset_root",
                ASSET_ROOT,
                "--batch_size",
                env_value("TEACHER_BATCH_SIZE", "256"),
                "--test_batch_size",
                env_value("TEACHER_TEST_BATCH_SIZE", "512"),
                "--num_epochs",
                "100",
                "--learning_rate",
                "1e-3",
                "--weight_decay",
                "0.0",
                "--l2_emb",
                "0.0",
                "--maxlen",
                "128",
                "--hidden_units",
                "64",
                "--num_blocks",
                "2",
                "--num_heads",
                "2",
                "--dropout_rate",
                "0.2",
                "--inner_size",
                "256",
                "--hidden_dropout_prob",
                "0.2",
                "--attn_dropout_prob",
                "0.2",
                "--hidden_act",
                "gelu",
                "--bert_mask_prob",
                "0.15",
                "--bert_rec_objective",
                "next_item",
                "--recsys_backbone",
                TEACHER_BACKBONE,
                "--sa_similarity",
                "cos",
                "--sa_repr_mode",
                "mean",
                "--seed",
                "42",
                "--save_dir",
                "./SeqRec/bert4rec",
                "--save_prefix",
                TEACHER_PREFIX,
                "--sa_alpha",
                "0.1",
                "--sa_beta",
                "0.1",
                "--sa_mlm_probability",
                "0.05",
                "--sa_temperature",
                "0.2",
                "--eval_every",
                "1",
                "--patience",
                "10",
            ],
            "teacher.log",
        )

    run_step(
        "Training Sequence Injection and reporting final metrics",
        [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={NPROC_PER_NODE}",
            "--master_port=29620",
            "main.py",
            "--train",
            "--multi_gpu",
            "--world_size",
            NPROC_PER_NODE,
            "--recsys",
            TEACHER_BACKBONE,
            "--rec_pre_trained_data",
            DATASET,
            "--data_root",
            DATA_ROOT,
            "--recsys_ckpt_path",
            str(teacher_checkpoint()),
            "--llm",
            LLM_NAME,
            "--llm_path",
            LLM_PATH,
            "--hf_local_only",
            "--hf_cache_dir",
            LLM_PATH,
            "--batch_size",
            env_value("SI_BATCH_SIZE", "20"),
            "--train_candidate_num",
            "4",
            "--candidate_chunk_size",
            "80",
            "--batch_size_infer",
            "10",
            "--llm_max_length",
            "1024",
            "--eval_item_batch",
            "32",
            "--eval_max_length",
            "1024",
            "--eval_min_length",
            "1024",
            "--maxlen",
            "128",
            "--stage2_lr",
            "1e-4",
            "--num_epochs",
            "25",
            "--early_stop_patience",
            "3",
            "--min_epochs_before_early_stop",
            "10",
            "--seed",
            "42",
            "--match_weight",
            "1.0",
            "--save_dir",
            SI_SAVE_DIR,
        ],
        "sequence_injection.log",
    )


if __name__ == "__main__":
    main()
