import argparse
import csv
import gc
import json
import math
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_parser, set_random_seed
from models.echorec_si import EchoRecSIModel
from models.echorec_teacher import build_recsys_model


@dataclass
class ModelSpec:
    label: str
    save_dir: str
    teacher_ckpt: str
    teacher_kind: str


class PrefixDataset(Dataset):
    def __init__(self, user_ids: np.ndarray, seqs: np.ndarray):
        self.user_ids = user_ids.astype(np.int64)
        self.seqs = seqs.astype(np.int32)

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    def __getitem__(self, idx: int):
        user_id = np.int64(self.user_ids[idx])
        seq = self.seqs[idx]
        pos = np.zeros([1], dtype=np.int32)
        neg = np.zeros([1], dtype=np.int32)
        return user_id, seq, pos, neg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CDs SSHG diagnosis")
    parser.add_argument("--dataset", type=str, default="CDs_and_Vinyl")
    parser.add_argument("--asset_root", type=str, default="./SA_assets")
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--viz_seed", type=int, default=42)
    parser.add_argument("--viz_users", type=int, default=1200)
    parser.add_argument("--neighbor_k", type=int, default=20)
    parser.add_argument("--rbo_p", type=float, default=0.9)
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_infer", type=int, default=8)
    parser.add_argument("--llm", type=str, default="llama-3b")
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--hf_local_only", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--eval_item_batch", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=1024)
    parser.add_argument("--eval_min_length", type=int, default=1024)
    parser.add_argument("--llm_max_length", type=int, default=1024)
    parser.add_argument("--inference_chunk_size", type=int, default=8)
    parser.add_argument("--candidate_chunk_size", type=int, default=80)
    parser.add_argument("--candidate_chunk_threshold", type=int, default=36)
    parser.add_argument("--min_candidate_chunk_size", type=int, default=20)
    parser.add_argument("--history_window", type=int, default=10)
    parser.add_argument("--student_eval_candidates", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./analysis/cds_sshg")
    parser.add_argument(
        "--pure_save_dir",
        type=str,
        default="cds_pure_sasrec_si_cand4_5090",
    )
    parser.add_argument(
        "--pure_teacher_ckpt",
        type=str,
        default="./SeqRec/sasrec/CDs_and_Vinyl/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
    )
    parser.add_argument(
        "--echo_save_dir",
        type=str,
        default="cds_si_cand4_5090",
    )
    parser.add_argument(
        "--echo_teacher_ckpt",
        type=str,
        default="./SeqRec/sasrec/CDs_and_Vinyl/cds_sa_teacher/model_metric_best.pth",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    set_random_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_base_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    args.multi_gpu = False
    args.local_rank = -1
    args.world_size = 1
    args.train = False
    args.extract = False
    args.train_student = False
    args.token = False
    args.nn_parameter = False
    args.use_amp = False
    args.disable_model_saving = True
    args.recsys = "sasrec"
    args.rec_pre_trained_data = cli_args.dataset
    args.maxlen = cli_args.maxlen
    args.llm = cli_args.llm
    args.llm_path = cli_args.llm_path
    args.hf_local_only = cli_args.hf_local_only
    args.hf_cache_dir = cli_args.hf_cache_dir or cli_args.llm_path
    args.batch_size_infer = cli_args.batch_size_infer
    args.eval_item_batch = cli_args.eval_item_batch
    args.eval_max_length = cli_args.eval_max_length
    args.eval_min_length = cli_args.eval_min_length
    args.llm_max_length = cli_args.llm_max_length
    args.inference_chunk_size = cli_args.inference_chunk_size
    args.candidate_chunk_size = cli_args.candidate_chunk_size
    args.candidate_chunk_threshold = cli_args.candidate_chunk_threshold
    args.min_candidate_chunk_size = cli_args.min_candidate_chunk_size
    args.train_history_window = cli_args.history_window
    args.student_eval_candidates = cli_args.student_eval_candidates
    args.save_dir = ""
    args.device = torch.device(f"cuda:{cli_args.device}" if torch.cuda.is_available() else "cpu")
    return args


def detect_best_epoch(model_dir: Path, dataset: str, llm: str, subdir: str = "best") -> int:
    target_dir = model_dir / subdir
    pattern = re.compile(rf"^{re.escape(dataset)}_{re.escape(llm)}_(\d+)_item_proj\.pt$")
    epochs: List[int] = []
    for file in target_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            epochs.append(int(match.group(1)))
    if not epochs:
        raise FileNotFoundError(f"No best checkpoint files found under {target_dir}")
    return max(epochs)


def parse_best_results(model_dir: Path, dataset: str, llm: str, epoch: int) -> Dict[str, float]:
    result_path = model_dir / "best" / f"{dataset}_{llm}_{epoch}_results.txt"
    metrics: Dict[str, float] = {}
    with result_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("LLM NDCG@10:"):
                match = re.search(r"LLM NDCG@10:\s*([0-9.]+),\s*LLM HR@10:\s*([0-9.]+)", line)
                if match:
                    metrics["test_ndcg10"] = float(match.group(1))
                    metrics["test_hr10"] = float(match.group(2))
            elif line.startswith("LLM NDCG@20:"):
                match = re.search(r"LLM NDCG@20:\s*([0-9.]+),\s*LLM HR@20:\s*([0-9.]+)", line)
                if match:
                    metrics["test_ndcg20"] = float(match.group(1))
                    metrics["test_hr20"] = float(match.group(2))
            elif line.startswith("Small NDCG@10:"):
                match = re.search(r"Small NDCG@10:\s*([0-9.]+),\s*Small HR@10:\s*([0-9.]+)", line)
                if match:
                    metrics["small_ndcg10"] = float(match.group(1))
                    metrics["small_hr10"] = float(match.group(2))
            elif line.startswith("Small NDCG@20:"):
                match = re.search(r"Small NDCG@20:\s*([0-9.]+),\s*Small HR@20:\s*([0-9.]+)", line)
                if match:
                    metrics["small_ndcg20"] = float(match.group(1))
                    metrics["small_hr20"] = float(match.group(2))
    return metrics


def right_align(items: Sequence[int], maxlen: int) -> np.ndarray:
    seq = np.zeros([maxlen], dtype=np.int32)
    items = [int(x) for x in items][-maxlen:]
    if items:
        seq[-len(items):] = np.asarray(items, dtype=np.int32)
    return seq


def load_sequence_assets(asset_root: Path, dataset: str, maxlen: int) -> Dict[str, np.ndarray]:
    asset_dir = asset_root / dataset
    with (asset_dir / "seq_keys_to_int.pkl").open("rb") as f:
        seq_keys_to_int = pickle.load(f)
    seq_items = sorted(seq_keys_to_int.items(), key=lambda x: x[1])
    user_ids = []
    seq_arrays = []
    seq_lengths = []
    seq_keys = []
    for seq_key, _ in seq_items:
        parts = [int(x) for x in seq_key.split(":") if x]
        seq_keys.append(seq_key)
        user_ids.append(parts[0])
        prefix_items = parts[1:]
        seq_arrays.append(right_align(prefix_items, maxlen))
        seq_lengths.append(len(prefix_items))

    semantic = torch.load(asset_dir / "user_semantic_embeddings.pt", map_location="cpu")
    if isinstance(semantic, torch.Tensor):
        semantic = semantic.float().cpu().numpy()
    else:
        semantic = np.asarray(semantic, dtype=np.float32)
    if semantic.shape[0] != len(seq_keys):
        raise RuntimeError("Semantic asset rows do not match seq_keys_to_int")
    return {
        "seq_keys": np.asarray(seq_keys, dtype=object),
        "user_ids": np.asarray(user_ids, dtype=np.int64),
        "seq_arrays": np.asarray(seq_arrays, dtype=np.int32),
        "seq_lengths": np.asarray(seq_lengths, dtype=np.int32),
        "semantic": semantic.astype(np.float32, copy=False),
    }


def sample_indices(total: int, sample_size: int, seed: int) -> np.ndarray:
    actual = min(total, sample_size)
    rng = random.Random(seed)
    picked = rng.sample(range(total), actual)
    picked.sort()
    return np.asarray(picked, dtype=np.int64)


def load_teacher_model(ckpt_path: Path, device: torch.device):
    kwargs, state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = kwargs["args"]
    saved_args.device = device
    model = build_recsys_model(kwargs["user_num"], kwargs["item_num"], saved_args)
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


def extract_teacher_embeddings(model, user_ids: np.ndarray, seq_arrays: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: List[torch.Tensor] = []
    zeros = np.zeros_like(seq_arrays, dtype=np.int32)
    with torch.no_grad():
        for start in range(0, seq_arrays.shape[0], batch_size):
            end = min(start + batch_size, seq_arrays.shape[0])
            emb = model(user_ids[start:end], seq_arrays[start:end], zeros[start:end], zeros[start:end], mode="log_only")
            outputs.append(emb.detach().float().cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


def load_si_model(base_args: argparse.Namespace, spec: ModelSpec):
    args = argparse.Namespace(**vars(base_args))
    args.save_dir = spec.save_dir
    args.recsys_ckpt_path = spec.teacher_ckpt
    model_dir = PROJECT_ROOT / "models" / args.rec_pre_trained_data / args.save_dir
    epoch = detect_best_epoch(model_dir, args.rec_pre_trained_data, args.llm)
    model = EchoRecSIModel(args).to(args.device)
    model.load_model(args, phase2_epoch=epoch, subdir="best")
    model.eval()
    model.extract_embs_list = []
    return model, epoch


def extract_llm_embeddings(model: EchoRecSIModel, user_ids: np.ndarray, seq_arrays: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        PrefixDataset(user_ids, seq_arrays),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    model.extract_embs_list = []
    with torch.no_grad():
        for u, seq, pos, neg in loader:
            model((u, seq, pos, neg, seq, None, None), mode="extract")
    return torch.cat(model.extract_embs_list, dim=0).numpy().astype(np.float32, copy=False)


def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def topk_neighbors(embeddings: np.ndarray, k: int) -> np.ndarray:
    emb = normalize(embeddings.astype(np.float32, copy=False))
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row_index = np.arange(sim.shape[0])[:, None]
    row_scores = sim[row_index, topk]
    order = np.argsort(-row_scores, axis=1)
    return topk[row_index, order].astype(np.int32, copy=False)


def truncated_rbo(left: Sequence[int], right: Sequence[int], p: float) -> float:
    seen_left = set()
    seen_right = set()
    accum = 0.0
    depth = min(len(left), len(right))
    for d in range(1, depth + 1):
        seen_left.add(int(left[d - 1]))
        seen_right.add(int(right[d - 1]))
        overlap_d = len(seen_left & seen_right) / float(d)
        accum += overlap_d * (p ** (d - 1))
    if depth == 0:
        return 0.0
    normalizer = (1.0 - p) / (1.0 - p ** depth)
    return float(normalizer * accum)


def compute_alignment_metrics(source: np.ndarray, target: np.ndarray, k: int, rbo_p: float) -> Dict[str, float]:
    left = topk_neighbors(source, k)
    right = topk_neighbors(target, k)
    overlaps = []
    jaccards = []
    rbos = []
    for row_left, row_right in zip(left, right):
        set_left = set(int(x) for x in row_left)
        set_right = set(int(x) for x in row_right)
        inter = len(set_left & set_right)
        union = len(set_left | set_right)
        overlaps.append(inter / float(k))
        jaccards.append(inter / float(union if union > 0 else 1))
        rbos.append(truncated_rbo(row_left, row_right, rbo_p))
    return {
        f"overlap@{k}": float(np.mean(overlaps)),
        f"jaccard@{k}": float(np.mean(jaccards)),
        f"rbo@{k}": float(np.mean(rbos)),
    }


def summarize(rows: List[Dict[str, float]], outcomes: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)

    preferred_order = {"Pure-SI": 0, "EchoRec": 1}
    ordered = sorted(grouped.keys(), key=lambda name: preferred_order.get(name, 99))
    metric_names = [
        "pre_jaccard@20",
        "post_jaccard@20",
        "post_rbo@20",
    ]
    output = []
    for model_name in ordered:
        model_rows = grouped[model_name]
        summary: Dict[str, float] = {"model": model_name}
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in model_rows]
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values, ddof=0))
        summary.update(outcomes[model_name])
        output.append(summary)
    return output


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    cli_args = parse_args()
    seed_everything(cli_args.seed)

    output_dir = (PROJECT_ROOT / cli_args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_sequence_assets(PROJECT_ROOT / cli_args.asset_root, cli_args.dataset, cli_args.maxlen)
    total_prefixes = int(payload["seq_arrays"].shape[0])
    print(f"[info] dataset={cli_args.dataset}, prefixes={total_prefixes}")

    seeds = list(dict.fromkeys(cli_args.seeds))
    if cli_args.viz_seed not in seeds:
        seeds.append(cli_args.viz_seed)
    sample_map = {
        seed: sample_indices(total_prefixes, max(cli_args.sample_size, cli_args.viz_users) if seed == cli_args.viz_seed else cli_args.sample_size, seed)
        for seed in seeds
    }

    device = torch.device(f"cuda:{cli_args.device}" if torch.cuda.is_available() else "cpu")
    raw_teacher = load_teacher_model((PROJECT_ROOT / cli_args.pure_teacher_ckpt).resolve(), device)
    sacp_teacher = load_teacher_model((PROJECT_ROOT / cli_args.echo_teacher_ckpt).resolve(), device)

    semantic_by_seed: Dict[int, np.ndarray] = {}
    raw_teacher_by_seed: Dict[int, np.ndarray] = {}
    sacp_teacher_by_seed: Dict[int, np.ndarray] = {}
    for seed in seeds:
        picked = sample_map[seed]
        seqs = payload["seq_arrays"][picked]
        users = payload["user_ids"][picked]
        semantic_by_seed[seed] = payload["semantic"][picked]
        raw_teacher_by_seed[seed] = extract_teacher_embeddings(raw_teacher, users, seqs, cli_args.teacher_batch_size)
        sacp_teacher_by_seed[seed] = extract_teacher_embeddings(sacp_teacher, users, seqs, cli_args.teacher_batch_size)

    del raw_teacher
    del sacp_teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_args = make_base_args(cli_args)
    specs = [
        ModelSpec("Pure-SI", cli_args.pure_save_dir, cli_args.pure_teacher_ckpt, "raw"),
        ModelSpec("EchoRec", cli_args.echo_save_dir, cli_args.echo_teacher_ckpt, "sacp"),
    ]

    llm_by_model: Dict[str, Dict[int, np.ndarray]] = {spec.label: {} for spec in specs}
    outcomes: Dict[str, Dict[str, float]] = {}
    manifest_models: Dict[str, Dict[str, float]] = {}

    for spec in specs:
        print(f"[info] loading {spec.label}")
        model, epoch = load_si_model(base_args, spec)
        model_dir = PROJECT_ROOT / "models" / cli_args.dataset / spec.save_dir
        outcome = parse_best_results(model_dir, cli_args.dataset, cli_args.llm, epoch)
        outcome["best_epoch"] = float(epoch)
        outcomes[spec.label] = outcome
        manifest_models[spec.label] = {
            "save_dir": spec.save_dir,
            "teacher_ckpt": spec.teacher_ckpt,
            "teacher_kind": spec.teacher_kind,
            "best_epoch": epoch,
        }

        for seed in seeds:
            picked = sample_map[seed]
            llm_by_model[spec.label][seed] = extract_llm_embeddings(
                model,
                payload["user_ids"][picked],
                payload["seq_arrays"][picked],
                cli_args.batch_size_infer,
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows: List[Dict[str, float]] = []
    for seed in cli_args.seeds:
        pre_raw = compute_alignment_metrics(semantic_by_seed[seed], raw_teacher_by_seed[seed], cli_args.neighbor_k, cli_args.rbo_p)
        pre_sacp = compute_alignment_metrics(semantic_by_seed[seed], sacp_teacher_by_seed[seed], cli_args.neighbor_k, cli_args.rbo_p)

        for spec in specs:
            teacher = raw_teacher_by_seed[seed] if spec.teacher_kind == "raw" else sacp_teacher_by_seed[seed]
            pre = pre_raw if spec.teacher_kind == "raw" else pre_sacp
            post = compute_alignment_metrics(teacher, llm_by_model[spec.label][seed], cli_args.neighbor_k, cli_args.rbo_p)
            rows.append(
                {
                    "seed": int(seed),
                    "model": spec.label,
                    "teacher_kind": spec.teacher_kind,
                    "pre_jaccard@20": pre[f"jaccard@{cli_args.neighbor_k}"],
                    "post_jaccard@20": post[f"jaccard@{cli_args.neighbor_k}"],
                    "post_rbo@20": post[f"rbo@{cli_args.neighbor_k}"],
                }
            )

    summary_rows = summarize(rows, outcomes)

    write_csv(
        output_dir / "per_seed_metrics.csv",
        rows,
        [
            "seed",
            "model",
            "teacher_kind",
            "pre_jaccard@20",
            "post_jaccard@20",
            "post_rbo@20",
        ],
    )
    write_csv(
        output_dir / "summary_metrics.csv",
        summary_rows,
        [
            "model",
            "pre_jaccard@20_mean",
            "pre_jaccard@20_std",
            "post_jaccard@20_mean",
            "post_jaccard@20_std",
            "post_rbo@20_mean",
            "post_rbo@20_std",
            "test_ndcg10",
            "test_hr10",
            "best_epoch",
        ],
    )

    viz_idx = sample_map[cli_args.viz_seed][: min(cli_args.viz_users, sample_map[cli_args.viz_seed].shape[0])]
    embedding_payload = {
        "sample_indices": viz_idx.astype(np.int64),
        "sample_user_ids": payload["user_ids"][viz_idx].astype(np.int64),
        "sample_seq_lengths": payload["seq_lengths"][viz_idx].astype(np.int32),
        "semantic": payload["semantic"][viz_idx].astype(np.float32),
        "teacher_raw": raw_teacher_by_seed[cli_args.viz_seed][: len(viz_idx)].astype(np.float32),
        "teacher_sacp": sacp_teacher_by_seed[cli_args.viz_seed][: len(viz_idx)].astype(np.float32),
        "llm_pure": llm_by_model["Pure-SI"][cli_args.viz_seed][: len(viz_idx)].astype(np.float32),
        "llm_echo": llm_by_model["EchoRec"][cli_args.viz_seed][: len(viz_idx)].astype(np.float32),
    }
    np.savez_compressed(output_dir / "embedding_payload.npz", **embedding_payload)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": cli_args.dataset,
                "sample_size": cli_args.sample_size,
                "viz_seed": cli_args.viz_seed,
                "viz_users": cli_args.viz_users,
                "neighbor_k": cli_args.neighbor_k,
                "rbo_p": cli_args.rbo_p,
                "models": manifest_models,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[done] metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
