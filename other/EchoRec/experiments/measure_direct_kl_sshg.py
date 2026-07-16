import argparse
import csv
import hashlib
import math
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SA.dataset import data_partition
from experiments.cds_sshg_diagnosis import load_sequence_assets, load_teacher_model


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset: str
    raw_teacher_ckpt: str
    sacp_teacher_ckpt: str


DATASETS: Dict[str, DatasetConfig] = {
    "movies": DatasetConfig(
        key="movies",
        dataset="Movies_and_TV",
        raw_teacher_ckpt="./SeqRec/sasrec/Movies_and_TV/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        sacp_teacher_ckpt="./SeqRec/sasrec/Movies_and_TV/movies_sa_teacher/model_metric_best.pth",
    ),
    "scientific": DatasetConfig(
        key="scientific",
        dataset="Industrial_and_Scientific",
        raw_teacher_ckpt="./SeqRec/sasrec/Industrial_and_Scientific/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        sacp_teacher_ckpt="./SeqRec/sasrec/Industrial_and_Scientific/scientific_sa_teacher/model_metric_best.pth",
    ),
    "electronics": DatasetConfig(
        key="electronics",
        dataset="Electronics",
        raw_teacher_ckpt="./SeqRec/sasrec/Electronics/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        sacp_teacher_ckpt="./SeqRec/sasrec/Electronics/electronics_sa_teacher/model_metric_best.pth",
    ),
    "cds": DatasetConfig(
        key="cds",
        dataset="CDs_and_Vinyl",
        raw_teacher_ckpt="./SeqRec/sasrec/CDs_and_Vinyl/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        sacp_teacher_ckpt="./SeqRec/sasrec/CDs_and_Vinyl/cds_sa_teacher/model_metric_best.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Measure KL-defined SSHG before and after SACP")
    parser.add_argument("--datasets", nargs="+", default=["movies", "scientific", "electronics", "cds"], choices=sorted(DATASETS))
    parser.add_argument("--asset_root", type=str, default="./SA_assets")
    parser.add_argument("--data_root", type=str, default="./SeqRec")
    parser.add_argument("--output_dir", type=str, default="./analysis/direct_kl_sshg")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--candidate_num", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.1])
    parser.add_argument("--teacher_batch_size", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=["user_neighborhood", "item_candidate"],
        default="user_neighborhood",
        help="user_neighborhood matches the theory/RQ4 neighborhood view; item_candidate is the next-item candidate variant.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def right_align(items: Sequence[int], maxlen: int) -> np.ndarray:
    seq = np.zeros([maxlen], dtype=np.int32)
    clipped = [int(item) for item in items][-maxlen:]
    if clipped:
        seq[-len(clipped):] = np.asarray(clipped, dtype=np.int32)
    return seq


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:8], 16)


def load_semantic_assets(asset_root: Path, dataset: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    asset_dir = asset_root / dataset
    item_semantic = torch.load(asset_dir / "item_semantic_embeddings.pt", map_location="cpu").float()
    user_semantic = torch.load(asset_dir / "user_semantic_embeddings.pt", map_location="cpu").float()
    with (asset_dir / "seq_keys_to_int.pkl").open("rb") as f:
        seq_keys_to_int = pickle.load(f)
    if user_semantic.shape[0] != len(seq_keys_to_int):
        raise RuntimeError(
            f"{dataset}: user_semantic rows do not match seq_keys: "
            f"{user_semantic.shape[0]} vs {len(seq_keys_to_int)}"
        )
    return item_semantic, user_semantic, seq_keys_to_int


def build_prefix_records(
    dataset: str,
    data_root: Path,
    seq_keys_to_int: Dict[str, int],
    maxlen: int,
) -> Tuple[List[Dict[str, object]], int]:
    user_train, _, _, _, itemnum, _ = data_partition(dataset, root_dir=str(data_root))
    records: List[Dict[str, object]] = []

    for seq_key, seq_idx in seq_keys_to_int.items():
        parts = [int(part) for part in seq_key.split(":") if part]
        if len(parts) < 2:
            continue
        user_id = parts[0]
        prefix = parts[1:]
        train_items = user_train.get(user_id, [])
        prefix_len = len(prefix)
        if prefix_len >= len(train_items):
            continue
        if list(train_items[:prefix_len]) != prefix:
            continue
        target = int(train_items[prefix_len])
        records.append(
            {
                "user": int(user_id),
                "seq_idx": int(seq_idx),
                "prefix": prefix,
                "target": target,
                "seq": right_align(prefix, maxlen),
            }
        )

    if not records:
        raise RuntimeError(f"{dataset}: no prefix records with next-item targets found")
    return records, int(itemnum)


def sample_records(records: Sequence[Dict[str, object]], sample_size: int, seed: int) -> List[Dict[str, object]]:
    actual = min(len(records), int(sample_size))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), actual))
    return [records[index] for index in indices]


def make_candidates(dataset: str, seed: int, record: Dict[str, object], itemnum: int, candidate_num: int) -> List[int]:
    target = int(record["target"])
    prefix = [int(item) for item in record["prefix"]]
    excluded = set(prefix)
    excluded.add(target)
    excluded.add(0)

    pool = [item for item in range(1, itemnum + 1) if item not in excluded]
    needed = max(0, int(candidate_num) - 1)
    rng = random.Random(stable_seed(dataset, seed, record["user"], record["seq_idx"]))
    negatives = rng.sample(pool, min(needed, len(pool)))
    return [target] + negatives


def get_item_embedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    item_emb = getattr(model, "item_emb")
    if isinstance(item_emb, torch.nn.Embedding):
        return item_emb.weight.detach()
    if isinstance(item_emb, torch.nn.Parameter):
        return item_emb.detach()
    if isinstance(item_emb, torch.Tensor):
        return item_emb.detach()
    raise TypeError(f"Unsupported item_emb type: {type(item_emb)}")


def extract_teacher_user_embeddings(
    model: torch.nn.Module,
    users: np.ndarray,
    seqs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    zeros = np.zeros_like(seqs)
    with torch.no_grad():
        for start in range(0, seqs.shape[0], batch_size):
            end = min(start + batch_size, seqs.shape[0])
            emb = model(
                users[start:end],
                seqs[start:end],
                zeros[start:end],
                zeros[start:end],
                mode="log_only",
            )
            outputs.append(emb.detach().float().cpu())
    return torch.cat(outputs, dim=0)


def candidate_scores(user_embs: torch.Tensor, item_matrix: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    user_norm = F.normalize(user_embs.float(), dim=-1)
    item_norm = F.normalize(item_matrix.float(), dim=-1)
    cand_emb = item_norm[candidates]
    return torch.sum(cand_emb * user_norm.unsqueeze(1), dim=-1)


def to_prob(scores: torch.Tensor, tau: float, eps: float = 1e-12) -> torch.Tensor:
    probs = torch.softmax(scores / float(tau), dim=-1)
    probs = torch.clamp(probs, min=eps)
    return probs / probs.sum(dim=-1, keepdim=True)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


def pairwise_scores(embeddings: torch.Tensor) -> torch.Tensor:
    emb = F.normalize(embeddings.float(), dim=-1)
    scores = emb @ emb.T
    scores.fill_diagonal_(-1e9)
    return scores


def compute_user_neighborhood_for_dataset(config: DatasetConfig, args: argparse.Namespace, device: torch.device) -> List[Dict[str, object]]:
    payload = load_sequence_assets(PROJECT_ROOT / args.asset_root, config.dataset, args.maxlen)
    total_prefixes = int(payload["seq_arrays"].shape[0])

    raw_model = load_teacher_model((PROJECT_ROOT / config.raw_teacher_ckpt).resolve(), device)
    sacp_model = load_teacher_model((PROJECT_ROOT / config.sacp_teacher_ckpt).resolve(), device)

    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        rng = random.Random(int(seed))
        actual = min(int(args.sample_size), total_prefixes)
        picked = np.asarray(sorted(rng.sample(range(total_prefixes), actual)), dtype=np.int64)
        users = payload["user_ids"][picked].astype(np.int64, copy=False)
        seqs = payload["seq_arrays"][picked].astype(np.int32, copy=False)

        semantic = torch.tensor(payload["semantic"][picked], dtype=torch.float32)
        raw_users = extract_teacher_user_embeddings(raw_model, users, seqs, args.teacher_batch_size, device)
        sacp_users = extract_teacher_user_embeddings(sacp_model, users, seqs, args.teacher_batch_size, device)

        sem_scores = pairwise_scores(semantic)
        raw_scores = pairwise_scores(raw_users)
        sacp_scores = pairwise_scores(sacp_users)

        for tau in args.temperatures:
            p_sem = to_prob(sem_scores, tau)
            p_raw = to_prob(raw_scores, tau)
            p_sacp = to_prob(sacp_scores, tau)
            kl_before = kl_divergence(p_raw, p_sem).numpy()
            kl_after = kl_divergence(p_sacp, p_sem).numpy()

            for local_idx, prefix_idx in enumerate(picked):
                before = float(kl_before[local_idx])
                after = float(kl_after[local_idx])
                reduction = before - after
                rows.append(
                    {
                        "dataset": config.key,
                        "dataset_name": config.dataset,
                        "seed": int(seed),
                        "tau": float(tau),
                        "mode": "user_neighborhood",
                        "user": int(users[local_idx]),
                        "seq_idx": int(prefix_idx),
                        "target": -1,
                        "kl_before": before,
                        "kl_after": after,
                        "kl_reduction": reduction,
                        "kl_reduction_pct": reduction / max(before, 1e-12),
                    }
                )

    return rows


def compute_item_candidate_for_dataset(config: DatasetConfig, args: argparse.Namespace, device: torch.device) -> List[Dict[str, object]]:
    asset_root = PROJECT_ROOT / args.asset_root
    data_root = PROJECT_ROOT / args.data_root
    item_semantic, user_semantic, seq_keys_to_int = load_semantic_assets(asset_root, config.dataset)
    records, itemnum = build_prefix_records(config.dataset, data_root, seq_keys_to_int, args.maxlen)

    raw_model = load_teacher_model((PROJECT_ROOT / config.raw_teacher_ckpt).resolve(), device)
    sacp_model = load_teacher_model((PROJECT_ROOT / config.sacp_teacher_ckpt).resolve(), device)
    raw_item_matrix = get_item_embedding_matrix(raw_model).cpu()
    sacp_item_matrix = get_item_embedding_matrix(sacp_model).cpu()

    rows: List[Dict[str, object]] = []
    for seed in args.seeds:
        sampled = sample_records(records, args.sample_size, int(seed))
        users = np.asarray([int(row["user"]) for row in sampled], dtype=np.int64)
        seqs = np.asarray([row["seq"] for row in sampled], dtype=np.int32)
        seq_indices = torch.tensor([int(row["seq_idx"]) for row in sampled], dtype=torch.long)
        candidates = torch.tensor(
            [make_candidates(config.dataset, int(seed), row, itemnum, args.candidate_num) for row in sampled],
            dtype=torch.long,
        )

        raw_users = extract_teacher_user_embeddings(raw_model, users, seqs, args.teacher_batch_size, device)
        sacp_users = extract_teacher_user_embeddings(sacp_model, users, seqs, args.teacher_batch_size, device)

        sem_user = user_semantic[seq_indices]
        sem_scores = candidate_scores(sem_user, item_semantic, candidates)
        raw_scores = candidate_scores(raw_users, raw_item_matrix, candidates)
        sacp_scores = candidate_scores(sacp_users, sacp_item_matrix, candidates)

        for tau in args.temperatures:
            p_sem = to_prob(sem_scores, tau)
            p_raw = to_prob(raw_scores, tau)
            p_sacp = to_prob(sacp_scores, tau)
            kl_before = kl_divergence(p_raw, p_sem).numpy()
            kl_after = kl_divergence(p_sacp, p_sem).numpy()

            for local_idx, row in enumerate(sampled):
                before = float(kl_before[local_idx])
                after = float(kl_after[local_idx])
                reduction = before - after
                rows.append(
                    {
                        "dataset": config.key,
                        "dataset_name": config.dataset,
                        "seed": int(seed),
                        "tau": float(tau),
                        "mode": "item_candidate",
                        "user": int(row["user"]),
                        "seq_idx": int(row["seq_idx"]),
                        "target": int(row["target"]),
                        "kl_before": before,
                        "kl_after": after,
                        "kl_reduction": reduction,
                        "kl_reduction_pct": reduction / max(before, 1e-12),
                    }
                )

    return rows


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), float(row["tau"])), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (dataset, tau), group in sorted(grouped.items()):
        before = np.asarray([float(row["kl_before"]) for row in group], dtype=np.float64)
        after = np.asarray([float(row["kl_after"]) for row in group], dtype=np.float64)
        reduction = before - after
        reduction_pct = reduction / np.maximum(before, 1e-12)
        summary_rows.append(
            {
                "dataset": dataset,
                "tau": tau,
                "num_points": int(len(group)),
                "before_mean": float(before.mean()),
                "after_mean": float(after.mean()),
                "reduction_mean": float(reduction.mean()),
                "reduction_pct_mean": float(reduction_pct.mean()),
                "before_median": float(np.median(before)),
                "after_median": float(np.median(after)),
                "reduction_median": float(np.median(reduction)),
                "reduction_pct_median": float(np.median(reduction_pct)),
            }
        )
    return summary_rows


def latex_table_rows(summary_rows: Sequence[Dict[str, object]], tau: float) -> str:
    rows = [row for row in summary_rows if abs(float(row["tau"]) - float(tau)) < 1e-12]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Direct measurement of KL-defined SSHG before and after SACP. Lower is better.}",
        "\\label{tab:direct_kl_sshg}",
        "\\small",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & Before SACP & After SACP & Mean Reduction & Median Reduction \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = str(row["dataset"]).capitalize()
        before = float(row["before_mean"])
        after = float(row["after_mean"])
        mean_red = 100.0 * float(row["reduction_pct_mean"])
        median_red = 100.0 * float(row["reduction_pct_median"])
        lines.append(f"{dataset} & {before:.4f} & {after:.4f} & {mean_red:.1f}\\% & {median_red:.1f}\\% \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    all_rows: List[Dict[str, object]] = []
    for key in args.datasets:
        print(f"[info] measuring {key} ({args.mode})", flush=True)
        if args.mode == "user_neighborhood":
            all_rows.extend(compute_user_neighborhood_for_dataset(DATASETS[key], args, device))
        else:
            all_rows.extend(compute_item_candidate_for_dataset(DATASETS[key], args, device))

    summary_rows = summarize(all_rows)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    write_csv(
        output_dir / "direct_kl_sshg_user_level.csv",
        all_rows,
        [
            "dataset",
            "dataset_name",
            "seed",
            "tau",
            "mode",
            "user",
            "seq_idx",
            "target",
            "kl_before",
            "kl_after",
            "kl_reduction",
            "kl_reduction_pct",
        ],
    )
    write_csv(
        output_dir / "direct_kl_sshg_summary.csv",
        summary_rows,
        [
            "dataset",
            "tau",
            "num_points",
            "before_mean",
            "after_mean",
            "reduction_mean",
            "reduction_pct_mean",
            "before_median",
            "after_median",
            "reduction_median",
            "reduction_pct_median",
        ],
    )
    primary_tau = float(args.temperatures[0])
    (output_dir / "direct_kl_sshg_table_tau_primary.tex").write_text(
        latex_table_rows(summary_rows, primary_tau),
        encoding="utf-8",
    )
    for row in summary_rows:
        print(
            f"{row['dataset']} tau={row['tau']}: "
            f"before={float(row['before_mean']):.4f}, "
            f"after={float(row['after_mean']):.4f}, "
            f"mean_reduction={100.0 * float(row['reduction_pct_mean']):.1f}%, "
            f"median_reduction={100.0 * float(row['reduction_pct_median']):.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
