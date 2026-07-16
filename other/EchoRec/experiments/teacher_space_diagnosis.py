import argparse
import csv
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.echorec_teacher import build_recsys_model


@dataclass(frozen=True)
class TeacherSpec:
    label: str
    ckpt: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Teacher-space semantic alignment diagnosis")
    parser.add_argument("--dataset", type=str, default="Movies_and_TV")
    parser.add_argument("--asset_root", type=str, default="./SA_assets")
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--sample_size", type=int, default=3000)
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--neighbor_ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--rbo_p", type=float, default=0.9)
    parser.add_argument("--wo_ckpt", type=str, required=True)
    parser.add_argument("--full_ckpt", type=str, required=True)
    parser.add_argument("--wo_label", type=str, default="GRU-w/o-SACP")
    parser.add_argument("--full_label", type=str, default="GRU-full-SACP")
    parser.add_argument("--output_dir", type=str, default="./analysis/teacher_space_diagnosis")
    parser.add_argument("--include_item_metrics", action="store_true")
    parser.add_argument("--item_sample_size", type=int, default=5000)
    return parser.parse_args()


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
    user_ids: List[int] = []
    seq_arrays: List[np.ndarray] = []
    seq_lengths: List[int] = []
    for seq_key, _ in seq_items:
        parts = [int(x) for x in seq_key.split(":") if x]
        user_ids.append(parts[0])
        prefix_items = parts[1:]
        seq_arrays.append(right_align(prefix_items, maxlen))
        seq_lengths.append(len(prefix_items))

    semantic = torch.load(asset_dir / "user_semantic_embeddings.pt", map_location="cpu")
    if isinstance(semantic, torch.Tensor):
        semantic = semantic.float().cpu().numpy()
    else:
        semantic = np.asarray(semantic, dtype=np.float32)

    if semantic.shape[0] != len(seq_items):
        raise RuntimeError(
            "user_semantic_embeddings.pt rows do not match seq_keys_to_int.pkl: "
            f"{semantic.shape[0]} vs {len(seq_items)}"
        )

    return {
        "user_ids": np.asarray(user_ids, dtype=np.int64),
        "seq_arrays": np.asarray(seq_arrays, dtype=np.int32),
        "seq_lengths": np.asarray(seq_lengths, dtype=np.int32),
        "semantic": semantic.astype(np.float32, copy=False),
    }


def sample_indices(total: int, sample_size: int, seed: int) -> np.ndarray:
    actual = min(int(total), int(sample_size))
    rng = random.Random(seed)
    picked = rng.sample(range(total), actual)
    picked.sort()
    return np.asarray(picked, dtype=np.int64)


def load_teacher_model(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    kwargs, state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = kwargs["args"]
    saved_args.device = device
    model = build_recsys_model(kwargs["user_num"], kwargs["item_num"], saved_args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {ckpt_path.name}: missing keys: {missing[:5]}")
    if unexpected:
        print(f"[warn] {ckpt_path.name}: unexpected keys: {unexpected[:5]}")
    return model.to(device).eval()


def extract_teacher_embeddings(
    model: torch.nn.Module,
    user_ids: np.ndarray,
    seq_arrays: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs: List[torch.Tensor] = []
    zeros = np.zeros_like(seq_arrays, dtype=np.int32)
    with torch.no_grad():
        for start in range(0, seq_arrays.shape[0], batch_size):
            end = min(start + batch_size, seq_arrays.shape[0])
            emb = model(user_ids[start:end], seq_arrays[start:end], zeros[start:end], zeros[start:end], mode="log_only")
            outputs.append(emb.detach().float().cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


def normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def topk_neighbors(embeddings: np.ndarray, k: int) -> np.ndarray:
    n = int(embeddings.shape[0])
    if n <= 1:
        raise ValueError("Need at least two embeddings for top-k diagnosis")
    k = min(int(k), n - 1)
    emb = normalize(embeddings.astype(np.float32, copy=False))
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row_index = np.arange(n)[:, None]
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


def neighbor_agreement(source: np.ndarray, target: np.ndarray, k: int, rbo_p: float) -> Dict[str, float]:
    source_topk = topk_neighbors(source, k)
    target_topk = topk_neighbors(target, k)
    overlaps = []
    jaccards = []
    rbos = []
    for row_source, row_target in zip(source_topk, target_topk):
        set_source = set(int(x) for x in row_source)
        set_target = set(int(x) for x in row_target)
        inter = len(set_source & set_target)
        union = len(set_source | set_target)
        overlaps.append(inter / float(k))
        jaccards.append(inter / float(union if union > 0 else 1))
        rbos.append(truncated_rbo(row_source, row_target, rbo_p))
    return {
        f"overlap@{k}": float(np.mean(overlaps)),
        f"jaccard@{k}": float(np.mean(jaccards)),
        f"rbo@{k}": float(np.mean(rbos)),
    }


def semantic_neighbor_margin(semantic: np.ndarray, teacher: np.ndarray, k: int, seed: int) -> Dict[str, float]:
    semantic_topk = topk_neighbors(semantic, k)
    teacher_norm = normalize(teacher.astype(np.float32, copy=False))
    row_index = np.arange(teacher_norm.shape[0])[:, None]
    pos_sim = np.sum(teacher_norm[:, None, :] * teacher_norm[semantic_topk], axis=-1)

    rng = np.random.default_rng(seed)
    random_idx = rng.integers(0, teacher_norm.shape[0], size=semantic_topk.shape, endpoint=False)
    random_idx = np.where(random_idx == row_index, (random_idx + 1) % teacher_norm.shape[0], random_idx)
    neg_sim = np.sum(teacher_norm[:, None, :] * teacher_norm[random_idx], axis=-1)

    pos_mean = float(np.mean(pos_sim))
    neg_mean = float(np.mean(neg_sim))
    return {
        f"sem_neighbor_teacher_pos_cos@{k}": pos_mean,
        f"sem_neighbor_teacher_rand_cos@{k}": neg_mean,
        f"sem_neighbor_teacher_margin@{k}": pos_mean - neg_mean,
    }


def get_item_embeddings(model: torch.nn.Module) -> np.ndarray:
    item_emb = getattr(model, "item_emb")
    if hasattr(item_emb, "weight"):
        return item_emb.weight.detach().float().cpu().numpy()
    return item_emb.detach().float().cpu().numpy()


def load_item_semantic(asset_root: Path, dataset: str) -> np.ndarray:
    semantic = torch.load(asset_root / dataset / "item_semantic_embeddings.pt", map_location="cpu")
    if isinstance(semantic, torch.Tensor):
        return semantic.float().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(semantic, dtype=np.float32)


def summarize(rows: Sequence[Dict[str, float]], labels: Sequence[str], neighbor_ks: Sequence[int]) -> List[Dict[str, float]]:
    output = []
    metric_names: List[str] = []
    for k in neighbor_ks:
        metric_names.extend(
            [
                f"seq_jaccard@{k}",
                f"seq_rbo@{k}",
                f"seq_margin@{k}",
            ]
        )
    if any(any(key.startswith("item_jaccard@") for key in row) for row in rows):
        for k in neighbor_ks:
            metric_names.extend([f"item_jaccard@{k}", f"item_rbo@{k}"])

    for label in labels:
        label_rows = [row for row in rows if row["model"] == label]
        summary: Dict[str, float] = {"model": label}
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in label_rows if metric_name in row]
            if not values:
                continue
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values, ddof=0))
        output.append(summary)
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    asset_root = (PROJECT_ROOT / args.asset_root).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir / args.dataset).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_sequence_assets(asset_root, args.dataset, args.maxlen)
    total = int(payload["seq_arrays"].shape[0])
    neighbor_ks = sorted(set(int(k) for k in args.neighbor_ks))
    max_k = max(neighbor_ks)
    if args.sample_size <= max_k:
        raise ValueError(f"--sample_size must be larger than max neighbor k ({max_k})")

    specs = [
        TeacherSpec(args.wo_label, (PROJECT_ROOT / args.wo_ckpt).resolve()),
        TeacherSpec(args.full_label, (PROJECT_ROOT / args.full_ckpt).resolve()),
    ]

    print(f"[info] dataset={args.dataset}, sequence_prefixes={total}, sample_size={args.sample_size}")
    print(f"[info] device={device}, neighbor_ks={neighbor_ks}")

    teacher_models = {spec.label: load_teacher_model(spec.ckpt, device) for spec in specs}

    item_semantic = None
    item_indices_by_seed: Dict[int, np.ndarray] = {}
    if args.include_item_metrics:
        item_semantic = load_item_semantic(asset_root, args.dataset)

    per_seed_rows: List[Dict[str, float]] = []
    for seed in args.seeds:
        picked = sample_indices(total, args.sample_size, seed)
        semantic = payload["semantic"][picked]
        users = payload["user_ids"][picked]
        seqs = payload["seq_arrays"][picked]

        if item_semantic is not None:
            max_item_id = min(item_semantic.shape[0] - 1, *(get_item_embeddings(m).shape[0] - 1 for m in teacher_models.values()))
            rng = random.Random(seed)
            item_sample_size = min(args.item_sample_size, max_item_id)
            item_indices = np.asarray(sorted(rng.sample(range(1, max_item_id + 1), item_sample_size)), dtype=np.int64)
            item_indices_by_seed[seed] = item_indices

        for spec in specs:
            model = teacher_models[spec.label]
            teacher = extract_teacher_embeddings(model, users, seqs, args.teacher_batch_size)
            row: Dict[str, float] = {"seed": int(seed), "model": spec.label, "sample_size": int(picked.shape[0])}

            for k in neighbor_ks:
                agreement = neighbor_agreement(semantic, teacher, k, args.rbo_p)
                margin = semantic_neighbor_margin(semantic, teacher, k, seed)
                row[f"seq_overlap@{k}"] = agreement[f"overlap@{k}"]
                row[f"seq_jaccard@{k}"] = agreement[f"jaccard@{k}"]
                row[f"seq_rbo@{k}"] = agreement[f"rbo@{k}"]
                row[f"seq_pos_cos@{k}"] = margin[f"sem_neighbor_teacher_pos_cos@{k}"]
                row[f"seq_rand_cos@{k}"] = margin[f"sem_neighbor_teacher_rand_cos@{k}"]
                row[f"seq_margin@{k}"] = margin[f"sem_neighbor_teacher_margin@{k}"]

            if item_semantic is not None:
                item_indices = item_indices_by_seed[seed]
                item_emb = get_item_embeddings(model)[item_indices]
                item_sem = item_semantic[item_indices]
                for k in neighbor_ks:
                    item_agreement = neighbor_agreement(item_sem, item_emb, k, args.rbo_p)
                    row[f"item_overlap@{k}"] = item_agreement[f"overlap@{k}"]
                    row[f"item_jaccard@{k}"] = item_agreement[f"jaccard@{k}"]
                    row[f"item_rbo@{k}"] = item_agreement[f"rbo@{k}"]

            per_seed_rows.append(row)
            print(
                f"[seed={seed}] {spec.label}: "
                f"seq_jaccard@20={row.get('seq_jaccard@20', float('nan')):.4f}, "
                f"seq_rbo@20={row.get('seq_rbo@20', float('nan')):.4f}, "
                f"seq_margin@20={row.get('seq_margin@20', float('nan')):.4f}"
            )

    summary_rows = summarize(per_seed_rows, [spec.label for spec in specs], neighbor_ks)

    write_csv(output_dir / "per_seed_metrics.csv", per_seed_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "sample_size": args.sample_size,
                "seeds": args.seeds,
                "neighbor_ks": neighbor_ks,
                "rbo_p": args.rbo_p,
                "teachers": {spec.label: str(spec.ckpt) for spec in specs},
                "include_item_metrics": bool(args.include_item_metrics),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[done] saved to {output_dir}")


if __name__ == "__main__":
    main()
