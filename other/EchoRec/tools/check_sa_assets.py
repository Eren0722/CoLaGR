import argparse
import os
import pickle

import numpy as np
import torch


def _load_pt(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(obj)}")
    return obj.float()


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_dataset_root(asset_root: str, dataset: str) -> str:
    dataset_root = os.path.join(asset_root, dataset)
    if os.path.isdir(dataset_root):
        return dataset_root
    if os.path.isdir(asset_root):
        return asset_root
    raise FileNotFoundError(f"Asset root not found: {asset_root}")


def _cosine_stats(emb: torch.Tensor, neighbors: np.ndarray, skip_padding_row: bool) -> dict:
    if emb.ndim != 2 or neighbors.ndim != 2:
        raise ValueError("Embeddings and neighbor table must both be 2D")

    emb = torch.nn.functional.normalize(emb.float(), dim=-1)

    start_row = 1 if skip_padding_row else 0
    if emb.shape[0] <= start_row:
        raise ValueError("Not enough rows to compute cosine stats")

    rows = torch.arange(start_row, emb.shape[0], dtype=torch.long)
    neighbor_tensor = torch.from_numpy(neighbors[rows.numpy()]).long()

    top1 = neighbor_tensor[:, 0]
    top1_cos = (emb[rows] * emb[top1]).sum(dim=-1)

    topk_cos = []
    for k in (1, 3, 5, 10):
        width = min(k, neighbor_tensor.shape[1])
        cur = neighbor_tensor[:, :width]
        cur_cos = (emb[rows].unsqueeze(1) * emb[cur]).sum(dim=-1).mean(dim=-1)
        topk_cos.append((k, float(cur_cos.mean().item())))

    duplicate_rows = int(sum(len(set(row.tolist())) != len(row.tolist()) for row in neighbor_tensor))
    self_rows = int(sum(int(r.item()) in set(row.tolist()) for r, row in zip(rows, neighbor_tensor)))

    return {
        "top1_mean": float(top1_cos.mean().item()),
        "top1_min": float(top1_cos.min().item()),
        "top1_max": float(top1_cos.max().item()),
        "topk_mean": topk_cos,
        "duplicate_rows": duplicate_rows,
        "self_rows": self_rows,
    }


def main():
    parser = argparse.ArgumentParser("Check semantic alignment assets")
    parser.add_argument("--asset_root", required=True, help="Asset root, e.g. ./SA_assets")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Movies_and_TV")
    args = parser.parse_args()

    root = _resolve_dataset_root(args.asset_root, args.dataset)
    files = {
        "item_semantic": os.path.join(root, "item_semantic_embeddings.pt"),
        "user_semantic": os.path.join(root, "user_semantic_embeddings.pt"),
        "seq_keys": os.path.join(root, "seq_keys_to_int.pkl"),
        "item_neighbors": os.path.join(root, "item_sorted_indices.npy"),
        "user_neighbors": os.path.join(root, "user_sorted_indices.npy"),
    }

    print(f"dataset_root: {root}")
    for name, path in files.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else None
        print(f"{name}: exists={exists} size={size} path={path}")

    missing = [name for name, path in files.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing asset files: {missing}")

    item_sem = _load_pt(files["item_semantic"])
    user_sem = _load_pt(files["user_semantic"])
    seq_keys = _load_pkl(files["seq_keys"])
    item_neighbors = np.load(files["item_neighbors"])
    user_neighbors = np.load(files["user_neighbors"])

    print()
    print(f"item_semantic shape={tuple(item_sem.shape)} dtype={item_sem.dtype}")
    print(f"user_semantic shape={tuple(user_sem.shape)} dtype={user_sem.dtype}")
    print(f"seq_keys len={len(seq_keys)}")
    print(f"item_neighbors shape={item_neighbors.shape} dtype={item_neighbors.dtype}")
    print(f"user_neighbors shape={user_neighbors.shape} dtype={user_neighbors.dtype}")

    if user_sem.shape[0] != len(seq_keys):
        print("ERROR: user_semantic row count does not match seq_keys length")
    else:
        print("OK: user_semantic row count matches seq_keys length")

    if item_neighbors.shape[0] != item_sem.shape[0]:
        print("ERROR: item_neighbors row count does not match item_semantic rows")
    else:
        print("OK: item_neighbors row count matches item_semantic rows")

    if user_neighbors.shape[0] != user_sem.shape[0]:
        print("ERROR: user_neighbors row count does not match user_semantic rows")
    else:
        print("OK: user_neighbors row count matches user_semantic rows")

    print()
    item_stats = _cosine_stats(item_sem, item_neighbors, skip_padding_row=True)
    user_stats = _cosine_stats(user_sem, user_neighbors, skip_padding_row=False)

    print(
        "item neighbor cosine:"
        f" top1_mean={item_stats['top1_mean']:.4f}"
        f" top1_min={item_stats['top1_min']:.4f}"
        f" top1_max={item_stats['top1_max']:.4f}"
        f" duplicate_rows={item_stats['duplicate_rows']}"
        f" self_rows={item_stats['self_rows']}"
    )
    for k, value in item_stats["topk_mean"]:
        print(f"  item top{k}_mean={value:.4f}")

    print(
        "user neighbor cosine:"
        f" top1_mean={user_stats['top1_mean']:.4f}"
        f" top1_min={user_stats['top1_min']:.4f}"
        f" top1_max={user_stats['top1_max']:.4f}"
        f" duplicate_rows={user_stats['duplicate_rows']}"
        f" self_rows={user_stats['self_rows']}"
    )
    for k, value in user_stats["topk_mean"]:
        print(f"  user top{k}_mean={value:.4f}")


if __name__ == "__main__":
    main()
