import os
import pickle
from typing import Dict, Optional

import numpy as np
import torch


SEMANTIC_FILE_HINTS = {
    "item_semantic": "item_semantic_embeddings.pt",
    "user_semantic": "user_semantic_embeddings.pt",
    "seq_keys": "seq_keys_to_int.pkl",
    "item_neighbors": "item_sorted_indices.npy",
    "user_neighbors": "user_sorted_indices.npy",
}


def _resolve_path(args, attr_name: str, default_file: str) -> Optional[str]:
    path = getattr(args, attr_name, None)
    if path:
        return path

    root = getattr(args, "sa_asset_root", None)
    if not root:
        return None

    dataset = getattr(args, "rec_pre_trained_data", "") or getattr(args, "dataset", "")
    candidate = os.path.join(root, dataset, default_file)
    if os.path.exists(candidate):
        return candidate

    fallback = os.path.join(root, default_file)
    if os.path.exists(fallback):
        return fallback
    return None


def _load_tensor_file(file_path: str):
    if file_path.endswith(".pt") or file_path.endswith(".pth"):
        return torch.load(file_path, map_location="cpu")
    if file_path.endswith(".npy"):
        return torch.from_numpy(np.load(file_path))
    raise ValueError(f"Unsupported tensor file format: {file_path}")


def _load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _compute_topk_from_embeddings(emb: torch.Tensor, topk: int, chunk_size: int = 1024) -> torch.Tensor:
    emb = emb.float()
    emb = torch.nn.functional.normalize(emb, dim=-1)
    n = emb.shape[0]
    k = min(topk, max(n - 1, 0))
    if k <= 0:
        return torch.empty((n, 0), dtype=torch.long)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    emb_dev = emb.to(device)

    all_indices = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim_chunk = emb_dev[start:end] @ emb_dev.T
        row_count = end - start
        local_rows = torch.arange(row_count, device=device)
        global_rows = torch.arange(start, end, device=device)
        sim_chunk[local_rows, global_rows] = float("-inf")
        _, idx_chunk = torch.topk(sim_chunk, k=k, dim=-1)
        all_indices.append(idx_chunk.cpu())
        del sim_chunk

    del emb_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()

    indices = torch.cat(all_indices, dim=0)
    return indices


def _align_neighbor_width(indices: torch.Tensor, k_num: int) -> torch.Tensor:
    """Force loaded/precomputed neighbor tables to respect the training k setting."""
    if indices.ndim != 2:
        raise ValueError(f"Neighbor index table must be 2D, got shape={tuple(indices.shape)}")
    if k_num <= 0:
        raise ValueError(f"sa_k_num must be positive, got {k_num}")
    if indices.shape[1] == k_num:
        return indices
    if indices.shape[1] < k_num:
        raise ValueError(
            f"Neighbor index table width ({indices.shape[1]}) is smaller than requested sa_k_num ({k_num})"
        )
    print(
        f"Warning: neighbor index table width {indices.shape[1]} does not match sa_k_num={k_num}; "
        f"truncating to top-{k_num}."
    )
    return indices[:, :k_num]


def _compute_item_topk_excluding_padding(emb: torch.Tensor, topk: int, chunk_size: int = 1024) -> torch.Tensor:
    """Compute item neighbors over real items only, excluding padding id 0."""
    if emb.ndim != 2:
        raise ValueError(f"Item semantic embeddings must be 2D, got shape={tuple(emb.shape)}")
    if emb.shape[0] <= 1:
        raise ValueError("Item semantic embeddings must include at least one real item besides padding")

    real_item_emb = emb[1:]
    real_item_topk = _compute_topk_from_embeddings(real_item_emb, topk, chunk_size=chunk_size) + 1
    out = torch.zeros((emb.shape[0], real_item_topk.shape[1]), dtype=torch.long)
    out[0] = torch.arange(1, real_item_topk.shape[1] + 1, dtype=torch.long)
    out[1:] = real_item_topk.long()
    return out


def _sanitize_neighbor_rows(indices: torch.Tensor, *, k_num: int, num_rows: int, kind: str) -> torch.Tensor:
    """Remove self-neighbors, invalid ids, and duplicates from a loaded neighbor table."""
    cleaned_rows = []
    needs_recompute = False

    for row_idx in range(num_rows):
        row = indices[row_idx].tolist()
        seen = set()
        cleaned = []

        for neighbor in row:
            neighbor = int(neighbor)
            if neighbor < 0 or neighbor >= num_rows:
                needs_recompute = True
                continue
            if neighbor == row_idx:
                needs_recompute = True
                continue
            if kind == "item" and row_idx != 0 and neighbor == 0:
                needs_recompute = True
                continue
            if neighbor in seen:
                needs_recompute = True
                continue
            seen.add(neighbor)
            cleaned.append(neighbor)
            if len(cleaned) == k_num:
                break

        if len(cleaned) < k_num:
            needs_recompute = True
            break

        cleaned_rows.append(cleaned)

    if needs_recompute:
        raise ValueError(f"Loaded {kind} neighbor table is stale or invalid and must be recomputed")

    return torch.tensor(cleaned_rows, dtype=torch.long)


def _load_or_recompute_neighbors(
    *,
    path: Optional[str],
    emb: torch.Tensor,
    k_num: int,
    kind: str,
) -> torch.Tensor:
    if kind not in {"item", "user"}:
        raise ValueError(f"Unsupported neighbor kind: {kind}")

    expected_rows = emb.shape[0]
    recompute_msg = None
    indices = None

    if path and os.path.exists(path):
        loaded = torch.from_numpy(np.load(path)).long()
        try:
            loaded = _align_neighbor_width(loaded, k_num)
            if loaded.shape[0] != expected_rows:
                raise ValueError(
                    f"row count mismatch: loaded={loaded.shape[0]} expected={expected_rows}"
                )
            indices = _sanitize_neighbor_rows(
                loaded,
                k_num=k_num,
                num_rows=expected_rows,
                kind=kind,
            )
        except ValueError as exc:
            recompute_msg = str(exc)

    if indices is not None:
        return indices

    if recompute_msg:
        print(f"Warning: {kind} neighbor table check failed ({recompute_msg}); recomputing from semantic embeddings.")

    if kind == "item":
        return _compute_item_topk_excluding_padding(emb, k_num)
    return _compute_topk_from_embeddings(emb, k_num)


def prepare_semantic_assets(args, user_train: Dict[int, list]):
    if not getattr(args, "enable_semantic_module", False):
        return

    required_paths = {
        "item_semantic": _resolve_path(args, "sa_item_semantic", SEMANTIC_FILE_HINTS["item_semantic"]),
        "user_semantic": _resolve_path(args, "sa_user_semantic", SEMANTIC_FILE_HINTS["user_semantic"]),
        "seq_keys": _resolve_path(args, "sa_seq_keys", SEMANTIC_FILE_HINTS["seq_keys"]),
    }

    missing = [key for key, path in required_paths.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing semantic assets. Provide --sa_asset_root or explicit sa_* paths: "
            + ", ".join(missing)
        )

    item_semantic_emb = _load_tensor_file(required_paths["item_semantic"])
    user_semantic_emb = _load_tensor_file(required_paths["user_semantic"])
    seq_keys_to_int = _load_pickle(required_paths["seq_keys"])
    if not isinstance(seq_keys_to_int, dict):
        raise ValueError("seq_keys_to_int.pkl content must be dict")

    seq_int_to_keys = {v: k for k, v in seq_keys_to_int.items()}
    if user_semantic_emb.shape[0] != len(seq_keys_to_int):
        raise ValueError(
            "user_semantic_embeddings.pt and seq_keys_to_int.pkl are inconsistent: "
            f"user_semantic_rows={user_semantic_emb.shape[0]}, seq_keys={len(seq_keys_to_int)}. "
            "Regenerate semantic alignment assets from the current dataset split before training."
        )

    item_neighbors_path = _resolve_path(args, "sa_item_neighbors", SEMANTIC_FILE_HINTS["item_neighbors"])
    user_neighbors_path = _resolve_path(args, "sa_user_neighbors", SEMANTIC_FILE_HINTS["user_neighbors"])
    k_num = getattr(args, "sa_k_num", 10)

    item_sorted_indices = _load_or_recompute_neighbors(
        path=item_neighbors_path,
        emb=item_semantic_emb,
        k_num=k_num,
        kind="item",
    )
    user_sorted_indices = _load_or_recompute_neighbors(
        path=user_neighbors_path,
        emb=user_semantic_emb,
        k_num=k_num,
        kind="user",
    )

    args.item_semantic_emb = item_semantic_emb
    args.user_semantic_emb = user_semantic_emb
    args.seq_keys_to_int = seq_keys_to_int
    args.seq_int_to_keys = seq_int_to_keys
    args.sorted_indices_numpy = item_sorted_indices.long().numpy()
    args.user_sorted_indices_numpy = user_sorted_indices.long().numpy()
    args.sorted_indices_tensor = item_sorted_indices.long()
    args.user_sorted_indices_tensor = user_sorted_indices.long()


def ensure_seq_keys_exist(args, user_train: Dict[int, list]):
    seq_keys_to_int = {}
    seq_int_to_keys = {}
    idx = 0
    maxlen = getattr(args, "maxlen", 50)
    for uid, seq in user_train.items():
        seq_start = 0
        for i in range(1, len(seq)):
            if i - seq_start > maxlen:
                seq_start += 1
            key = ":".join(map(str, [uid] + seq[seq_start:i]))
            if key not in seq_keys_to_int:
                seq_keys_to_int[key] = idx
                seq_int_to_keys[idx] = key
                idx += 1
    args.seq_keys_to_int = seq_keys_to_int
    args.seq_int_to_keys = seq_int_to_keys
    return seq_keys_to_int
