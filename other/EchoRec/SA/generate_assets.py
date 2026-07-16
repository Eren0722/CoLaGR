import os
import pickle
import argparse
import gzip
from typing import Dict, List
from argparse import Namespace

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from SA.dataset import data_partition
from SA.assets import ensure_seq_keys_exist, _compute_topk_from_embeddings, _compute_item_topk_excluding_padding
from models.echorec_llm import EchoRecLLM as StageOneLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate semantic alignment assets")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., Movies_and_TV")
    parser.add_argument("--data_root", default="./SeqRec", help="Root directory that contains data_<dataset>")
    parser.add_argument("--asset_root", default="./SA_assets", help="Directory to store generated assets")

    parser.add_argument("--llm", choices=["llama", "llama-3b"], default="llama", help="LLM alias to reuse from stage one")
    parser.add_argument("--llm_path", default=None, help="Local HuggingFace checkpoint directory for llama models")
    parser.add_argument("--device", default="cuda:0", help="Device for embedding generation (e.g., cuda:0 or cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for embedding inference")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokens per prompt (tokenizer truncation)")
    parser.add_argument("--maxlen", type=int, default=128, help="Sequence history truncation length (must match train_sa --maxlen)")
    parser.add_argument("--history_trunc", type=int, default=10, help="Max history length when describing a sequence")
    parser.add_argument("--neighbor_k", type=int, default=10, help="Top-K neighbors to precompute (optional)")

    parser.add_argument("--skip_neighbors", action="store_true", help="Do not precompute neighbor indices")

    return parser.parse_args()


def load_llm_from_module_one(llm_name: str, device_str: str):
    dummy_args = Namespace(token=False, nn_parameter=False)
    module = StageOneLLM(device=device_str, llm_model=llm_name, args=dummy_args)
    module.eval()
    return module.llm_tokenizer, module.llm_model


def _resolve_hf_snapshot_dir(path: str) -> str:
    """If user passes the root cache folder, jump into the actual snapshot."""
    if not os.path.isdir(path):
        return path

    # Already a valid checkpoint directory?
    if os.path.isfile(os.path.join(path, "config.json")):
        return path

    snapshots_dir = os.path.join(path, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return path

    # Prefer refs/main pointer; fallback to the newest snapshot folder
    ref_main = os.path.join(path, "refs", "main")
    if os.path.isfile(ref_main):
        with open(ref_main, "r", encoding="utf-8") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    candidates = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)]
    candidates = [d for d in candidates if os.path.isdir(d)]
    if candidates:
        candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        return candidates[0]

    return path


def load_llm_from_local_path(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"指定的 LLM 缓存路径不存在: {path}")
    path = _resolve_hf_snapshot_dir(path)
    dtype = torch.float16 if device.type != "cpu" and torch.cuda.is_available() else torch.float32
    kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    use_device_map = device.type != "cpu" and torch.cuda.is_available()
    if use_device_map:
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModel.from_pretrained(path, **kwargs)

    # Align tokenizer special tokens with module-one settings to enable padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'bos_token': '</s>'})
    tokenizer.add_special_tokens({'eos_token': '</s>'})
    tokenizer.add_special_tokens({'unk_token': '</s>'})
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[UserRep]', '[HistoryEmb]', '[UserOut]', '[ItemOut]']
    })
    model.resize_token_embeddings(len(tokenizer))

    if not use_device_map:
        model = model.to(device)
    model.eval()
    return tokenizer, model


def load_llm(args: argparse.Namespace, device: torch.device):
    if args.llm_path:
        return load_llm_from_local_path(args.llm_path, device)
    return load_llm_from_module_one(args.llm, args.device)


def load_text_dict(dataset: str, data_root: str) -> Dict[str, Dict]:
    meta_path = os.path.join(data_root, f"data_{dataset}", "text_name_dict.json.gz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"找不到 text_name_dict.json.gz，期望路径: {meta_path}. 请先确保数据准备完成。"
        )

    try:
        with gzip.open(meta_path, "rb") as f:
            return pickle.load(f)
    except (OSError, gzip.BadGzipFile):
        with open(meta_path, "rb") as f:
            return pickle.load(f)


def safe_lookup(mapping: Dict, key: int, default: str = "") -> str:
    if isinstance(mapping, dict):
        if key in mapping:
            return mapping[key]
        str_key = str(key)
        if str_key in mapping:
            return mapping[str_key]
    return default


def build_item_prompts(item_num: int, text_dict: Dict[str, Dict]) -> List[str]:
    titles = text_dict.get("title", {})
    descs = text_dict.get("description", {})
    prompts = ["Padding item"] * (item_num + 1)
    for item_id in range(1, item_num + 1):
        title = safe_lookup(titles, item_id, "Unknown Title")
        desc = safe_lookup(descs, item_id, "No Description available")
        prompts[item_id] = f"Item {item_id}: {title}. Description: {desc}."
    return prompts


def build_sequence_prompts(seq_keys_to_int: Dict[str, int], text_dict: Dict[str, Dict], history_trunc: int) -> List[str]:
    titles = text_dict.get("title", {})
    seq_prompts = [""] * len(seq_keys_to_int)
    for seq_key, idx in seq_keys_to_int.items():
        parts = seq_key.split(":")
        if not parts:
            seq_prompts[idx] = "Empty sequence"
            continue
        user_id = parts[0]
        item_ids = [int(x) for x in parts[1:] if x]
        history_titles = []
        for iid in item_ids[-history_trunc:]:
            history_titles.append(safe_lookup(titles, iid, f"Item {iid}"))
        if not history_titles:
            prompt = f"User {user_id} history is empty."
        else:
            joined = ", ".join(history_titles)
            prompt = f"User {user_id} recently interacted with: {joined}."
        seq_prompts[idx] = prompt
    return seq_prompts


def ensure_output_dir(asset_root: str, dataset: str) -> str:
    out_dir = os.path.join(asset_root, dataset)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def encode_texts(texts: List[str], tokenizer, model, device: torch.device,
                 batch_size: int, max_length: int, desc: str) -> torch.Tensor:
    outputs = []
    total_steps = (len(texts) + batch_size - 1) // batch_size
    for start in tqdm(range(0, len(texts), batch_size), total=total_steps, desc=desc):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            result = model(**enc)          # AutoModel: 无 lm_head, 不计算 logits
            hidden = result.last_hidden_state  # (batch, seq_len, hidden_dim)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        outputs.append(pooled.cpu())
        del result, hidden, enc
        torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = data_partition(args.dataset, root_dir=args.data_root)
    user_train, _, _, usernum, itemnum, _ = dataset

    text_dict = load_text_dict(args.dataset, args.data_root)
    seq_keys_args = argparse.Namespace(maxlen=args.maxlen)
    ensure_seq_keys_exist(seq_keys_args, user_train)
    seq_keys_to_int = seq_keys_args.seq_keys_to_int

    output_dir = ensure_output_dir(args.asset_root, args.dataset)
    seq_keys_path = os.path.join(output_dir, "seq_keys_to_int.pkl")
    with open(seq_keys_path, "wb") as f:
        pickle.dump(seq_keys_to_int, f)
    print(f"✔ 保存 seq_keys_to_int 到 {seq_keys_path}, 总计 {len(seq_keys_to_int)} 条")

    tokenizer, model = load_llm(args, device)

    # Item embeddings
    item_prompts = build_item_prompts(itemnum, text_dict)
    print(f"开始生成 {len(item_prompts) - 1} 个物品语义向量…")
    item_emb = encode_texts(item_prompts, tokenizer, model, device, args.batch_size, args.max_length, desc="Items")
    item_emb_path = os.path.join(output_dir, "item_semantic_embeddings.pt")
    torch.save(item_emb, item_emb_path)
    print(f"✔ 保存 item_semantic_embeddings: {item_emb.shape} -> {item_emb_path}")

    # Sequence/user embeddings
    seq_prompts = build_sequence_prompts(seq_keys_to_int, text_dict, args.history_trunc)
    print(f"开始生成 {len(seq_prompts)} 条序列语义向量…")
    seq_emb = encode_texts(seq_prompts, tokenizer, model, device, args.batch_size, args.max_length, desc="Sequences")
    user_emb_path = os.path.join(output_dir, "user_semantic_embeddings.pt")
    torch.save(seq_emb, user_emb_path)
    print(f"✔ 保存 user_semantic_embeddings: {seq_emb.shape} -> {user_emb_path}")

    if not args.skip_neighbors:
        print(f"计算 Top-{args.neighbor_k} 近邻…")
        item_neighbors = _compute_item_topk_excluding_padding(item_emb, args.neighbor_k)
        np.save(os.path.join(output_dir, "item_sorted_indices.npy"), item_neighbors.numpy())
        user_neighbors = _compute_topk_from_embeddings(seq_emb, args.neighbor_k)
        np.save(os.path.join(output_dir, "user_sorted_indices.npy"), user_neighbors.numpy())
        print("✔ 近邻索引保存完成")

    print("所有语义资产生成完成！")


if __name__ == "__main__":
    main()
