import os
import random
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device(args) -> torch.device:
    if str(args.device).lower() == "cpu" or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.device}")
    return args.device


def _dataset_dir(data_root: str, dataset_name: str):
    candidates = [
        os.path.join(data_root, dataset_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def data_partition_si(dataset_name: str, data_root: str = "./datasets"):
    base = _dataset_dir(data_root, dataset_name)
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)
    usernum = 0
    itemnum = 0

    for split, target in (("train", user_train), ("valid", user_valid), ("test", user_test)):
        path = os.path.join(base, f"{dataset_name}_{split}.txt")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user, item = map(int, line.rstrip().split(" "))
                target[user].append(item)
                usernum = max(usernum, user)
                itemnum = max(itemnum, item)
    return user_train, user_valid, user_test, usernum, itemnum


def build_seq_for_val(user_train, uid, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for item in reversed(user_train[uid]):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break
    return seq


def build_seq_for_test(user_train, user_valid, uid, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    if len(user_valid.get(uid, [])) > 0:
        seq[idx] = user_valid[uid][0]
        idx -= 1
    for item in reversed(user_train[uid]):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break
    return seq


def make_candidate(seq, target_id, itemnum, num_neg=99):
    seen = set(seq[seq > 0].tolist())
    seen.update({0, target_id})
    neg_ids = []
    while len(neg_ids) < num_neg:
        candidate = np.random.randint(1, itemnum + 1)
        if candidate not in seen and candidate not in neg_ids:
            neg_ids.append(candidate)
    random.shuffle(neg_ids)
    return [target_id] + neg_ids[:num_neg]


def evaluate_sampled(model, user_list, user_train, user_valid, user_test, itemnum, maxlen, split="test"):
    model.eval()
    hr10 = ndcg10 = hr20 = ndcg20 = users = 0.0
    with torch.no_grad():
        for uid in user_list:
            if split == "test":
                if len(user_test.get(uid, [])) < 1:
                    continue
                seq = build_seq_for_test(user_train, user_valid, uid, maxlen)
                target = user_test[uid][0]
            else:
                if len(user_valid.get(uid, [])) < 1:
                    continue
                seq = build_seq_for_val(user_train, uid, maxlen)
                target = user_valid[uid][0]
            if target <= 0:
                continue

            candidates = np.array([make_candidate(seq, target, itemnum)], dtype=np.int64)
            scores = model.predict(np.array([uid]), np.array([seq], dtype=np.int64), candidates)
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            rank = int((scores[0] > scores[0, 0]).sum())
            users += 1
            if rank < 10:
                hr10 += 1
                ndcg10 += 1.0 / np.log2(rank + 2)
            if rank < 20:
                hr20 += 1
                ndcg20 += 1.0 / np.log2(rank + 2)

    denom = max(users, 1.0)
    return {
        "HR@10": round(hr10 / denom, 4),
        "NDCG@10": round(ndcg10 / denom, 4),
        "HR@20": round(hr20 / denom, 4),
        "NDCG@20": round(ndcg20 / denom, 4),
        "users": int(users),
    }


def run_evaluation(model, si_data, args, split):
    user_train, user_valid, user_test, _, itemnum = si_data
    source = user_valid if split == "val" else user_test
    users = [user for user in source if len(source[user]) >= 1]
    return evaluate_sampled(model, users, user_train, user_valid, user_test, itemnum, args.maxlen, split=split)


def append_eval_results(results_path: str, epoch: int, metrics: Dict[str, float], split: str):
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    metric_str = "\t".join(
        [f"{key}={value:.6f}" for key, value in metrics.items() if isinstance(value, (int, float))]
    )
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(f"epoch={epoch}\t{split}\t{metric_str}\t{timestamp}\n")


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save([model.kwargs, model.state_dict()], path)
