import os
import random
import time
from collections import defaultdict
from typing import Dict, Tuple

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
    raw_device = args.device
    args.is_hpu = False
    args._htcore = None
    if raw_device.lower() == "cpu":
        args.device = torch.device("cpu")
        return args.device

    if raw_device.lower() == "hpu":
        args.device = torch.device("hpu")
        args.nn_parameter = True
        args.is_hpu = True
        try:
            import habana_frameworks.torch.core as htcore  # type: ignore

            args._htcore = htcore
        except ImportError:
            args._htcore = None
        return args.device

    if not torch.cuda.is_available():
        args.device = torch.device("cpu")
        return args.device

    args.device = torch.device(f"cuda:{raw_device}")
    return args.device


def data_partition_si(dataset_name: str, data_root: str = "./SeqRec"):
    base = os.path.join(data_root, f"data_{dataset_name}")
    usernum = 0
    itemnum = 0
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    for split in ["train", "valid", "test"]:
        fpath = os.path.join(base, f"{dataset_name}_{split}.txt")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                u, i = line.rstrip().split(" ")
                u, i = int(u), int(i)
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                if split == "train":
                    user_train[u].append(i)
                elif split == "valid":
                    user_valid[u].append(i)
                else:
                    user_test[u].append(i)

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
    try:
        seq[idx] = user_valid[uid][0]
        idx -= 1
    except (KeyError, IndexError):
        pass
    for item in reversed(user_train[uid]):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break
    return seq


def make_candidate_si(seq, target_id, itemnum, num_neg=99):
    history = set(seq[seq > 0].tolist())
    history.add(0)
    history.add(target_id)

    neg_ids = []
    while len(neg_ids) < num_neg:
        cand = np.random.randint(1, itemnum + 1)
        if cand not in history and cand not in neg_ids:
            neg_ids.append(cand)

    random.shuffle(neg_ids)
    return [target_id] + neg_ids[:num_neg]


def evaluate_sampled_si(
    model,
    user_list,
    user_train,
    user_valid,
    user_test,
    itemnum,
    maxlen,
    split="test",
    top_k=10,
    num_neg=99,
):
    model.eval()
    hr_list, ndcg_list = [], []
    hr20_list, ndcg20_list = [], []

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

            candidates = make_candidate_si(seq, target, itemnum, num_neg)
            candidates_arr = np.array([candidates], dtype=np.int64)
            user_arr = np.array([uid], dtype=np.int64)
            seq_arr = np.array([seq], dtype=np.int64)

            scores = model.predict(user_arr, seq_arr, candidates_arr)
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()

            rank = int((scores[0] > scores[0, 0]).sum())

            if rank < top_k:
                hr_list.append(1)
                ndcg_list.append(1.0 / np.log2(rank + 2))
            else:
                hr_list.append(0)
                ndcg_list.append(0.0)

            if rank < 20:
                hr20_list.append(1)
                ndcg20_list.append(1.0 / np.log2(rank + 2))
            else:
                hr20_list.append(0)
                ndcg20_list.append(0.0)

    n = len(hr_list) if hr_list else 1
    return {
        f"HR@{top_k}": round(np.sum(hr_list) / n, 4),
        f"NDCG@{top_k}": round(np.sum(ndcg_list) / n, 4),
        "HR@20": round(np.sum(hr20_list) / n, 4),
        "NDCG@20": round(np.sum(ndcg20_list) / n, 4),
        "users": n,
    }


def run_evaluation(model, si_data, args, split):
    si_user_train, si_user_valid, si_user_test, _, si_itemnum = si_data
    if split == "val":
        user_list = [u for u in si_user_valid if len(si_user_valid[u]) >= 1]
    else:
        user_list = [u for u in si_user_test if len(si_user_test[u]) >= 1]
    return evaluate_sampled_si(
        model,
        user_list,
        si_user_train,
        si_user_valid,
        si_user_test,
        si_itemnum,
        args.maxlen,
        split=split,
        top_k=10,
        num_neg=99,
    )


def append_eval_results(results_path: str, epoch: int, metrics: Dict[str, float], split: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    metric_str = "\t".join([f"{k}={v:.6f}" for k, v in metrics.items()])
    line = f"epoch={epoch}\t{split}\t{metric_str}\t{ts}\n"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(line)


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save([model.kwargs, model.state_dict()], path)


def random_neq(left: int, right: int, seen: set):
    value = np.random.randint(left, right)
    while value in seen:
        value = np.random.randint(left, right)
    return value


def build_prefix_examples(user_train, maxlen: int):
    uid_list, item_list, target_list, item_list_length = [], [], [], []
    for uid, item_id_seq in user_train.items():
        seq_start = 0
        for idx in range(1, len(item_id_seq)):
            if idx - seq_start > maxlen:
                seq_start += 1
            uid_list.append(uid)
            item_list.append(item_id_seq[seq_start:idx])
            target_list.append(item_id_seq[idx])
            item_list_length.append(idx - seq_start)
    return uid_list, item_list, target_list, item_list_length


class PrefixDataset(torch.utils.data.Dataset):
    def __init__(self, uid_list, item_list, target_list, item_list_length, maxlen: int):
        self.uid_list = uid_list
        self.item_list = item_list
        self.target_list = target_list
        self.item_list_length = item_list_length
        self.maxlen = maxlen

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        seq = self.item_list[idx]
        padded_seq = np.zeros(self.maxlen, dtype=np.int32)
        clipped_seq = seq[-self.maxlen :]
        padded_seq[-len(clipped_seq) :] = clipped_seq
        return (
            torch.tensor(self.uid_list[idx], dtype=torch.long),
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(self.target_list[idx], dtype=torch.long),
            torch.tensor(self.item_list_length[idx], dtype=torch.long),
        )


def build_sasrec_supervision_from_prefix(
    user_ids: np.ndarray,
    seq_batch: np.ndarray,
    target_batch: np.ndarray,
    user_train,
    itemnum: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pos_batch = np.zeros_like(seq_batch, dtype=np.int32)
    neg_batch = np.zeros_like(seq_batch, dtype=np.int32)

    for row_idx, (uid, seq, target) in enumerate(zip(user_ids.tolist(), seq_batch, target_batch.tolist())):
        nonzero_positions = np.where(seq > 0)[0]
        if nonzero_positions.size == 0:
            continue

        seq_items = seq[nonzero_positions].tolist()
        next_items = seq_items[1:] + [int(target)]
        pos_batch[row_idx, nonzero_positions] = np.asarray(next_items, dtype=np.int32)

        rated = set(user_train[int(uid)])
        for col_idx, pos_item in zip(nonzero_positions, next_items):
            if pos_item != 0:
                neg_batch[row_idx, col_idx] = random_neq(1, itemnum + 1, rated)

    return pos_batch, neg_batch


class SASRecBatchSampler:
    def __init__(self, user_train, usernum: int, itemnum: int, batch_size: int, maxlen: int):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.valid_users = [u for u in range(1, usernum + 1) if len(user_train[u]) > 1]
        if not self.valid_users:
            raise RuntimeError("No valid users with sequence length > 1 found for SASRec sampling.")

    def _sample_one(self):
        user = random.choice(self.valid_users)
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1
        ts = set(self.user_train[user])

        for item in reversed(self.user_train[user][:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = item
            idx -= 1
            if idx == -1:
                break

        return user, seq, pos, neg

    def next_batch(self):
        users, seqs, poss, negs = [], [], [], []
        for _ in range(self.batch_size):
            user, seq, pos, neg = self._sample_one()
            users.append(user)
            seqs.append(seq)
            poss.append(pos)
            negs.append(neg)
        return (
            np.asarray(users, dtype=np.int64),
            np.asarray(seqs, dtype=np.int32),
            np.asarray(poss, dtype=np.int32),
            np.asarray(negs, dtype=np.int32),
        )


def sasrec_bce_loss(
    model: nn.Module,
    user_ids: np.ndarray,
    seq_batch: np.ndarray,
    pos_batch: np.ndarray,
    neg_batch: np.ndarray,
    device: torch.device,
    bce_criterion: nn.Module,
    l2_emb: float,
) -> torch.Tensor:
    pos_logits, neg_logits = model(user_ids, seq_batch, pos_batch, neg_batch)
    pos_labels = torch.ones(pos_logits.shape, device=device)
    neg_labels = torch.zeros(neg_logits.shape, device=device)
    indices = np.where(pos_batch != 0)
    if len(indices[0]) == 0:
        return torch.tensor(0.0, device=device)

    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
    loss = loss + bce_criterion(neg_logits[indices], neg_labels[indices])

    if l2_emb > 0:
        if getattr(model, "nn_parameter", False):
            loss = loss + l2_emb * torch.norm(model.item_emb)
        else:
            for param in model.item_emb.parameters():
                loss = loss + l2_emb * torch.norm(param)
    return loss
