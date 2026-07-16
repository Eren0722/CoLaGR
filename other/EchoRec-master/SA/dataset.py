import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, args, uid_list, item_list, target_list, item_list_length, seq_unique_key_list, maxlen):
        self.uid_list = uid_list
        self.item_list = item_list
        self.target_list = target_list
        self.item_list_length = item_list_length
        self.seq_unique_key_list = seq_unique_key_list
        self.maxlen = maxlen
        self.mlm_probability = args.sa_mlm_probability
        self.item_neighbors = args.sorted_indices_numpy
        self.user_neighbors = args.user_sorted_indices_numpy
        self.seq_int_to_keys = args.seq_int_to_keys

    def __len__(self):
        return len(self.item_list)

    def _pad(self, seq):
        padded = np.zeros(self.maxlen, dtype=np.int32)
        seq = seq[-self.maxlen:]
        padded[-len(seq):] = seq
        return padded

    def _replace_items(self, seq):
        replaced = []
        for token in seq:
            if token == 0 or np.random.random() >= self.mlm_probability:
                replaced.append(token)
                continue
            neighbors = self.item_neighbors[token] if token < len(self.item_neighbors) else []
            replaced.append(int(np.random.choice(neighbors)) if len(neighbors) > 0 else token)
        return replaced

    def _neighbor_sequences(self, seq_unique_id):
        sequences = []
        for neighbor_id in self.user_neighbors[seq_unique_id]:
            key = self.seq_int_to_keys[int(neighbor_id)]
            items = [int(item) for item in key.split(":")[1:] if item]
            sequences.append(self._pad(items))
        return np.array(sequences, dtype=np.int32)

    def __getitem__(self, idx):
        seq = self.item_list[idx]
        seq_unique_id = self.seq_unique_key_list[idx]
        padded_seq = self._pad(seq)
        return (
            torch.tensor(seq_unique_id, dtype=torch.long),
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(self.target_list[idx], dtype=torch.long),
            torch.tensor(self.item_list_length[idx], dtype=torch.long),
            torch.tensor(self._replace_items(padded_seq), dtype=torch.long),
            torch.tensor(self._replace_items(padded_seq), dtype=torch.long),
            torch.tensor(self._neighbor_sequences(seq_unique_id), dtype=torch.long),
            torch.tensor(self.uid_list[idx], dtype=torch.long),
        )


def _dataset_dir(root_dir, dataset_name):
    candidates = [
        os.path.join(root_dir, dataset_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def data_partition(dataset_name, root_dir="./datasets"):
    usernum = 0
    itemnum = 0
    user_sequences = defaultdict(list)
    base = _dataset_dir(root_dir, dataset_name)
    for split in ("train", "valid", "test"):
        path = os.path.join(base, f"{dataset_name}_{split}.txt")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                user, item = map(int, line.rstrip().split(" "))
                user_sequences[user].append(item)
                usernum = max(usernum, user)
                itemnum = max(itemnum, item)

    user_train, user_valid, user_test = {}, {}, {}
    for user, seq in user_sequences.items():
        if len(seq) < 3:
            user_train[user] = seq
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = seq[:-2]
            user_valid[user] = seq[:-1]
            user_test[user] = seq[:]
    return [user_train, user_valid, user_test, usernum, itemnum, (set(user_valid), set(user_test))]


def data_augmentation(data_dict, max_seq_len, seq_keys_to_int):
    uid_list, item_list, target_list, item_list_length, seq_unique_key_list = [], [], [], [], []
    missing = []
    for uid, item_id_seq in data_dict.items():
        seq_start = 0
        for idx in range(1, len(item_id_seq)):
            if idx - seq_start > max_seq_len:
                seq_start += 1
            seq = item_id_seq[seq_start:idx]
            key = ":".join(map(str, [uid] + seq))
            seq_id = seq_keys_to_int.get(key)
            if seq_id is None:
                if len(missing) < 5:
                    missing.append(key)
                continue
            uid_list.append(uid)
            item_list.append(seq)
            target_list.append(item_id_seq[idx])
            item_list_length.append(idx - seq_start)
            seq_unique_key_list.append(seq_id)
    if missing:
        raise RuntimeError("Semantic assets are stale. Regenerate SA assets before training. Examples: " + ", ".join(missing))
    return uid_list, item_list, target_list, item_list_length, seq_unique_key_list
