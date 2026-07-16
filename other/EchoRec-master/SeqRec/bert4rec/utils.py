import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


def random_neq(left, right, seen):
    value = np.random.randint(left, right)
    while value in seen:
        value = np.random.randint(left, right)
    return value


class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len):
        del num_user
        self.user_train = user_train
        self.user_ids = sorted(user_train.keys())
        self.num_user = len(self.user_ids)
        self.num_item = num_item
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = self.user_train[user_id][-1]
        length_idx = self.max_len - 1
        seen = set(self.user_train[user_id])

        for item in reversed(self.user_train[user_id][:-1]):
            seq[length_idx] = item
            pos[length_idx] = nxt
            if nxt != 0:
                neg[length_idx] = random_neq(1, self.num_item + 1, seen)
            nxt = item
            length_idx -= 1
            if length_idx == -1:
                break
        return user_id, seq, pos, neg


class SeqDataset_Inference(Dataset):
    def __init__(self, user_train, user_valid, user_test, use_user, num_item, max_len):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user

    def __len__(self):
        return len(self.use_user)

    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        cursor = self.max_len - 1
        if len(self.user_valid.get(user_id, [])) > 0:
            seq[cursor] = self.user_valid[user_id][0]
            cursor -= 1
        for item in reversed(self.user_train[user_id]):
            seq[cursor] = item
            cursor -= 1
            if cursor == -1:
                break
        rated = set(self.user_train[user_id])
        rated.add(0)
        pos = self.user_test[user_id][0]
        neg = np.array([random_neq(1, self.num_item + 1, rated)])
        return user_id, seq, pos, neg


class SeqDataset_Validation(Dataset):
    def __init__(self, user_train, user_valid, use_user, num_item, max_len):
        self.user_train = user_train
        self.user_valid = user_valid
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user

    def __len__(self):
        return len(self.use_user)

    def __getitem__(self, idx):
        user_id = self.use_user[idx]
        seq = np.zeros([self.max_len], dtype=np.int32)
        cursor = self.max_len - 1
        for item in reversed(self.user_train[user_id]):
            seq[cursor] = item
            cursor -= 1
            if cursor == -1:
                break
        rated = set(self.user_train[user_id])
        rated.add(0)
        pos = self.user_valid[user_id][0]
        neg = np.array([random_neq(1, self.num_item + 1, rated)])
        return user_id, seq, pos, neg


def data_partition(fname, args, path=None):
    usernum = 0
    itemnum = 0
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    if path:
        base = path
    else:
        data_root = getattr(args, "data_root", "./datasets")
        candidates = [
            f"{data_root}/{args.rec_pre_trained_data}/{fname}",
        ]
        base = next((candidate for candidate in candidates if os.path.exists(f"{candidate}_train.txt")), candidates[0])
    for split, target in (("train", user_train), ("valid", user_valid), ("test", user_test)):
        with open(f"{base}_{split}.txt", "r", encoding="utf-8") as f:
            for line in f:
                user, item = map(int, line.rstrip().split(" "))
                target[user].append(item)
                usernum = max(usernum, user)
                itemnum = max(itemnum, item)
    eval_set = [set(user_valid.keys()), set(user_test.keys())]
    return [user_train, user_valid, user_test, usernum, itemnum, eval_set]
