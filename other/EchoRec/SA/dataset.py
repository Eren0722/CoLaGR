import os
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


def calculate_valid_score(valid_result, valid_metric=None):
    if valid_metric:
        return valid_result[valid_metric]
    return valid_result["Recall@10"]


def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError("Training loss is nan")


class SequenceDataset(Dataset):
    """Dataset with semantic alignment augmentations."""

    def __init__(self, args, uid_list, item_list, target_list, item_list_length, seq_unique_key_list, maxlen):
        self.item_list = item_list
        self.target_list = target_list
        self.item_list_length = item_list_length
        self.uid_list = uid_list
        self.maxlen = maxlen
        self.mlm_probability = getattr(args, "sa_mlm_probability", 0.2)
        self.neighbor_matrix = args.sorted_indices_numpy
        self.user_neighbor_matrix = args.user_sorted_indices_numpy
        self.seq_unique_key_list = seq_unique_key_list
        self.seq_int_to_keys = args.seq_int_to_keys

    def __len__(self):
        return len(self.item_list)

    def recall_similar_seqs(self, neighbor_matrix, seq_unique_id):
        neighbors = neighbor_matrix[seq_unique_id]
        neighbor_seq_list = []
        for neighbor_seq_id in neighbors:
            neighbor_seq_key = self.seq_int_to_keys[neighbor_seq_id]
            neighbor_seq_key = neighbor_seq_key.split(":")
            neighbor_seq_key = list(map(int, neighbor_seq_key))
            neighbor_seq = neighbor_seq_key[1:]
            neighbor_seq = self.padding_and_truncation(neighbor_seq)
            neighbor_seq_list.append(neighbor_seq)
        return np.array(neighbor_seq_list)

    def replace_input_ids(self, input_ids, p, neighbor_matrix):
        replaced_input_ids = []
        for token in input_ids:
            if token == 0:
                replaced_input_ids.append(token)
            else:
                if random.random() < p:
                    neighbors = neighbor_matrix[token]
                    if len(neighbors) > 0:
                        replaced_token = random.choice(neighbors)
                    else:
                        replaced_token = token
                    replaced_input_ids.append(replaced_token)
                else:
                    replaced_input_ids.append(token)
        return replaced_input_ids

    def padding_and_truncation(self, seq):
        length = len(seq)
        if length < self.maxlen:
            padded_seq = np.zeros(self.maxlen, dtype=np.int32)
            padded_seq[-length:] = seq[:]
        else:
            padded_seq = seq
        padded_seq = padded_seq[-self.maxlen:]
        return padded_seq

    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.item_list[idx]
        target = self.target_list[idx]
        length = self.item_list_length[idx]
        seq_unique_id = self.seq_unique_key_list[idx]

        padded_seq = self.padding_and_truncation(seq)
        similar_seqs = self.recall_similar_seqs(self.user_neighbor_matrix, seq_unique_id)
        aug_seq1 = self.replace_input_ids(padded_seq, self.mlm_probability, self.neighbor_matrix)
        aug_seq2 = self.replace_input_ids(padded_seq, self.mlm_probability, self.neighbor_matrix)

        return (
            torch.tensor(seq_unique_id, dtype=torch.long),
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(aug_seq1, dtype=torch.long),
            torch.tensor(aug_seq2, dtype=torch.long),
            torch.tensor(similar_seqs, dtype=torch.long),
            torch.tensor(uid, dtype=torch.long),
        )


class TestDataset(Dataset):
    def __init__(self, item_list, max_seq_len):
        self.uid_list, self.item_list, self.target_list, self.item_list_length = self.data_process(item_list)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.item_list[idx]
        target = self.target_list[idx]
        length = self.item_list_length[idx]
        if length < self.max_seq_len:
            padded_seq = np.zeros(self.max_seq_len, dtype=np.int32)
            padded_seq[-length:] = seq[:]
        else:
            padded_seq = seq
        padded_seq = padded_seq[-self.max_seq_len:]
        return (
            torch.tensor(uid, dtype=torch.long),
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )

    def data_process(self, data_dict):
        uid_list, item_list, target_list, item_list_length = [], [], [], []
        for uid, item_id_seq in data_dict.items():
            if len(item_id_seq) > 1:
                uid_list.append(uid)
                item_list.append(item_id_seq[:-1])
                target_list.append(item_id_seq[-1])
                item_list_length.append(len(item_id_seq[:-1]))
        return uid_list, item_list, target_list, item_list_length


def data_partition(dataset_name, root_dir='./SeqRec'):
    train_file = os.path.join(root_dir, f'data_{dataset_name}', f'{dataset_name}_train.txt')
    valid_file = os.path.join(root_dir, f'data_{dataset_name}', f'{dataset_name}_valid.txt')
    test_file = os.path.join(root_dir, f'data_{dataset_name}', f'{dataset_name}_test.txt')

    usernum = 0
    itemnum = 0
    User = defaultdict(list)

    for file_path in [train_file, valid_file, test_file]:
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                u, i = line.rstrip().split(' ')
                u = int(u)
                i = int(i)
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                User[u].append(i)

    user_train = {}
    user_valid = {}
    user_test = {}

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = User[user][:-1]
            user_test[user] = User[user][:]
    eval_set = (set(user_valid.keys()), set(user_test.keys()))
    return [user_train, user_valid, user_test, usernum, itemnum, eval_set]


def data_augmentation(data_dict, max_seq_len, seq_keys_to_int):
    uid_list, item_list, target_list, item_list_length = [], [], [], []
    seq_unique_key_list = []
    missing_sequences = 0
    missing_examples = []

    for uid, item_id_seq in data_dict.items():
        seq_start = 0
        for i in range(1, len(item_id_seq)):
            if i - seq_start > max_seq_len:
                seq_start += 1
            seq_unique_key = ":".join(map(str, [uid] + item_id_seq[seq_start:i]))
            seq_id = seq_keys_to_int.get(seq_unique_key)
            if seq_id is None:
                missing_sequences += 1
                if len(missing_examples) < 5:
                    missing_examples.append(seq_unique_key)
                continue
            uid_list.append(uid)
            item_list.append(item_id_seq[seq_start:i])
            target_list.append(item_id_seq[i])
            item_list_length.append(i - seq_start)
            seq_unique_key_list.append(seq_id)

    if missing_sequences > 0:
        total_sequences = missing_sequences + len(seq_unique_key_list)
        missing_ratio = missing_sequences / max(total_sequences, 1)
        preview = ", ".join(missing_examples)
        raise RuntimeError(
            "data_augmentation found sequences missing from seq_keys_to_int. "
            f"missing={missing_sequences}, total={total_sequences}, ratio={missing_ratio:.2%}. "
            "This usually means the semantic alignment assets were generated from a different dataset split or stale data. "
            f"Examples: {preview}. Please regenerate semantic assets before training."
        )

    return uid_list, item_list, target_list, item_list_length, seq_unique_key_list


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / max(true_users, 1)


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / np.log2(j + 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / np.log2(i + 2) for i in range(k)])
    if not res:
        return 1.0
    return res


def mrr_at_k(actual, predicted, topk):
    sum_mrr = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        for rank, item in enumerate(predicted[i][:topk], start=1):
            if item in act_set:
                sum_mrr += 1.0 / rank
                break
    return sum_mrr / max(num_users, 1)


def get_full_sort_score(answers, pred_list):
    recall, ndcg, mrr = [], [], []
    for k in [5, 10, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
        mrr.append(mrr_at_k(answers, pred_list, k))
    return {
        "HIT@5": round(recall[0], 4),
        "NDCG@5": round(ndcg[0], 4),
        "MRR@5": round(mrr[0], 4),
        "HIT@10": round(recall[1], 4),
        "NDCG@10": round(ndcg[1], 4),
        "MRR@10": round(mrr[1], 4),
        "HIT@20": round(recall[2], 4),
        "NDCG@20": round(ndcg[2], 4),
        "MRR@20": round(mrr[2], 4),
    }
