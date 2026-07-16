import gzip
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


SUPPORTED_DATASETS = {
    "CDs_and_Vinyl": {"sample_rate": 0.33},
    "Movies_and_TV": {"sample_rate": 0.05},
    "Electronics": {"sample_rate": 0.05},
    "Industrial_and_Scientific": {"sample_rate": 1.0},
}


def _normal_text(value, default):
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0] if value else ""
    value = str(value).strip()
    return value or default


def _description(value):
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item)
    return _normal_text(value, "Empty description")


def _output_dir(dataset_name):
    root = os.environ.get("ECHOREC_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
    path = os.path.join(root, dataset_name)
    os.makedirs(path, exist_ok=True)
    return path


def _load_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_local_split(raw_dir, dataset_name, split):
    path = os.path.join(raw_dir, f"{dataset_name}_{split}.jsonl")
    if not os.path.exists(path):
        path_gz = path + ".gz"
        if os.path.exists(path_gz):
            path = path_gz
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing local split file: {path}")
    return list(_load_jsonl(path))


def _load_local_meta(raw_dir, dataset_name):
    path = os.path.join(raw_dir, f"meta_{dataset_name}.jsonl")
    if not os.path.exists(path):
        path_gz = path + ".gz"
        if os.path.exists(path_gz):
            path = path_gz
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing local metadata file: {path}")
    return list(_load_jsonl(path))


def _load_amazon_2023(dataset_name):
    if os.environ.get("ECHOREC_LOCAL_DATA", ""):
        raw_dir = os.environ["ECHOREC_LOCAL_DATA"]
        splits = {split: _load_local_split(raw_dir, dataset_name, split) for split in ("train", "valid", "test")}
        meta_rows = _load_local_meta(raw_dir, dataset_name)
        return splits, meta_rows

    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"5core_last_out_{dataset_name}",
        trust_remote_code=True,
    )
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{dataset_name}",
        trust_remote_code=True,
    )
    return {split: reviews[split] for split in ("train", "valid", "test")}, meta["full"]


def _meta_table(meta_rows):
    table = {}
    for row in tqdm(meta_rows, desc="Loading metadata"):
        asin = row.get("parent_asin") or row.get("asin")
        if not asin:
            continue
        table[asin] = (
            _normal_text(row.get("title"), "Empty title"),
            _description(row.get("description")),
        )
    return table


def preprocess_raw_5core(dataset_name):
    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {supported}")

    random.seed(0)
    np.random.seed(0)

    splits, meta = _load_amazon_2023(dataset_name)
    meta_dict = _meta_table(meta)
    sample_rate = SUPPORTED_DATASETS[dataset_name]["sample_rate"]

    usermap = {}
    itemmap = {}
    user_split = {split: defaultdict(list) for split in ("train", "valid", "test")}
    time_dict = defaultdict(dict)
    id2asin = {}

    for split in ("train", "valid", "test"):
        for row in tqdm(splits[split], desc=f"Reading {split}"):
            user_raw = row.get("user_id")
            asin = row.get("parent_asin")
            if not user_raw or not asin:
                continue
            user_id = usermap.setdefault(user_raw, len(usermap) + 1)
            item_id = itemmap.setdefault(asin, len(itemmap) + 1)
            user_split[split][user_id].append(item_id)
            id2asin[item_id] = asin
            time_dict[item_id][user_id] = row.get("timestamp", 0)

    users = sorted(set().union(*(set(values.keys()) for values in user_split.values())))
    keep_count = max(1, int(len(users) * sample_rate))
    keep_users = set(random.sample(users, keep_count))

    count_u = defaultdict(int)
    count_i = defaultdict(int)
    for user_id in keep_users:
        for split in ("train", "valid", "test"):
            for item_id in user_split[split][user_id]:
                count_u[user_id] += 1
                count_i[item_id] += 1

    out_dir = _output_dir(dataset_name)
    text_dict = {"time": defaultdict(dict), "description": {}, "title": {}}
    usermap_final = {}
    itemmap_final = {}
    keep_train = set()

    for split in ("train", "valid", "test"):
        path = os.path.join(out_dir, f"{dataset_name}_{split}.txt")
        with open(path, "w", encoding="utf-8") as stream:
            for user_id in tqdm(users, desc=f"Writing {split}"):
                if user_id not in keep_users or count_u[user_id] <= 4:
                    continue
                items = [item for item in user_split[split][user_id] if count_i[item] > 4]
                if split == "train":
                    if len(items) <= 4:
                        continue
                    keep_train.add(user_id)
                elif user_id not in keep_train:
                    continue
                if not items:
                    continue

                final_user = usermap_final.setdefault(user_id, len(usermap_final) + 1)
                for source_item in items:
                    final_item = itemmap_final.setdefault(source_item, len(itemmap_final) + 1)
                    title, desc = meta_dict.get(id2asin[source_item], ("Empty title", "Empty description"))
                    text_dict["title"][final_item] = title
                    text_dict["description"][final_item] = desc
                    text_dict["time"][final_item][final_user] = time_dict[source_item][user_id]
                    stream.write(f"{final_user} {final_item}\n")

    with open(os.path.join(out_dir, "text_name_dict.json.gz"), "wb") as stream:
        pickle.dump(text_dict, stream)

    print(f"Prepared {dataset_name}: users={len(usermap_final)}, items={len(itemmap_final)}, output={out_dir}")
    return out_dir
