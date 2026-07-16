import ast
import csv
import gzip
import json
import os
import pickle
import random
import subprocess
import urllib.request
from collections import defaultdict

import numpy as np
from tqdm import tqdm


_HF_CONFIG_MAP = {
    'Beauty': 'Beauty_and_Personal_Care',
}

_AMAZON_2014_URLS = {
    'Beauty_2014': {
        'reviews': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz',
        'meta': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz',
    },
    'Office_2014': {
        'reviews': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Office_Products_5.json.gz',
        'meta': 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Office_Products.json.gz',
    },
}

_AMAZON_2018_URLS = {
    'Luxury_Beauty': {
        'reviews': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Luxury_Beauty.json.gz',
        'meta': 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Luxury_Beauty.json.gz',
    },
}


def _download_if_needed(url, dest):
    if os.path.exists(dest):
        print(f'  exists: {dest}')
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f'  download: {url} -> {dest}')

    try:
        subprocess.run(
            [
                'aria2c',
                '-c',
                '--auto-file-renaming=false',
                '--file-allocation=none',
                '-x',
                '16',
                '-s',
                '16',
                '-k',
                '1M',
                '--retry-wait=3',
                '--max-tries=0',
                '--timeout=120',
                '--connect-timeout=30',
                '-d',
                os.path.dirname(dest),
                '-o',
                os.path.basename(dest),
                url,
            ],
            timeout=7200,
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        print(f'  aria2c unavailable or failed ({exc}); trying wget')

    try:
        subprocess.run(
            ['wget', '--tries=5', '--timeout=120', '-O', dest, url],
            timeout=7200,
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        print(f'  wget unavailable or failed ({exc}); falling back to urllib')

    urllib.request.urlretrieve(url, dest)


def _load_json_lines(path):
    with open(path, 'rb') as probe:
        magic = probe.read(2)

    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def _parse_amazon_2014_gz(path):
    return list(_load_json_lines(path))


def _normalize_text(value, default):
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0] if value else ''
    value = str(value)
    value = value.strip()
    return value or default


def _build_output_dir(fname):
    out_dir = os.path.join(os.path.dirname(__file__), '..', f'data_{fname}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_leave_one_out(fname, user_seqs, itemmap, meta_dict):
    out_dir = _build_output_dir(fname)
    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}
    id2asin = {v: k for k, v in itemmap.items()}

    train_path = os.path.join(out_dir, f'{fname}_train.txt')
    valid_path = os.path.join(out_dir, f'{fname}_valid.txt')
    test_path = os.path.join(out_dir, f'{fname}_test.txt')

    with open(train_path, 'w', encoding='utf-8') as f_train, \
         open(valid_path, 'w', encoding='utf-8') as f_valid, \
         open(test_path, 'w', encoding='utf-8') as f_test:
        for userid, seq in user_seqs.items():
            if len(seq) < 3:
                continue
            for iid, _ in seq[:-2]:
                f_train.write(f'{userid} {iid}\n')
            iid_v, _ = seq[-2]
            iid_t, _ = seq[-1]
            f_valid.write(f'{userid} {iid_v}\n')
            f_test.write(f'{userid} {iid_t}\n')

            for iid, ts in seq:
                asin = id2asin[iid]
                title, desc = meta_dict.get(asin, ('Empty title', 'Empty description'))
                text_dict['title'][iid] = title
                text_dict['description'][iid] = desc
                text_dict['time'][iid][userid] = ts

    with open(os.path.join(out_dir, 'text_name_dict.json.gz'), 'wb') as tf:
        pickle.dump(text_dict, tf)

    return out_dir


def _baseline_extract_meta_fields(meta_dict, asin):
    raw_title, raw_desc = meta_dict.get(asin, [None, None])

    if raw_desc is None:
        desc = 'Empty description'
    elif len(raw_desc) == 0:
        desc = 'Empty description'
    else:
        desc = raw_desc[0]

    if raw_title is None:
        title = 'Empty title'
    elif len(raw_title) == 0:
        title = 'Empty title'
    else:
        title = raw_title

    return title, desc


def validate_preprocessed_split(fname):
    out_dir = os.path.join(os.path.dirname(__file__), '..', f'data_{fname}')
    train_path = os.path.join(out_dir, f'{fname}_train.txt')
    valid_path = os.path.join(out_dir, f'{fname}_valid.txt')
    test_path = os.path.join(out_dir, f'{fname}_test.txt')
    text_path = os.path.join(out_dir, 'text_name_dict.json.gz')

    required = [train_path, valid_path, test_path, text_path]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        print(f'[check] missing files for {fname}: {missing}')
        return False

    def _read_user_counts(path):
        counts = defaultdict(int)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                user_id, _ = line.split(' ')
                counts[int(user_id)] += 1
        return counts

    train_counts = _read_user_counts(train_path)
    valid_counts = _read_user_counts(valid_path)
    test_counts = _read_user_counts(test_path)

    train_users = set(train_counts.keys())
    valid_users = set(valid_counts.keys())
    test_users = set(test_counts.keys())

    if not train_users or not valid_users or not test_users:
        print(f'[check] empty split detected for {fname}')
        return False

    if train_users != valid_users or train_users != test_users:
        print(
            f'[check] inconsistent user sets for {fname}: '
            f'train={len(train_users)}, valid={len(valid_users)}, test={len(test_users)}'
        )
        return False

    max_user = max(train_users)
    dense_users = set(range(1, max_user + 1))
    if train_users != dense_users:
        print(
            f'[check] user ids are not dense for {fname}: '
            f'distinct={len(train_users)}, max_user={max_user}'
        )
        return False

    for user_id in train_users:
        if train_counts[user_id] < 1:
            print(f'[check] user {user_id} has empty train prefix in {fname}')
            return False
        if valid_counts[user_id] != 1 or test_counts[user_id] != 1:
            print(
                f'[check] user {user_id} violates leave-one-out counts in {fname}: '
                f"train={train_counts[user_id]}, valid={valid_counts[user_id]}, test={test_counts[user_id]}"
            )
            return False

    print(f'[check] {fname} split is consistent: users={len(train_users)}')
    return True


def preprocess_2014(fname):
    random.seed(0)
    np.random.seed(0)

    urls = _AMAZON_2014_URLS[fname]
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache_amazon2014')
    reviews_path = os.path.join(cache_dir, f'{fname}_reviews.json.gz')
    meta_path = os.path.join(cache_dir, f'{fname}_meta.json.gz')

    print(f'[2014] downloading {fname} if needed...')
    _download_if_needed(urls['reviews'], reviews_path)
    _download_if_needed(urls['meta'], meta_path)

    print('[2014] loading metadata...')
    meta_dict = {}
    for item in tqdm(_load_json_lines(meta_path)):
        asin = item.get('asin', '')
        if not asin:
            continue
        title = _normalize_text(item.get('title'), 'Empty title')
        desc = _normalize_text(item.get('description'), 'Empty description')
        meta_dict[asin] = (title, desc)

    print('[2014] loading reviews...')
    reviews = [r for r in _load_json_lines(reviews_path) if r.get('reviewerID') and r.get('asin')]
    reviews.sort(key=lambda x: (x['reviewerID'], x.get('unixReviewTime', 0)))

    user_items = defaultdict(list)
    for review in reviews:
        user_items[review['reviewerID']].append((review['asin'], review.get('unixReviewTime', 0)))

    sample_rate = {
        'Beauty_2014': 0.35,
        'Office_2014': 1.0,
    }
    users = list(user_items.keys())
    keep_users = random.sample(users, int(len(users) * sample_rate[fname]))
    print(f'[2014] sampled users: {len(keep_users)} / {len(users)}')

    itemmap = {}
    user_seqs = {}
    usernum = 0

    for uid in keep_users:
        seq = user_items[uid]
        if len(seq) < 5:
            continue
        usernum += 1
        mapped_seq = []
        for asin, ts in seq:
            if asin not in itemmap:
                itemmap[asin] = len(itemmap) + 1
            mapped_seq.append((itemmap[asin], ts))
        user_seqs[usernum] = mapped_seq

    out_dir = _write_leave_one_out(fname, user_seqs, itemmap, meta_dict)
    print(f'[2014] final users={len(user_seqs)}, items={len(itemmap)}')
    print(f'[2014] data written to: {out_dir}/')
    return out_dir


def _allmrec_filter_amazon_v2(reviews, dataset_name):
    count_u = defaultdict(int)
    count_i = defaultdict(int)
    beauty_or_toys = ('Beauty' in dataset_name) or ('Toys' in dataset_name)

    for review in reviews:
        if beauty_or_toys and review.get('overall', 5.0) < 3:
            continue
        count_u[review['reviewerID']] += 1
        count_i[review['asin']] += 1

    threshold = 4 if beauty_or_toys else 5
    filtered = [
        review for review in reviews
        if count_u[review['reviewerID']] >= threshold and count_i[review['asin']] >= threshold
    ]
    print(
        f'[2018] A-LLMRec filter: {len(reviews)} -> {len(filtered)} reviews '
        f"(threshold={threshold}, rating_gate={'overall>=3' if beauty_or_toys else 'none'})"
    )
    return filtered


def preprocess_amazon_v2(fname):
    random.seed(0)
    np.random.seed(0)

    urls = _AMAZON_2018_URLS[fname]
    raw_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data', 'amazon_v2')
    reviews_path = os.path.join(raw_dir, f'{fname}.json.gz')
    meta_path = os.path.join(raw_dir, f'meta_{fname}.json.gz')

    print(f'[2018] downloading {fname} if needed...')
    _download_if_needed(urls['reviews'], reviews_path)
    _download_if_needed(urls['meta'], meta_path)

    print('[2018] loading metadata...')
    meta_dict = {}
    for item in tqdm(_load_json_lines(meta_path)):
        asin = item.get('asin', '')
        if not asin:
            continue
        title = _normalize_text(item.get('title'), 'Empty title')
        desc = _normalize_text(item.get('description'), 'Empty description')
        meta_dict[asin] = (title, desc)

    print('[2018] loading reviews...')
    reviews = [r for r in _load_json_lines(reviews_path) if r.get('reviewerID') and r.get('asin')]
    reviews = _allmrec_filter_amazon_v2(reviews, fname)
    reviews.sort(key=lambda x: (x['reviewerID'], x.get('unixReviewTime', 0)))

    user_items = defaultdict(list)
    for review in reviews:
        user_items[review['reviewerID']].append((review['asin'], review.get('unixReviewTime', 0)))

    itemmap = {}
    user_seqs = {}
    usernum = 0

    for uid, seq in user_items.items():
        if len(seq) < 3:
            continue
        usernum += 1
        mapped_seq = []
        for asin, ts in seq:
            if asin not in itemmap:
                itemmap[asin] = len(itemmap) + 1
            mapped_seq.append((itemmap[asin], ts))
        user_seqs[usernum] = mapped_seq

    out_dir = _write_leave_one_out(fname, user_seqs, itemmap, meta_dict)
    print(f'[2018] final users={len(user_seqs)}, items={len(itemmap)}')
    print(f'[2018] data written to: {out_dir}/')
    return out_dir


def preprocess_raw_5core_local_csv(fname):
    random.seed(0)
    np.random.seed(0)

    hf_name = _HF_CONFIG_MAP.get(fname, fname)
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache_amazon2023')
    os.makedirs(cache_dir, exist_ok=True)

    split_files = {
        'train': os.path.join(cache_dir, f'{hf_name}_train.csv'),
        'valid': os.path.join(cache_dir, f'{hf_name}_valid.csv'),
        'test': os.path.join(cache_dir, f'{hf_name}_test.csv'),
    }
    meta_path = os.path.join(cache_dir, f'meta_{hf_name}.jsonl')

    for split, file_path in split_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing local csv file for split '{split}': {file_path}. "
                'Please download it first.'
            )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'Missing local meta file: {meta_path}. Please download it first.'
        )

    print('Load Meta Data (local csv mode)')
    meta_dict = {}
    for item in tqdm(_load_json_lines(meta_path)):
        meta_dict[item['parent_asin']] = [item.get('title'), item.get('description')]

    usermap = {}
    usernum = 0
    itemmap = {}
    itemnum = 0
    user_all = defaultdict(list)
    user_split = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    id2asin = {}
    time_dict = defaultdict(dict)

    for split in ['train', 'valid', 'test']:
        with open(split_files[split], 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader):
                user_id = row['user_id']
                asin = row['parent_asin']

                if user_id in usermap:
                    userid = usermap[user_id]
                else:
                    usernum += 1
                    userid = usernum
                    usermap[user_id] = userid

                if asin in itemmap:
                    itemid = itemmap[asin]
                else:
                    itemnum += 1
                    itemid = itemnum
                    itemmap[asin] = itemid

                user_all[userid].append(itemid)
                user_split[split][userid].append(itemid)
                id2asin[itemid] = asin

                ts_raw = row.get('timestamp', 0)
                try:
                    ts_val = int(float(ts_raw))
                except (TypeError, ValueError):
                    ts_val = 0
                time_dict[itemid][userid] = ts_val

    sample_rate = {
        'Movies_and_TV': 0.05,
        'Electronics': 0.05,
        'Industrial_and_Scientific': 1.0,
        'CDs_and_Vinyl': 0.33,
        'Books': 0.008,
        'Toys_and_Games': 0.12,
        'Beauty': 0.08,
        'Video_Games': 0.25,
    }
    users_all = list(user_all.keys())
    sample_ratio = sample_rate[fname]
    keep_users = random.sample(users_all, int(len(users_all) * sample_ratio))

    print('num users raw', len(users_all))
    print('num sample user', len(keep_users))

    count_u = defaultdict(int)
    count_i = defaultdict(int)
    keep_user_dict = defaultdict(int)
    keep_train_dict = defaultdict(int)

    for uid in keep_users:
        keep_user_dict[uid] = 1
        for split in ['train', 'valid', 'test']:
            for iid in user_split[split][uid]:
                count_i[iid] += 1
                count_u[uid] += 1

    out_dir = _build_output_dir(fname)
    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}
    usermap_final = {}
    itemmap_final = {}
    usernum_final = 0
    itemnum_final = 0

    for split in ['train', 'valid', 'test']:
        seen_user = defaultdict(int)
        out_path = os.path.join(out_dir, f'{fname}_{split}.txt')
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for user_id_, items in tqdm(user_split[split].items()):
                if seen_user[user_id_] == 1:
                    continue
                seen_user[user_id_] = 1

                if keep_user_dict[user_id_] != 1 or count_u[user_id_] <= 4:
                    continue

                usable_items = [iid for iid in items if count_i[iid] > 4]
                if split == 'train' and len(usable_items) <= 4:
                    continue
                if split == 'train':
                    keep_train_dict[user_id_] = 1
                elif keep_train_dict[user_id_] != 1:
                    continue

                if user_id_ in usermap_final:
                    userid = usermap_final[user_id_]
                else:
                    usernum_final += 1
                    userid = usernum_final
                    usermap_final[user_id_] = userid

                items_to_write = usable_items if split == 'train' else [iid for iid in items if count_i[iid] > 4]
                for iid_src in items_to_write:
                    if iid_src in itemmap_final:
                        itemid = itemmap_final[iid_src]
                    else:
                        itemnum_final += 1
                        itemid = itemnum_final
                        itemmap_final[iid_src] = itemid

                    title, desc = _baseline_extract_meta_fields(meta_dict, id2asin[iid_src])
                    text_dict['title'][itemid] = title
                    text_dict['description'][itemid] = desc
                    text_dict['time'][itemid][userid] = time_dict[iid_src][user_id_]
                    f_out.write(f'{userid} {itemid}\n')

    with open(os.path.join(out_dir, 'text_name_dict.json.gz'), 'wb') as tf:
        pickle.dump(text_dict, tf)

    del text_dict
    del meta_dict
    return out_dir


def preprocess_raw_5core_hf(fname):
    from datasets import load_dataset

    random.seed(0)
    np.random.seed(0)

    hf_name = _HF_CONFIG_MAP.get(fname, fname)
    print(f'Pipeline name: {fname}, HF config: {hf_name}')
    dataset = load_dataset('McAuley-Lab/Amazon-Reviews-2023', f'5core_last_out_{hf_name}', trust_remote_code=True)
    meta_dataset = load_dataset('McAuley-Lab/Amazon-Reviews-2023', f'raw_meta_{hf_name}', trust_remote_code=True)

    print('Load Meta Data')
    meta_dict = {}
    for item in tqdm(meta_dataset['full']):
        meta_dict[item['parent_asin']] = [item.get('title'), item.get('description')]
    del meta_dataset

    usermap = {}
    usernum = 0
    itemmap = {}
    itemnum = 0
    user_all = defaultdict(list)
    user_split = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    id2asin = {}
    time_dict = defaultdict(dict)

    for split in ['train', 'valid', 'test']:
        split_data = dataset[split]
        for row in tqdm(split_data):
            user_id = row['user_id']
            asin = row['parent_asin']

            if user_id in usermap:
                userid = usermap[user_id]
            else:
                usernum += 1
                userid = usernum
                usermap[user_id] = userid

            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid

            user_all[userid].append(itemid)
            user_split[split][userid].append(itemid)
            id2asin[itemid] = asin
            time_dict[itemid][userid] = row.get('timestamp', 0)

    sample_rate = {
        'Movies_and_TV': 0.05,
        'Electronics': 0.05,
        'Industrial_and_Scientific': 1.0,
        'CDs_and_Vinyl': 0.33,
        'Books': 0.008,
        'Toys_and_Games': 0.12,
        'Beauty': 0.08,
        'Video_Games': 0.25,
    }
    users_all = list(user_all.keys())
    sample_ratio = sample_rate[fname]
    keep_users = random.sample(users_all, int(len(users_all) * sample_ratio))

    print('num users raw', len(users_all))
    print('num sample user', len(keep_users))

    count_u = defaultdict(int)
    count_i = defaultdict(int)
    keep_user_dict = defaultdict(int)
    keep_train_dict = defaultdict(int)

    for uid in keep_users:
        keep_user_dict[uid] = 1
        for split in ['train', 'valid', 'test']:
            for iid in user_split[split][uid]:
                count_i[iid] += 1
                count_u[uid] += 1

    out_dir = _build_output_dir(fname)
    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}
    usermap_final = {}
    itemmap_final = {}
    usernum_final = 0

    for split in ['train', 'valid', 'test']:
        split_data = dataset[split]
        seen_user = defaultdict(int)
        out_path = os.path.join(out_dir, f'{fname}_{split}.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            for row in tqdm(split_data):
                user_id = row['user_id']
                user_id_ = usermap[user_id]

                if seen_user[user_id_] == 1:
                    continue
                seen_user[user_id_] = 1

                if keep_user_dict[user_id_] != 1 or count_u[user_id_] <= 4:
                    continue

                usable_items = [iid for iid in user_split[split][user_id_] if count_i[iid] > 4]
                if split == 'train' and len(usable_items) <= 4:
                    continue
                if split == 'train':
                    keep_train_dict[user_id_] = 1
                elif keep_train_dict[user_id_] != 1:
                    continue

                if user_id_ in usermap_final:
                    userid = usermap_final[user_id_]
                else:
                    usernum_final += 1
                    userid = usernum_final
                    usermap_final[user_id_] = userid

                items_to_write = usable_items if split == 'train' else [iid for iid in user_split[split][user_id_] if count_i[iid] > 4]
                for iid_src in items_to_write:
                    if iid_src in itemmap_final:
                        itemid = itemmap_final[iid_src]
                    else:
                        itemmap_final[iid_src] = len(itemmap_final) + 1
                        itemid = itemmap_final[iid_src]

                    title, desc = _baseline_extract_meta_fields(meta_dict, id2asin[iid_src])
                    text_dict['title'][itemid] = title
                    text_dict['description'][itemid] = desc
                    text_dict['time'][itemid][userid] = time_dict[iid_src][user_id_]
                    f.write(f'{userid} {itemid}\n')

    with open(os.path.join(out_dir, 'text_name_dict.json.gz'), 'wb') as tf:
        pickle.dump(text_dict, tf)

    del text_dict
    del meta_dict
    del dataset
    return out_dir


def preprocess_raw_5core(fname):
    """Use local aria2-downloaded CSV/JSONL when enabled, otherwise HuggingFace datasets."""
    use_local = os.environ.get('USE_LOCAL_HF_2023_FILES', '0') == '1'
    if use_local:
        print('[2023] USE_LOCAL_HF_2023_FILES=1 -> using local CSV/JSONL cache (aria2 mode)')
        return preprocess_raw_5core_local_csv(fname)
    return preprocess_raw_5core_hf(fname)
