import argparse
import os
from collections import Counter

from accelerate import Accelerator

from genrec.utils import get_config, get_dataset, get_tokenizer, init_device, init_seed


def parse_unknown_args(unparsed):
    config = {}
    for item in unparsed:
        if not item.startswith('--') or '=' not in item:
            raise ValueError(f'Invalid argument {item}; expected --key=value')
        key, value = item[2:].split('=', 1)
        try:
            value = eval(value)
        except Exception:
            pass
        config[key] = value
    return config


def build_config(model, dataset, overrides):
    config = get_config(model, dataset, None, overrides)
    config['device'], config['use_ddp'] = init_device()
    config['accelerator'] = Accelerator()
    init_seed(config['rand_seed'], config['reproducibility'])
    return config


def load_dataset_and_tokenizer(model, dataset, overrides):
    config = build_config(model, dataset, overrides)
    raw_dataset = get_dataset(dataset)(config)
    split_datasets = raw_dataset.split()
    tokenizer = get_tokenizer(model)(config, raw_dataset)
    return config, raw_dataset, split_datasets, tokenizer


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_examples(split_dataset, split):
    sample_id = 0
    for example in split_dataset:
        user = example['user']
        seq = example['item_seq']
        if split == 'train':
            for i in range(len(seq) - 1):
                yield {
                    'sample_id': sample_id,
                    'user': user,
                    'history_items': seq[:i + 1],
                    'target_item': seq[i + 1],
                }
                sample_id += 1
        else:
            yield {
                'sample_id': sample_id,
                'user': user,
                'history_items': seq[:-1],
                'target_item': seq[-1],
            }
            sample_id += 1


def item_popularity(split_dataset):
    counter = Counter()
    for example in split_dataset:
        counter.update(example['item_seq'])
    return counter
