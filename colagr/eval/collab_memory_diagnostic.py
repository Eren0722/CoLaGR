import argparse
import heapq
import json
import math
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from colagr.common import load_dataset_and_tokenizer


def metrics_for_rank(rank, cutoffs=(5, 10, 50)):
    result = {}
    for cutoff in cutoffs:
        hit = float(rank is not None and rank <= cutoff)
        result[f'recall@{cutoff}'] = hit
        if cutoff <= 10:
            result[f'ndcg@{cutoff}'] = hit / math.log2(rank + 1) if hit else 0.0
    return result


def add_metrics(total, values):
    for key, value in values.items():
        total[key] += value


def rank_of(items, target):
    try:
        return items.index(target) + 1
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', required=True)
    parser.add_argument('--vq_method', default='rqkmeans')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--neighbors_per_item', type=int, default=200)
    parser.add_argument('--history_length', type=int, default=10)
    parser.add_argument('--candidate_k', type=int, default=50)
    parser.add_argument('--base_candidates', default=None)
    args = parser.parse_args()

    overrides = {
        'category': args.category,
        'vq_method': args.vq_method,
        'num_proc': 1,
        'use_latte_latent': False,
    }
    _, _, splits, _ = load_dataset_and_tokenizer('CoLaGR', args.dataset, overrides)

    transitions = defaultdict(Counter)
    popularity = Counter()
    for example in tqdm(splits['train'], desc='Build collaborative memory'):
        sequence = [str(item) for item in example['item_seq']]
        popularity.update(sequence)
        for target_pos in range(1, len(sequence)):
            target = sequence[target_pos]
            start = max(0, target_pos - args.window)
            for source_pos in range(start, target_pos):
                distance = target_pos - source_pos
                transitions[sequence[source_pos]][target] += 1.0 / distance

    memory = {
        source: heapq.nlargest(
            args.neighbors_per_item,
            counter.items(),
            key=lambda pair: pair[1],
        )
        for source, counter in transitions.items()
    }
    popular_items = [item for item, _ in popularity.most_common(args.candidate_k * 4)]

    base_by_sample = {}
    if args.base_candidates is not None:
        records = torch.load(args.base_candidates, map_location='cpu')
        base_by_sample = {
            int(record['sample_id']): [str(item) for item in record['candidates']]
            for record in records
        }

    memory_total = defaultdict(float)
    base_total = defaultdict(float)
    union_total = defaultdict(float)
    memory_only_hits = 0
    base_only_hits = 0
    both_hits = 0
    union_hits = 0
    samples = 0

    for sample_id, example in enumerate(tqdm(splits['val'], desc='Evaluate collaborative memory')):
        sequence = [str(item) for item in example['item_seq']]
        history, target = sequence[:-1], sequence[-1]
        history_set = set(history)
        scores = Counter()
        recent = history[-args.history_length:]
        for recency, source in enumerate(reversed(recent)):
            query_weight = 1.0 / (1.0 + recency)
            for candidate, weight in memory.get(source, ()):
                if candidate not in history_set:
                    scores[candidate] += query_weight * weight
        memory_items = [
            item for item, _ in scores.most_common(args.candidate_k)
        ]
        if len(memory_items) < args.candidate_k:
            selected = set(memory_items)
            memory_items.extend(
                item for item in popular_items
                if item not in history_set and item not in selected
            )
            memory_items = memory_items[:args.candidate_k]

        base_items = base_by_sample.get(sample_id, [])[:args.candidate_k]
        union_items = base_items + [
            item for item in memory_items if item not in set(base_items)
        ]
        cutoffs = tuple(dict.fromkeys((5, 10, args.candidate_k)))
        add_metrics(memory_total, metrics_for_rank(rank_of(memory_items, target), cutoffs))
        if base_by_sample:
            add_metrics(base_total, metrics_for_rank(rank_of(base_items, target), cutoffs))
            add_metrics(union_total, metrics_for_rank(rank_of(union_items, target), cutoffs))
            memory_hit = target in memory_items
            base_hit = target in base_items
            memory_only_hits += int(memory_hit and not base_hit)
            base_only_hits += int(base_hit and not memory_hit)
            both_hits += int(memory_hit and base_hit)
            union_hits += int(memory_hit or base_hit)
        samples += 1

    def means(total):
        return {key: value / max(samples, 1) for key, value in sorted(total.items())}

    result = {
        'samples': samples,
        'memory': means(memory_total),
        'base': means(base_total) if base_by_sample else None,
        'union': means(union_total) if base_by_sample else None,
        f'memory_only_hit_rate@{args.candidate_k}': memory_only_hits / max(samples, 1),
        f'base_only_hit_rate@{args.candidate_k}': base_only_hits / max(samples, 1),
        f'both_hit_rate@{args.candidate_k}': both_hits / max(samples, 1),
        'beam_memory_union_coverage': union_hits / max(samples, 1),
    }
    print('COLLAB_MEMORY_DIAGNOSTIC=' + json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
