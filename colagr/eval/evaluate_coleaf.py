import argparse
import heapq
import json
import math
import random
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from colagr.common import load_dataset_and_tokenizer


def parse_int_list(value):
    return [int(item) for item in value.split(',') if item.strip()]


def parse_float_list(value):
    return [float(item) for item in value.split(',') if item.strip()]


def rank_metrics(items, target, cutoffs=(5, 10, 50)):
    try:
        rank = items.index(target) + 1
    except ValueError:
        rank = None
    result = {}
    for cutoff in cutoffs:
        hit = float(rank is not None and rank <= cutoff)
        result[f'recall@{cutoff}'] = hit
        result[f'ndcg@{cutoff}'] = hit / math.log2(rank + 1) if hit else 0.0
    return result


def add_metrics(total, values):
    for key, value in values.items():
        total[key] += value


def normalize_scores(scores):
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {}
    return {item: score / maximum for item, score in scores.items()}


def residual_leaf_scores(
    memory_scores,
    item2tokens,
    parent_to_items,
    candidate_items=None,
):
    """Return leaf utility after removing its parent-subtree value.

    ``memory_scores`` is sparse because the transition memory keeps only its
    top-k neighbors. Missing catalog items therefore have utility zero, but
    they still participate in the parent mean. Scores are returned for the
    requested candidates so a base item without memory support can receive a
    negative residual instead of being silently assigned zero.
    """
    parent_means = {
        parent: sum(float(memory_scores.get(item, 0.0)) for item in items) / len(items)
        for parent, items in parent_to_items.items()
        if items
    }
    items = (
        list(candidate_items)
        if candidate_items is not None
        else list(item2tokens)
    )
    residual = {}
    for item in items:
        item = str(item)
        score = float(memory_scores.get(item, 0.0))
        tokens = item2tokens.get(item)
        if tokens is None:
            residual[item] = score
            continue
        parent = tuple(int(token) for token in tokens[:-1])
        residual[item] = score - parent_means.get(parent, 0.0)
    return residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', required=True)
    parser.add_argument('--vq_method', default='rqkmeans')
    parser.add_argument('--base_candidates', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--neighbors_per_item', type=int, default=200)
    parser.add_argument('--history_lengths', default='3,5,10')
    parser.add_argument('--memory_ks', default='10,20,50')
    parser.add_argument('--alphas', default='0.02,0.05,0.1,0.2,0.4,0.7,1.0')
    parser.add_argument('--rrf_constant', type=float, default=10.0)
    parser.add_argument(
        '--allow_rank_fallback',
        action='store_true',
        help='Use reciprocal rank only for legacy audit artifacts without beam scores.',
    )
    parser.add_argument(
        '--memory_source',
        choices=('transition', 'last_item', 'cooccurrence', 'popularity'),
        default='transition',
    )
    parser.add_argument(
        '--fusion_mode',
        choices=('score', 'rrf', 'memory_only'),
        default='score',
    )
    parser.add_argument(
        '--leaf_score_mode',
        choices=('absolute', 'residual'),
        default='absolute',
        help='Use absolute CoLeaf utility or within-subtree residual utility.',
    )
    parser.add_argument(
        '--candidate_mode',
        choices=('union', 'base_only'),
        default='union',
    )
    parser.add_argument('--shuffle_memory_seed', type=int, default=None)
    parser.add_argument('--record_output', default=None)
    parser.add_argument(
        '--shuffle_leaf_seed',
        type=int,
        default=None,
        help='Permute leaf scores only within each fixed candidate set.',
    )
    parser.add_argument(
        '--detailed_record_output',
        default=None,
        help='Save fixed-candidate IDs, scores, and ranks for every sample.',
    )
    args = parser.parse_args()

    history_lengths = parse_int_list(args.history_lengths)
    memory_ks = parse_int_list(args.memory_ks)
    alphas = parse_float_list(args.alphas)
    max_history = max(history_lengths)
    max_memory_k = max(memory_ks)

    overrides = {
        'category': args.category,
        'vq_method': args.vq_method,
        'num_proc': 1,
        'use_latte_latent': False,
    }
    _, _, splits, tokenizer = load_dataset_and_tokenizer(
        'CoLaGR',
        args.dataset,
        overrides,
    )
    item2tokens = {
        str(item): tuple(int(token) for token in tokens)
        for item, tokens in tokenizer.item2tokens.items()
    }
    parent_to_items = defaultdict(list)
    for item, tokens in item2tokens.items():
        parent_to_items[tokens[:-1]].append(item)

    transitions = defaultdict(Counter)
    popularity = Counter()
    for example in tqdm(splits['train'], desc='Build collaborative memory'):
        sequence = [str(item) for item in example['item_seq']]
        popularity.update(sequence)
        for target_pos in range(1, len(sequence)):
            target = sequence[target_pos]
            start = (
                target_pos - 1
                if args.memory_source == 'last_item'
                else max(0, target_pos - args.window)
            )
            for source_pos in range(start, target_pos):
                distance = target_pos - source_pos
                transitions[sequence[source_pos]][target] += 1.0 / distance
                if args.memory_source == 'cooccurrence':
                    transitions[target][sequence[source_pos]] += 1.0 / distance

    memory = {
        source: heapq.nlargest(
            args.neighbors_per_item,
            counter.items(),
            key=lambda pair: pair[1],
        )
        for source, counter in transitions.items()
    }
    popular_items = [item for item, _ in popularity.most_common(max_memory_k * 4)]
    popular_scores = dict(popularity.most_common(max_memory_k))
    base_records = torch.load(args.base_candidates, map_location='cpu')
    base_by_sample = {}
    base_generation_scores = {}
    for record in base_records:
        sample_id = int(record['sample_id'])
        candidates = [str(item) for item in record['candidates']]
        scores = record.get('generation_scores')
        if scores is None:
            if not args.allow_rank_fallback:
                raise ValueError(
                    'Candidate artifact lacks cumulative generation_scores. '
                    'Re-export with export_beam_candidates.py --with_scores.'
                )
            scores = [1.0 / (args.rrf_constant + rank) for rank in range(1, len(candidates) + 1)]
        if len(candidates) != len(scores):
            raise ValueError(
                f'Candidate/score length mismatch for sample {sample_id}: '
                f'{len(candidates)} candidates versus {len(scores)} scores.'
            )
        if len(set(candidates)) != len(candidates):
            raise ValueError(f'Duplicate candidates for sample {sample_id}.')
        base_by_sample[sample_id] = candidates
        base_generation_scores[sample_id] = {
            item: float(score) for item, score in zip(candidates, scores)
        }
    memory_example_indices = list(range(len(splits[args.split])))
    if args.shuffle_memory_seed is not None:
        random.Random(args.shuffle_memory_seed).shuffle(memory_example_indices)

    configs = [
        (history_length, memory_k, alpha)
        for history_length in history_lengths
        for memory_k in memory_ks
        for alpha in alphas
    ]
    totals = {config: defaultdict(float) for config in configs}
    delta_stats = {
        config: {
            metric: {'sum': 0.0, 'sum_sq': 0.0, 'wins': 0, 'losses': 0}
            for metric in (
                'ndcg@5', 'ndcg@10', 'ndcg@50',
                'recall@5', 'recall@10', 'recall@50',
            )
        }
        for config in configs
    }
    movement_stats = {
        config: {
            'target_in_beam': 0,
            'target_top10_before': 0,
            'target_top10_after': 0,
            'promoted': 0,
            'demoted': 0,
            'unchanged': 0,
            'original_rank_sum': 0.0,
            'calibrated_rank_sum': 0.0,
        }
        for config in configs
    }
    base_total = defaultdict(float)
    union_oracle_total = defaultdict(float)
    user_records = []
    samples = 0

    for sample_id, example in enumerate(tqdm(splits[args.split], desc='Sweep fusion')):
        sequence = [str(item) for item in example['item_seq']]
        history, target = sequence[:-1], sequence[-1]
        history_set = set(history)
        memory_example = splits[args.split][memory_example_indices[sample_id]]
        memory_sequence = [str(item) for item in memory_example['item_seq']]
        memory_history = memory_sequence[:-1]
        base_items = base_by_sample.get(sample_id, [])
        base_item_scores = base_generation_scores.get(sample_id)
        if base_item_scores is None:
            raise ValueError(f'Missing candidate record for sample {sample_id}.')
        base_metrics = rank_metrics(base_items, target)
        add_metrics(base_total, base_metrics)

        memory_scores_by_history = {}
        for history_length in history_lengths:
            if args.memory_source == 'popularity':
                scores = Counter({
                    candidate: weight
                    for candidate, weight in popular_scores.items()
                    if candidate not in history_set
                })
            else:
                scores = Counter()
                query_history = (
                    memory_history[-1:]
                    if args.memory_source == 'last_item'
                    else memory_history[-history_length:]
                )
                for recency, source in enumerate(reversed(query_history)):
                    query_weight = 1.0 / (1.0 + recency)
                    for candidate, weight in memory.get(source, ()):
                        if candidate not in history_set:
                            scores[candidate] += query_weight * weight
            memory_scores_by_history[history_length] = scores

        largest_memory = [
            item
            for item, _ in memory_scores_by_history[max_history].most_common(max_memory_k)
        ]
        union = base_items + [item for item in largest_memory if item not in set(base_items)]
        oracle_items = [target] + [item for item in union if item != target] if target in union else union
        add_metrics(union_oracle_total, rank_metrics(oracle_items, target))

        for history_length, memory_k, alpha in configs:
            raw_memory_scores = memory_scores_by_history[history_length]
            memory_items = [
                item for item, _ in raw_memory_scores.most_common(memory_k)
            ]
            if len(memory_items) < memory_k:
                selected = set(memory_items)
                memory_items.extend(
                    item for item in popular_items
                    if item not in history_set and item not in selected
                )
                memory_items = memory_items[:memory_k]
            memory_scores = normalize_scores({
                item: float(raw_memory_scores.get(item, 0.0))
                for item in memory_items
            })
            candidates = base_items + [
                item for item in memory_items if item not in base_item_scores
            ] if args.candidate_mode == 'union' else base_items
            if args.leaf_score_mode == 'residual':
                # Decompose utility at the same candidate set used for fusion;
                # the parent mean still sees every catalog leaf.
                memory_scores = residual_leaf_scores(
                    memory_scores,
                    item2tokens,
                    parent_to_items,
                    candidate_items=candidates,
                )
            if args.shuffle_leaf_seed is not None:
                values = [float(memory_scores.get(item, 0.0)) for item in candidates]
                random.Random(args.shuffle_leaf_seed + sample_id).shuffle(values)
                memory_scores = dict(zip(candidates, values))
            original_order = {
                item: rank for rank, item in enumerate(candidates)
            }
            memory_ranks = {
                item: rank
                for rank, item in enumerate(memory_items, start=1)
            }

            def fusion_score(item):
                if args.fusion_mode == 'memory_only':
                    return memory_scores.get(item, 0.0)
                if args.fusion_mode == 'rrf':
                    memory_rank = memory_ranks.get(item)
                    memory_term = (
                        1.0 / (args.rrf_constant + memory_rank)
                        if memory_rank is not None else 0.0
                    )
                    return base_item_scores.get(item, float('-inf')) + alpha * memory_term
                return (
                    base_item_scores.get(item, float('-inf'))
                    + alpha * memory_scores.get(item, 0.0)
                )

            ranked = sorted(
                candidates,
                key=lambda item: (
                    fusion_score(item),
                    -original_order[item],
                ),
                reverse=True,
            )
            if set(ranked) != set(candidates) or len(ranked) != len(candidates):
                raise RuntimeError('CoLeaf changed the fixed candidate set.')
            config = (history_length, memory_k, alpha)
            fused_metrics = rank_metrics(ranked, target)
            add_metrics(totals[config], fused_metrics)
            try:
                original_rank = base_items.index(target) + 1
            except ValueError:
                original_rank = None
            try:
                calibrated_rank = ranked.index(target) + 1
            except ValueError:
                calibrated_rank = None
            movement = movement_stats[config]
            if original_rank is not None:
                movement['target_in_beam'] += 1
                movement['target_top10_before'] += int(original_rank <= 10)
                movement['target_top10_after'] += int(
                    calibrated_rank is not None and calibrated_rank <= 10
                )
                movement['original_rank_sum'] += original_rank
                movement['calibrated_rank_sum'] += calibrated_rank
                movement['promoted'] += int(calibrated_rank < original_rank)
                movement['demoted'] += int(calibrated_rank > original_rank)
                movement['unchanged'] += int(calibrated_rank == original_rank)

            if args.detailed_record_output is not None and len(configs) == 1:
                rank_after = {item: rank for rank, item in enumerate(ranked, start=1)}
                user_records.append({
                    'sample_id': sample_id,
                    'target_item': target,
                    'candidates_before': list(candidates),
                    'candidates_after': list(ranked),
                    's_gen': [float(base_item_scores.get(item, float('-inf'))) for item in candidates],
                    's_leaf': [float(memory_scores.get(item, 0.0)) for item in candidates],
                    's_final': [float(fusion_score(item)) for item in candidates],
                    'rank_before': list(range(1, len(candidates) + 1)),
                    'rank_after': [rank_after[item] for item in candidates],
                    'ground_truth_in_beam': original_rank is not None,
                    'ground_truth_rank_before': original_rank,
                    'ground_truth_rank_after': calibrated_rank,
                })
            elif args.record_output is not None and len(configs) == 1:
                positive_scores = [
                    float(raw_memory_scores.get(item, 0.0))
                    for item in base_items
                    if raw_memory_scores.get(item, 0.0) > 0
                ]
                score_sum = sum(positive_scores)
                entropy = 0.0
                if score_sum > 0:
                    for score in positive_scores:
                        probability = score / score_sum
                        entropy -= probability * math.log(probability)
                base_item_set = set(base_items)
                supporting_history = sum(
                    any(candidate in base_item_set for candidate, _ in memory.get(source, ()))
                    for source in memory_history[-history_length:]
                )
                user_records.append({
                    'sample_id': sample_id,
                    'target_item': target,
                    'target_in_beam50': original_rank is not None,
                    'original_target_rank': original_rank,
                    'calibrated_target_rank': calibrated_rank,
                    'memory_score': float(raw_memory_scores.get(target, 0.0)),
                    'memory_rank': memory_ranks.get(target),
                    'nonzero_support_count': len(positive_scores),
                    'supporting_history_items': supporting_history,
                    'max_raw_support': max(positive_scores, default=0.0),
                    'memory_entropy': entropy,
                })
            for metric, fused_value in fused_metrics.items():
                delta = fused_value - base_metrics[metric]
                stats = delta_stats[config][metric]
                stats['sum'] += delta
                stats['sum_sq'] += delta * delta
                stats['wins'] += int(delta > 0)
                stats['losses'] += int(delta < 0)
        samples += 1

    def means(total):
        return {
            key: value / max(samples, 1)
            for key, value in sorted(total.items())
        }

    ranked_results = []
    for config, total in totals.items():
        metrics = means(total)
        paired = {}
        for metric, stats in delta_stats[config].items():
            mean_delta = stats['sum'] / max(samples, 1)
            variance = max(
                stats['sum_sq'] / max(samples, 1) - mean_delta * mean_delta,
                0.0,
            )
            standard_error = math.sqrt(variance / max(samples, 1))
            paired[metric] = {
                'mean_delta': mean_delta,
                'standard_error': standard_error,
                'z_score': mean_delta / standard_error if standard_error > 0 else 0.0,
                'wins': stats['wins'],
                'losses': stats['losses'],
            }
        ranked_results.append({
            'history_length': config[0],
            'memory_k': config[1],
            'alpha': config[2],
            'paired': paired,
            'movement': {
                **movement_stats[config],
                'beam50_to_top10_before': (
                    movement_stats[config]['target_top10_before']
                    / max(movement_stats[config]['target_in_beam'], 1)
                ),
                'beam50_to_top10_after': (
                    movement_stats[config]['target_top10_after']
                    / max(movement_stats[config]['target_in_beam'], 1)
                ),
                'mean_original_rank': (
                    movement_stats[config]['original_rank_sum']
                    / max(movement_stats[config]['target_in_beam'], 1)
                ),
                'mean_calibrated_rank': (
                    movement_stats[config]['calibrated_rank_sum']
                    / max(movement_stats[config]['target_in_beam'], 1)
                ),
            },
            **metrics,
        })
    ranked_results.sort(
        key=lambda result: (result['ndcg@10'], result['ndcg@5']),
        reverse=True,
    )
    output = {
        'samples': samples,
        'candidate_mode': args.candidate_mode,
        'memory_source': args.memory_source,
        'fusion_mode': args.fusion_mode,
        'leaf_score_mode': args.leaf_score_mode,
        'shuffle_memory_seed': args.shuffle_memory_seed,
        'shuffle_leaf_seed': args.shuffle_leaf_seed,
        'base': means(base_total),
        'union_oracle': means(union_oracle_total),
        'top_configs': ranked_results[:20],
    }
    record_output = args.detailed_record_output or args.record_output
    if record_output is not None:
        if len(configs) != 1:
            raise ValueError(
                '--record_output requires exactly one history length, memory_k, and alpha.'
            )
        torch.save(user_records, record_output)
    print('COLAGR_RESULT=' + json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
