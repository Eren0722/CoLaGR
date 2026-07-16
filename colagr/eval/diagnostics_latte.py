import argparse
import json
import os

import torch


def load_copref(path, limit):
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict) and data.get('format') == 'colagr_copref_tensor_v1':
        if limit is not None:
            data = {
                **data,
                'sample_id': data['sample_id'][:limit],
                'target_item': data['target_item'][:limit],
                'target_sid': data['target_sid'][:limit],
                'copref': [level[:limit] for level in data['copref']],
                'prefix_count': data['prefix_count'][:limit],
                'entropy': data['entropy'][:limit],
                'target_code_rank': data['target_code_rank'][:limit],
            }
        return data
    if limit is not None:
        data = data[:limit]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--copref_path', required=True)
    parser.add_argument('--sid_artifacts_dir', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    records = load_copref(args.copref_path, args.limit)
    level_token_ids = torch.load(os.path.join(args.sid_artifacts_dir, 'level_token_ids.pt'), map_location='cpu')
    meta = json.load(open(os.path.join(args.sid_artifacts_dir, 'tokenizer_meta.json')))

    if isinstance(records, dict) and records.get('format') == 'colagr_copref_tensor_v1':
        sums = torch.cat([level.sum(dim=-1) for level in records['copref']])
        negative_count = sum(int((level < 0).sum().item()) for level in records['copref'])
        nan_count = sum(int(torch.isnan(level).sum().item()) for level in records['copref'])
        prefix_counts = records['prefix_count'].reshape(-1).float()
        entropies = records['entropy'].reshape(-1).float()
        ranks = records['target_code_rank'].reshape(-1).float()
        random_ranks = torch.cat([
            torch.full((records['sample_id'].numel(),), (len(tokens) + 1) / 2, dtype=torch.float)
            for tokens in level_token_ids
        ])
        hit5 = int((ranks <= 5).sum().item())
        hit10 = int((ranks <= 10).sum().item())
        total = int(ranks.numel())
        num_records = int(records['sample_id'].numel())
    else:
        sums, negative_count, nan_count = [], 0, 0
        prefix_counts, entropies, ranks, random_ranks = [], [], [], []
        hit5, hit10, total = 0, 0, 0
        for record in records:
            prefix_counts.extend(record['prefix_count'])
            entropies.extend(record['entropy'])
            ranks.extend(record['target_code_rank'])
            for level, dist in enumerate(record['copref']):
                dist = torch.as_tensor(dist)
                sums.append(float(dist.sum().item()))
                negative_count += int((dist < 0).sum().item())
                nan_count += int(torch.isnan(dist).sum().item())
                random_ranks.append((len(level_token_ids[level]) + 1) / 2)
                rank = int(record['target_code_rank'][level])
                hit5 += int(rank <= 5)
                hit10 += int(rank <= 10)
                total += 1
        sums = torch.tensor(sums, dtype=torch.float)
        prefix_counts = torch.tensor(prefix_counts, dtype=torch.float)
        entropies = torch.tensor(entropies, dtype=torch.float)
        ranks = torch.tensor(ranks, dtype=torch.float)
        random_ranks = torch.tensor(random_ranks, dtype=torch.float)
        num_records = len(records)

    gate_values = None
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        gates = ckpt.get('fuse_gates') if isinstance(ckpt, dict) else None
        if gates is not None:
            gate_values = torch.sigmoid(gates).tolist()

    print(f'num_records: {num_records}')
    print(f'copref_sum_min: {float(sums.min()):.6f}')
    print(f'copref_sum_max: {float(sums.max()):.6f}')
    print(f'copref_sum_mean: {float(sums.mean()) if sums.numel() else 0.0:.6f}')
    print(f'nan_count: {nan_count}')
    print(f'negative_count: {negative_count}')
    print(f'prefix_count_mean: {float(prefix_counts.mean()) if prefix_counts.numel() else 0.0:.6f}')
    print(f'entropy_mean: {float(entropies.mean()) if entropies.numel() else 0.0:.6f}')
    print(f'target_code_mean_rank: {float(ranks.mean()) if ranks.numel() else 0.0:.6f}')
    print(f'target_code_median_rank: {float(torch.median(ranks)) if ranks.numel() else 0.0:.6f}')
    print(f'random_rank_baseline: {float(random_ranks.mean()) if random_ranks.numel() else 0.0:.6f}')
    print(f'target_code_hit@5: {hit5 / max(total, 1):.6f}')
    print(f'target_code_hit@10: {hit10 / max(total, 1):.6f}')
    print(f'gate_values: {gate_values}')
    print('invalid_sid_rate: requires generated predictions; not computed from CoPref cache')
    print('prefix_survival: requires generated predictions; not computed from CoPref cache')
    print('copref_kl_over_epochs: requires training logs; not computed from CoPref cache')
    if (float(ranks.mean()) if ranks.numel() else 0.0) >= (float(random_ranks.mean()) if random_ranks.numel() else 0.0):
        print('WARNING: target-code rank is not better than random; check teacher/item/SID alignment.')
    else:
        print('OK: target-code rank is better than random.')
    print(f"meta_coreason_token_ids: {meta.get('coreason_token_ids')}")


if __name__ == '__main__':
    main()
