import argparse
import json
import os

import torch
from tqdm import tqdm

from colagr.common import ensure_dir


def local_maps(level_token_ids):
    return [{int(token): idx for idx, token in enumerate(tokens.tolist())} for tokens in level_token_ids]


def rank_of_target(distribution, target_local):
    order = torch.argsort(distribution, descending=True)
    return int((order == target_local).nonzero(as_tuple=False)[0, 0].item()) + 1


def prepare_sid_cache(item2sid, token_to_local):
    cache = {}
    for item, sid in item2sid.items():
        sid = [int(token) for token in sid]
        cache[str(item)] = {
            'sid': sid,
            'locals': [token_to_local[level][int(token)] for level, token in enumerate(sid)],
            'prefixes': [tuple(sid[:level]) for level in range(len(sid))],
        }
    return cache


def prepare_item_tensor_maps(item2sid, token_to_local):
    item_to_idx = {str(item): idx for idx, item in enumerate(item2sid)}
    num_items = len(item_to_idx)
    num_levels = len(token_to_local)
    sid_tokens = torch.empty(num_items, num_levels, dtype=torch.long)
    sid_locals = torch.empty(num_items, num_levels, dtype=torch.long)
    for item, idx in item_to_idx.items():
        sid = [int(token) for token in item2sid[item]]
        sid_tokens[idx] = torch.tensor(sid, dtype=torch.long)
        sid_locals[idx] = torch.tensor(
            [token_to_local[level][sid[level]] for level in range(num_levels)],
            dtype=torch.long,
        )
    return item_to_idx, sid_tokens, sid_locals


def build_record(record, item2sid, sid_cache, level_token_ids, temp, tau):
    target_item = str(record['target_item'])
    target_sid = item2sid[target_item]
    target_cache = sid_cache[target_item]
    top_cached = [sid_cache.get(str(item)) for item in record['top_items']]
    valid_pairs = [
        (idx, cached) for idx, cached in enumerate(top_cached)
        if cached is not None
    ]
    top_scores = torch.as_tensor(record['top_scores'], dtype=torch.float)
    p_teacher = torch.softmax(top_scores / temp, dim=0)

    coprefs, prefix_count, entropies, target_ranks = [], [], [], []
    for level, tokens in enumerate(level_token_ids):
        target_prefix = target_sid[:level]
        target_prefix_key = tuple(int(token) for token in target_prefix)

        if valid_pairs:
            valid_indices = torch.tensor([idx for idx, _ in valid_pairs], dtype=torch.long)
            local_indices = torch.tensor([cached['locals'][level] for _, cached in valid_pairs], dtype=torch.long)
            valid_probs = p_teacher.index_select(0, valid_indices)
            q_global = torch.bincount(
                local_indices,
                weights=valid_probs,
                minlength=len(tokens),
            ).float()
            prefix_mask = torch.tensor(
                [cached['prefixes'][level] == target_prefix_key for _, cached in valid_pairs],
                dtype=torch.bool,
            )
            matched = int(prefix_mask.sum().item())
            if matched > 0:
                q_prefix = torch.bincount(
                    local_indices[prefix_mask],
                    weights=valid_probs[prefix_mask],
                    minlength=len(tokens),
                ).float()
            else:
                q_prefix = torch.zeros(len(tokens), dtype=torch.float)
        else:
            q_global = torch.zeros(len(tokens), dtype=torch.float)
            q_prefix = torch.zeros(len(tokens), dtype=torch.float)
            matched = 0

        global_sum = q_global.sum()
        if global_sum <= 0:
            q_global.fill_(1.0 / len(q_global))
        else:
            q_global /= global_sum
        prefix_sum = q_prefix.sum()
        if prefix_sum > 0:
            q_prefix /= prefix_sum

        alpha = matched / (matched + tau)
        q = alpha * q_prefix + (1.0 - alpha) * q_global
        q = q.clamp_min(0)
        q /= q.sum().clamp_min(1e-12)

        target_local = target_cache['locals'][level]
        coprefs.append(q)
        prefix_count.append(int(matched))
        entropies.append(float(-(q * q.clamp_min(1e-12).log()).sum().item()))
        target_ranks.append(rank_of_target(q, target_local))

    return {
        'sample_id': int(record['sample_id']),
        'target_item': target_item,
        'target_sid': [int(token) for token in target_sid],
        'copref': coprefs,
        'prefix_count': prefix_count,
        'entropy': entropies,
        'target_code_rank': target_ranks,
    }


def rank_of_target_batch(distributions, target_locals):
    order = torch.argsort(distributions, dim=-1, descending=True)
    return (order == target_locals.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1


def build_records_batch(records, item_to_idx, sid_tokens, sid_locals, level_token_ids, temp, tau, device):
    batch_size = len(records)
    num_levels = sid_tokens.shape[1]
    top_m = max(len(record['top_items']) for record in records)
    invalid_item_idx = len(item_to_idx)

    top_item_indices = torch.full((batch_size, top_m), invalid_item_idx, dtype=torch.long)
    score_matrix = torch.full((batch_size, top_m), float('-inf'), dtype=torch.float)
    target_indices = torch.empty(batch_size, dtype=torch.long)

    for row, record in enumerate(records):
        target_indices[row] = item_to_idx[str(record['target_item'])]
        cur_scores = torch.as_tensor(record['top_scores'], dtype=torch.float)
        for col, item in enumerate(record['top_items']):
            top_item_indices[row, col] = item_to_idx.get(str(item), invalid_item_idx)
        score_matrix[row, :len(cur_scores)] = cur_scores

    valid_mask = top_item_indices != invalid_item_idx
    safe_top_indices = top_item_indices.clamp_max(invalid_item_idx - 1)
    top_item_indices = safe_top_indices.to(device)
    target_indices = target_indices.to(device)
    valid_mask = valid_mask.to(device)
    score_matrix = score_matrix.to(device)

    p_teacher = torch.softmax(score_matrix / temp, dim=-1) * valid_mask.float()
    p_teacher = p_teacher / p_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    target_sid_tokens = sid_tokens.index_select(0, target_indices)
    target_sid_locals = sid_locals.index_select(0, target_indices)
    top_sid_tokens = sid_tokens.index_select(0, top_item_indices.reshape(-1)).view(batch_size, top_m, num_levels)
    top_sid_locals = sid_locals.index_select(0, top_item_indices.reshape(-1)).view(batch_size, top_m, num_levels)

    per_level_coprefs, per_level_prefix_counts, per_level_entropies, per_level_ranks = [], [], [], []
    rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, top_m)
    for level, tokens in enumerate(level_token_ids):
        width = len(tokens)
        q_global = torch.zeros(batch_size, width, dtype=torch.float, device=device)
        q_global.index_put_(
            (rows.reshape(-1), top_sid_locals[:, :, level].reshape(-1)),
            p_teacher.reshape(-1),
            accumulate=True,
        )
        q_global = q_global / q_global.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if level == 0:
            prefix_match = valid_mask
        else:
            prefix_match = (top_sid_tokens[:, :, :level] == target_sid_tokens[:, None, :level]).all(dim=-1) & valid_mask
        prefix_weights = p_teacher * prefix_match.float()
        q_prefix = torch.zeros(batch_size, width, dtype=torch.float, device=device)
        q_prefix.index_put_(
            (rows.reshape(-1), top_sid_locals[:, :, level].reshape(-1)),
            prefix_weights.reshape(-1),
            accumulate=True,
        )
        prefix_sums = q_prefix.sum(dim=-1, keepdim=True)
        q_prefix = torch.where(prefix_sums > 0, q_prefix / prefix_sums.clamp_min(1e-12), q_prefix)

        matched = prefix_match.sum(dim=-1).float()
        alpha = (matched / (matched + tau)).unsqueeze(1)
        q = alpha * q_prefix + (1.0 - alpha) * q_global
        q = q.clamp_min(0)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        target_local = target_sid_locals[:, level]
        ranks = rank_of_target_batch(q, target_local)
        entropies = -(q * q.clamp_min(1e-12).log()).sum(dim=-1)
        per_level_coprefs.append(q.cpu())
        per_level_prefix_counts.append(matched.cpu().long())
        per_level_entropies.append(entropies.cpu())
        per_level_ranks.append(ranks.cpu().long())

    output = []
    target_sid_tokens_cpu = target_sid_tokens.cpu()
    for row, record in enumerate(records):
        output.append({
            'sample_id': int(record['sample_id']),
            'target_item': str(record['target_item']),
            'target_sid': [int(token) for token in target_sid_tokens_cpu[row].tolist()],
            'copref': [per_level_coprefs[level][row] for level in range(num_levels)],
            'prefix_count': [int(per_level_prefix_counts[level][row]) for level in range(num_levels)],
            'entropy': [float(per_level_entropies[level][row]) for level in range(num_levels)],
            'target_code_rank': [int(per_level_ranks[level][row]) for level in range(num_levels)],
        })
    return output


def build_records_batch_fast(records, item_to_idx, sid_tokens, sid_locals, level_token_ids, temp, tau, device):
    """Return column-oriented records to avoid per-record tensor slicing in the hot path."""
    batch_size = len(records)
    num_levels = sid_tokens.shape[1]
    top_m = max(len(record['top_items']) for record in records)
    invalid_item_idx = len(item_to_idx)

    top_item_indices = torch.full((batch_size, top_m), invalid_item_idx, dtype=torch.long)
    score_matrix = torch.full((batch_size, top_m), float('-inf'), dtype=torch.float)
    target_indices = torch.empty(batch_size, dtype=torch.long)

    for row, record in enumerate(records):
        target_indices[row] = item_to_idx[str(record['target_item'])]
        cur_scores = torch.as_tensor(record['top_scores'], dtype=torch.float)
        mapped_items = [item_to_idx.get(str(item), invalid_item_idx) for item in record['top_items']]
        top_item_indices[row, :len(mapped_items)] = torch.tensor(mapped_items, dtype=torch.long)
        score_matrix[row, :len(cur_scores)] = cur_scores

    valid_mask = top_item_indices != invalid_item_idx
    safe_top_indices = top_item_indices.clamp_max(invalid_item_idx - 1).to(device)
    target_indices = target_indices.to(device)
    valid_mask = valid_mask.to(device)
    score_matrix = score_matrix.to(device)

    p_teacher = torch.softmax(score_matrix / temp, dim=-1) * valid_mask.float()
    p_teacher = p_teacher / p_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    target_sid_tokens = sid_tokens.index_select(0, target_indices)
    target_sid_locals = sid_locals.index_select(0, target_indices)
    top_sid_tokens = sid_tokens.index_select(0, safe_top_indices.reshape(-1)).view(batch_size, top_m, num_levels)
    top_sid_locals = sid_locals.index_select(0, safe_top_indices.reshape(-1)).view(batch_size, top_m, num_levels)

    coprefs_by_level, prefix_counts_by_level, entropies_by_level, ranks_by_level = [], [], [], []
    rows = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, top_m)
    for level, tokens in enumerate(level_token_ids):
        width = len(tokens)
        q_global = torch.zeros(batch_size, width, dtype=torch.float, device=device)
        q_global.index_put_(
            (rows.reshape(-1), top_sid_locals[:, :, level].reshape(-1)),
            p_teacher.reshape(-1),
            accumulate=True,
        )
        q_global = q_global / q_global.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        if level == 0:
            prefix_match = valid_mask
        else:
            prefix_match = (top_sid_tokens[:, :, :level] == target_sid_tokens[:, None, :level]).all(dim=-1) & valid_mask
        prefix_weights = p_teacher * prefix_match.float()
        q_prefix = torch.zeros(batch_size, width, dtype=torch.float, device=device)
        q_prefix.index_put_(
            (rows.reshape(-1), top_sid_locals[:, :, level].reshape(-1)),
            prefix_weights.reshape(-1),
            accumulate=True,
        )
        prefix_sums = q_prefix.sum(dim=-1, keepdim=True)
        q_prefix = torch.where(prefix_sums > 0, q_prefix / prefix_sums.clamp_min(1e-12), q_prefix)

        matched = prefix_match.sum(dim=-1).float()
        alpha = (matched / (matched + tau)).unsqueeze(1)
        q = alpha * q_prefix + (1.0 - alpha) * q_global
        q = q.clamp_min(0)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        target_local = target_sid_locals[:, level]
        coprefs_by_level.append(q.cpu())
        prefix_counts_by_level.append(matched.cpu().long())
        entropies_by_level.append((-(q * q.clamp_min(1e-12).log()).sum(dim=-1)).cpu())
        ranks_by_level.append(rank_of_target_batch(q, target_local).cpu().long())

    return {
        'records': records,
        'target_sid_tokens': target_sid_tokens.cpu(),
        'coprefs_by_level': coprefs_by_level,
        'prefix_counts_by_level': prefix_counts_by_level,
        'entropies_by_level': entropies_by_level,
        'ranks_by_level': ranks_by_level,
    }


def append_batch_output(output, batch_result):
    records = batch_result['records']
    target_sid_tokens = batch_result['target_sid_tokens']
    coprefs_by_level = batch_result['coprefs_by_level']
    prefix_counts_by_level = batch_result['prefix_counts_by_level']
    entropies_by_level = batch_result['entropies_by_level']
    ranks_by_level = batch_result['ranks_by_level']
    num_levels = len(coprefs_by_level)
    for row, record in enumerate(records):
        output.append({
            'sample_id': int(record['sample_id']),
            'target_item': str(record['target_item']),
            'target_sid': [int(token) for token in target_sid_tokens[row].tolist()],
            'copref': [coprefs_by_level[level][row] for level in range(num_levels)],
            'prefix_count': [int(prefix_counts_by_level[level][row]) for level in range(num_levels)],
            'entropy': [float(entropies_by_level[level][row]) for level in range(num_levels)],
            'target_code_rank': [int(ranks_by_level[level][row]) for level in range(num_levels)],
        })


def append_tensor_output(output, batch_result):
    """Append a GPU-built batch in column format for fast save/load."""
    if not output:
        output.update({
            'format': 'colagr_copref_tensor_v1',
            'sample_id': [],
            'target_item': [],
            'target_sid': [],
            'copref': [[] for _ in batch_result['coprefs_by_level']],
            'prefix_count': [],
            'entropy': [],
            'target_code_rank': [],
        })

    records = batch_result['records']
    output['sample_id'].append(torch.tensor(
        [int(record['sample_id']) for record in records],
        dtype=torch.long,
    ))
    output['target_item'].extend(str(record['target_item']) for record in records)
    output['target_sid'].append(batch_result['target_sid_tokens'].long())
    for level, copref in enumerate(batch_result['coprefs_by_level']):
        output['copref'][level].append(copref.float())
    output['prefix_count'].append(torch.stack(batch_result['prefix_counts_by_level'], dim=1).long())
    output['entropy'].append(torch.stack(batch_result['entropies_by_level'], dim=1).float())
    output['target_code_rank'].append(torch.stack(batch_result['ranks_by_level'], dim=1).long())
def finalize_tensor_output(output):
    if not output:
        raise ValueError('No CoPref records were built; check teacher_topm inputs.')
    result = {
        'format': output['format'],
        'sample_id': torch.cat(output['sample_id'], dim=0),
        'target_item': output['target_item'],
        'target_sid': torch.cat(output['target_sid'], dim=0),
        'copref': [torch.cat(level_parts, dim=0) for level_parts in output['copref']],
        'prefix_count': torch.cat(output['prefix_count'], dim=0),
        'entropy': torch.cat(output['entropy'], dim=0),
        'target_code_rank': torch.cat(output['target_code_rank'], dim=0),
    }
    return result


def save_copref_output(output, path, tensor_output):
    if tensor_output:
        torch.save(finalize_tensor_output(output), path)
    else:
        torch.save(output, path)


def summarize_copref_sums(output, tensor_output):
    if tensor_output:
        return torch.cat([
            part.sum(dim=-1)
            for level_parts in output['copref']
            for part in level_parts
        ])
    return torch.tensor([
        float(level_dist.sum().item())
        for item in output for level_dist in item['copref']
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts_dir', required=True)
    parser.add_argument('--teacher_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--temp', type=float, default=2.0)
    parser.add_argument('--tau', type=float, default=5.0)
    parser.add_argument('--min_sid_coverage', type=float, default=0.99)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--legacy_per_record', action='store_true')
    parser.add_argument(
        '--quiet_progress', action='store_true',
        help='Disable tqdm refreshes; print only split summaries.',
    )
    parser.add_argument(
        '--output_format',
        choices=['tensor', 'records'],
        default='tensor',
        help='tensor is faster for full runs; records keeps the old list-of-dicts cache.',
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    item2sid = json.load(open(os.path.join(args.artifacts_dir, 'item2sid.json')))
    level_token_ids = torch.load(os.path.join(args.artifacts_dir, 'level_token_ids.pt'), map_location='cpu')
    token_to_local = local_maps(level_token_ids)
    sid_cache = prepare_sid_cache(item2sid, token_to_local)
    sid_items = set(sid_cache)
    item_to_idx, sid_tokens_cpu, sid_locals_cpu = prepare_item_tensor_maps(item2sid, token_to_local)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    sid_tokens = sid_tokens_cpu.to(device)
    sid_locals = sid_locals_cpu.to(device)
    print(f'Build CoPref device={device}, batch_size={args.batch_size}, items={len(item_to_idx)}')

    for split in ['train', 'val', 'test']:
        teacher_path = os.path.join(args.teacher_dir, f'teacher_topm_{split}.pt')
        if not os.path.exists(teacher_path):
            continue
        teacher_records = torch.load(teacher_path, map_location='cpu')
        tensor_output = args.output_format == 'tensor' and not args.legacy_per_record
        output = {} if tensor_output else []
        covered, total = 0, 0
        missing_examples = []
        progress = tqdm(
            teacher_records,
            desc=f'Build CoPref [{split}]',
            disable=args.quiet_progress,
            mininterval=30.0,
            dynamic_ncols=False,
        )
        record_batch = []
        for record in progress:
            total += len(record['top_items'])
            top_item_keys = [str(item) for item in record['top_items']]
            covered += sum(1 for item in top_item_keys if item in sid_items)
            if len(missing_examples) < 20:
                for item in top_item_keys:
                    if item not in sid_items:
                        missing_examples.append(item)
                        if len(missing_examples) >= 20:
                            break
            if args.legacy_per_record:
                output.append(build_record(record, item2sid, sid_cache, level_token_ids, args.temp, args.tau))
            else:
                record_batch.append(record)
                if len(record_batch) >= args.batch_size:
                    batch_result = build_records_batch_fast(
                        record_batch,
                        item_to_idx,
                        sid_tokens,
                        sid_locals,
                        level_token_ids,
                        args.temp,
                        args.tau,
                        device,
                    )
                    if tensor_output:
                        append_tensor_output(output, batch_result)
                    else:
                        append_batch_output(output, batch_result)
                    record_batch = []
            if not args.quiet_progress:
                progress.set_postfix({
                    'records': (sum(part.numel() for part in output.get('sample_id', [])) if tensor_output else len(output)) + len(record_batch),
                    'coverage': f'{covered / max(total, 1):.4f}',
                })
        if record_batch:
            batch_result = build_records_batch_fast(
                record_batch,
                item_to_idx,
                sid_tokens,
                sid_locals,
                level_token_ids,
                args.temp,
                args.tau,
                device,
            )
            if tensor_output:
                append_tensor_output(output, batch_result)
            else:
                append_batch_output(output, batch_result)
        coverage = covered / max(total, 1)
        if coverage < args.min_sid_coverage:
            raise ValueError(
                f'{split} SID coverage {coverage:.4f} < {args.min_sid_coverage}; '
                f'missing examples: {missing_examples}'
            )
        sums = summarize_copref_sums(output, tensor_output)
        save_copref_output(output, os.path.join(args.output_dir, f'copref_{split}.pt'), tensor_output)
        if tensor_output:
            n_records = int(sum(part.numel() for part in output['sample_id']))
        else:
            n_records = len(output)
        print(
            f'{split}: wrote {n_records} records; coverage={coverage:.4f}; '
            f'sum=[{sums.min():.4f},{sums.max():.4f}]; format={"tensor" if tensor_output else "records"}'
        )


if __name__ == '__main__':
    main()
