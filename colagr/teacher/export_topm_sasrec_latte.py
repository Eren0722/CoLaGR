import argparse
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from colagr.common import ensure_dir, load_dataset_and_tokenizer, parse_unknown_args, split_examples


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {'1', 'true', 'yes', 'y'}


def import_llmsrec_sasrec(llmsrec_root):
    if llmsrec_root is None:
        model_path = os.path.join(os.path.dirname(__file__), 'llmsrec_sasrec', 'model.py')
    else:
        direct_path = os.path.join(llmsrec_root, 'model.py')
        nested_path = os.path.join(llmsrec_root, 'SeqRec', 'sasrec', 'model.py')
        model_path = direct_path if os.path.exists(direct_path) else nested_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'LLM-SRec SASRec model.py not found: {model_path}')
    spec = importlib.util.spec_from_file_location('llmsrec_sasrec_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SASRec


def normalize_device(device_arg):
    if device_arg == 'cpu' or not torch.cuda.is_available():
        return 'cpu'
    if isinstance(device_arg, str) and device_arg.startswith('cuda'):
        return device_arg
    return f'cuda:{device_arg}'


def load_sasrec_checkpoint(SASRec, checkpoint_path, usernum, itemnum, device, allow_untrained, defaults):
    if checkpoint_path is None:
        if not allow_untrained:
            raise ValueError(
                '--checkpoint is required for SASRec teacher export. '
                'Use --allow_untrained=True only for smoke tests.'
            )
        args = SimpleNamespace(**defaults)
        args.device = device
        model = SASRec(usernum, itemnum, args).to(device)
        return model.eval(), args

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) == 2:
        kwargs, state_dict = checkpoint
        kwargs['args'].device = device
        model = SASRec(**kwargs).to(device)
        model.load_state_dict(state_dict)
        return model.eval(), kwargs['args']

    if isinstance(checkpoint, dict) and 'kwargs' in checkpoint and 'state_dict' in checkpoint:
        kwargs = checkpoint['kwargs']
        kwargs['args'].device = device
        model = SASRec(**kwargs).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval(), kwargs['args']

    args = SimpleNamespace(**defaults)
    args.device = device
    model = SASRec(usernum, itemnum, args).to(device)
    model.load_state_dict(checkpoint)
    return model.eval(), args


def build_sasrec_sequence(history_ids, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for item_id in reversed(history_ids):
        seq[idx] = item_id
        idx -= 1
        if idx == -1:
            break
    return seq


def score_topm(model, user_id, history_ids, candidate_ids, id2item, maxlen, top_m, score_batch_size):
    seq = build_sasrec_sequence(history_ids, maxlen)
    scores = []
    with torch.no_grad():
        for start in range(0, len(candidate_ids), score_batch_size):
            cur_candidates = candidate_ids[start:start + score_batch_size]
            logits = model.predict(
                np.array([user_id], dtype=np.int32),
                np.array([seq], dtype=np.int32),
                np.array(cur_candidates, dtype=np.int32),
            )
            logits = logits.detach().cpu().reshape(-1)
            scores.append(logits)
    scores = torch.cat(scores, dim=0)
    top_scores, top_indices = torch.topk(scores, k=min(top_m, scores.numel()))
    top_item_ids = [candidate_ids[int(idx)] for idx in top_indices.tolist()]
    top_items = [id2item[item_id] for item_id in top_item_ids]
    return top_items, top_scores.float()


def get_item_embedding_matrix(model, candidate_ids, device):
    ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=device)
    if model.nn_parameter:
        return model.item_emb.index_select(0, ids)
    return model.item_emb(ids)


def score_topm_batch(
    model, examples, all_candidate_ids, item_embs, id2item, dataset, maxlen, top_m,
):
    seqs = np.stack([
        build_sasrec_sequence(
            [dataset.item2id[item] for item in example['history_items']],
            maxlen,
        )
        for example in examples
    ])
    candidate_tensor = torch.as_tensor(all_candidate_ids, dtype=torch.long, device=model.dev)

    with torch.no_grad():
        log_feats = model.log2feats(seqs)
        final_feats = log_feats[:, -1, :]
        scores = final_feats.matmul(item_embs.T)

        for row, example in enumerate(examples):
            history_ids = {dataset.item2id[item] for item in example['history_items']}
            if history_ids:
                history_mask = torch.isin(
                    candidate_tensor,
                    torch.as_tensor(list(history_ids), dtype=torch.long, device=model.dev),
                )
                scores[row, history_mask] = float('-inf')

        top_scores, top_indices = torch.topk(scores, k=min(top_m, scores.shape[1]), dim=-1)

    records = []
    hits = 0
    top_scores_cpu = top_scores.detach().cpu()
    top_indices_cpu = top_indices.detach().cpu()
    for row, example in enumerate(examples):
        top_item_ids = [all_candidate_ids[int(idx)] for idx in top_indices_cpu[row].tolist()]
        top_items = [id2item[item_id] for item_id in top_item_ids]
        hits += int(example['target_item'] in top_items)
        records.append({
            **example,
            'top_items': top_items,
            'top_scores': top_scores_cpu[row].float(),
        })
    return records, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--top_m', type=int, default=200)
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--limit_samples', type=int, default=None)
    parser.add_argument('--llmsrec_root', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--score_batch_size', type=int, default=8192)
    parser.add_argument('--export_batch_size', type=int, default=512)
    parser.add_argument('--legacy_per_sample', type=parse_bool, default=False)
    parser.add_argument('--allow_untrained', type=parse_bool, default=False)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--maxlen', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--nn_parameter', type=parse_bool, default=False)
    parser.add_argument(
        '--quiet_progress', action='store_true',
        help='Disable tqdm refreshes; print only split summaries.',
    )
    args, unknown = parser.parse_known_args()

    overrides = parse_unknown_args(unknown)
    overrides.update({'category': args.category})
    _, dataset, split_datasets, tokenizer = load_dataset_and_tokenizer('CoLaGR', args.dataset, overrides)
    ensure_dir(args.output_dir)

    llmsrec_root = os.path.abspath(args.llmsrec_root) if args.llmsrec_root is not None else None
    SASRec = import_llmsrec_sasrec(llmsrec_root)
    device = normalize_device(str(args.device))
    usernum = dataset.n_users - 1
    itemnum = dataset.n_items - 1
    defaults = {
        'device': device,
        'hidden_units': args.hidden_units,
        'maxlen': args.maxlen,
        'num_blocks': args.num_blocks,
        'num_heads': args.num_heads,
        'dropout_rate': args.dropout_rate,
        'nn_parameter': args.nn_parameter,
    }
    model, sasrec_args = load_sasrec_checkpoint(
        SASRec,
        args.checkpoint,
        usernum,
        itemnum,
        device,
        args.allow_untrained,
        defaults,
    )

    if model.item_num != itemnum:
        raise ValueError(
            f'SASRec checkpoint item_num={model.item_num} does not match Latte itemnum={itemnum}. '
            'Train/export SASRec on the same Latte dataset id mapping.'
        )
    if model.user_num != usernum:
        raise ValueError(
            f'SASRec checkpoint user_num={model.user_num} does not match Latte usernum={usernum}. '
            'Train/export SASRec on the same Latte dataset id mapping.'
        )

    id2item = dataset.id_mapping['id2item']
    all_candidate_ids = [
        item_id for item_id in range(1, dataset.n_items)
        if id2item[item_id] in tokenizer.item2tokens
    ]
    maxlen = int(getattr(sasrec_args, 'maxlen', args.maxlen))
    item_embs = None
    if not args.legacy_per_sample:
        item_embs = get_item_embedding_matrix(model, all_candidate_ids, device)
        print(
            f'Using batched full-item scoring: device={device}, '
            f'export_batch_size={args.export_batch_size}, candidates={len(all_candidate_ids)}'
        )

    for split in [part.strip() for part in args.splits.split(',') if part.strip()]:
        records = []
        hit_count = 0
        if args.limit_samples is None:
            split_total = len(split_datasets[split])
            if split == 'train':
                split_total = sum(len(seq) - 1 for seq in split_datasets[split]['item_seq'])
        else:
            split_total = args.limit_samples

        progress = tqdm(
            split_examples(split_datasets[split], split),
            total=split_total,
            desc=f'Export top-M [{split}]',
            disable=args.quiet_progress,
            mininterval=30.0,
            dynamic_ncols=False,
        )
        batch_examples = []
        for example in progress:
            batch_examples.append(example)
            if len(batch_examples) < args.export_batch_size:
                if args.limit_samples is None or len(records) + len(batch_examples) < args.limit_samples:
                    continue

            if args.legacy_per_sample:
                batch_records, batch_hits = [], 0
                for cur_example in batch_examples:
                    user_id = dataset.user2id[cur_example['user']]
                    history_ids = [dataset.item2id[item] for item in cur_example['history_items']]
                    history_set = set(history_ids)
                    candidate_ids = [item_id for item_id in all_candidate_ids if item_id not in history_set]
                    top_items, top_scores = score_topm(
                        model,
                        user_id,
                        history_ids,
                        candidate_ids,
                        id2item,
                        maxlen,
                        args.top_m,
                        args.score_batch_size,
                    )
                    batch_hits += int(cur_example['target_item'] in top_items)
                    batch_records.append({
                        **cur_example,
                        'top_items': top_items,
                        'top_scores': top_scores,
                    })
            else:
                batch_records, batch_hits = score_topm_batch(
                    model,
                    batch_examples,
                    all_candidate_ids,
                    item_embs,
                    id2item,
                    dataset,
                    maxlen,
                    args.top_m,
                )

            hit_count += batch_hits
            records.extend(batch_records)
            batch_examples = []
            if not args.quiet_progress:
                progress.set_postfix({
                    'records': len(records),
                    f'hit@{args.top_m}': f'{hit_count / max(len(records), 1):.4f}',
                })
            if args.limit_samples is not None and len(records) >= args.limit_samples:
                break

        if batch_examples and (args.limit_samples is None or len(records) < args.limit_samples):
            if args.legacy_per_sample:
                batch_records, batch_hits = [], 0
                for cur_example in batch_examples:
                    user_id = dataset.user2id[cur_example['user']]
                    history_ids = [dataset.item2id[item] for item in cur_example['history_items']]
                    history_set = set(history_ids)
                    candidate_ids = [item_id for item_id in all_candidate_ids if item_id not in history_set]
                    top_items, top_scores = score_topm(
                        model,
                        user_id,
                        history_ids,
                        candidate_ids,
                        id2item,
                        maxlen,
                        args.top_m,
                        args.score_batch_size,
                    )
                    batch_hits += int(cur_example['target_item'] in top_items)
                    batch_records.append({
                        **cur_example,
                        'top_items': top_items,
                        'top_scores': top_scores,
                    })
            else:
                batch_records, batch_hits = score_topm_batch(
                    model,
                    batch_examples,
                    all_candidate_ids,
                    item_embs,
                    id2item,
                    dataset,
                    maxlen,
                    args.top_m,
                )
            hit_count += batch_hits
            records.extend(batch_records)

        if args.limit_samples is not None:
            records = records[:args.limit_samples]

        torch.save(records, os.path.join(args.output_dir, f'teacher_topm_{split}.pt'))
        hit_rate = hit_count / max(len(records), 1)
        print(f'{split}: wrote {len(records)} records; SASRec target hit@{args.top_m}={hit_rate:.4f}')


if __name__ == '__main__':
    main()
