import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from genrec.utils import get_pipeline


def sid_to_item_map(tokenizer):
    mapping = {}
    for item, tokens in tokenizer.item2tokens.items():
        mapping[tuple(int(token) for token in tokens)] = str(item)
    return mapping


def export_split_candidates(pipeline, split, output_path, with_scores=False):
    dataloader = DataLoader(
        pipeline.tokenized_datasets[split],
        batch_size=pipeline.config['eval_batch_size'],
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[split],
        num_workers=pipeline.config.get('dataloader_num_workers', 4),
        pin_memory=True,
        persistent_workers=pipeline.config.get('dataloader_num_workers', 4) > 0,
    )
    pipeline.model, dataloader = pipeline.accelerator.prepare(
        pipeline.model, dataloader
    )
    pipeline.model.eval()
    unwrapped_model = pipeline.accelerator.unwrap_model(pipeline.model)
    token_to_item = sid_to_item_map(pipeline.tokenizer)
    records = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(pipeline.accelerator.device, non_blocking=True)
                if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            generated = unwrapped_model.generate(
                batch,
                n_return_sequences=max(pipeline.config['topk']),
                return_scores=with_scores,
            )
            if with_scores:
                preds, scores = generated
                scores = scores.detach().cpu()
            else:
                preds = generated
            preds = preds.detach().cpu()
            sample_ids = batch['sample_id'].detach().cpu().tolist()
            labels = batch['labels'][:, :pipeline.tokenizer.n_digit].detach().cpu().tolist()
            for row, sample_id in enumerate(sample_ids):
                candidates = []
                generation_scores = []
                seen = set()
                target_tokens = tuple(int(token) for token in labels[row])
                target_item = token_to_item.get(target_tokens)
                if target_item is None:
                    raise ValueError(f'Cannot map target SID for sample {sample_id}')
                for beam_rank, sid in enumerate(preds[row].tolist()):
                    item = token_to_item.get(tuple(int(token) for token in sid))
                    if item is None or item in seen:
                        continue
                    candidates.append(item)
                    if with_scores:
                        # ``generate_colagr_beam`` returns the accumulated
                        # autoregressive log-probability for this exact beam.
                        generation_scores.append(float(scores[row, beam_rank].item()))
                    seen.add(item)
                record = {
                    'sample_id': int(sample_id),
                    'candidates': candidates,
                }
                if with_scores:
                    if len(candidates) != len(generation_scores):
                        raise RuntimeError('Candidate and generation-score lengths diverged.')
                    record['generation_scores'] = generation_scores
                    record['generation_score_type'] = 'cumulative_log_probability'
                records.append(record)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Keep candidates and scores in one tensor artifact so their alignment is
    # preserved across validation, test, and downstream CoLeaf diagnostics.
    torch.save(records, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--model', default='CoLaGR')
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', required=True)
    parser.add_argument('--config_file', default='genrec/models/CoLaGR/config.yaml')
    parser.add_argument('--vq_method', default='rqkmeans')
    parser.add_argument('--level_token_ids_path', required=True)
    parser.add_argument('--valid_prefix_trie_path', required=True)
    parser.add_argument('--copref_train_path', default=None)
    parser.add_argument('--copref_val_path', default=None)
    parser.add_argument('--copref_test_path', default=None)
    parser.add_argument('--lambda_pref', type=float, default=1.0)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--num_beams', type=int, default=50)
    parser.add_argument('--topk', default='[5,10,50]')
    parser.add_argument('--with_scores', action='store_true')
    parser.add_argument('--allow_checkpoint_extra_keys', action='store_true')
    parser.add_argument('--use_copref_module', default=True)
    parser.add_argument('--use_copref_loss', default=True)
    parser.add_argument('--grounding_anchor', choices=('action', 'cp'), default='cp')
    parser.add_argument('--direct_copref_alignment', default=False)
    parser.add_argument(
        '--copref_reasoning_mode',
        choices=('per_level', 'one_shot', 'centralized'),
        default='per_level',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = vars(args).copy()
    checkpoint = overrides.pop('checkpoint')
    output = overrides.pop('output')
    split = overrides.pop('split')
    model = overrides.pop('model')
    dataset = overrides.pop('dataset')
    config_file = overrides.pop('config_file')
    overrides['eval_only'] = True
    pipeline = get_pipeline(model)(
        model_name=model,
        dataset_name=dataset,
        checkpoint_path=checkpoint,
        config_file=config_file,
        config_dict=overrides,
    )
    export_split_candidates(pipeline, split, output, with_scores=args.with_scores)


if __name__ == '__main__':
    main()
