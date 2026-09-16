import argparse
import json
import os

import numpy as np
import torch

from colagr.common import ensure_dir, load_dataset_and_tokenizer, parse_unknown_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='CoLaGR')
    parser.add_argument('--dataset', default='AmazonReviews2023')
    parser.add_argument('--category', required=True)
    parser.add_argument('--vq_method', default='rqkmeans')
    parser.add_argument('--output_dir', required=True)
    args, unknown = parser.parse_known_args()

    overrides = parse_unknown_args(unknown)
    overrides.update({'category': args.category, 'vq_method': args.vq_method})
    _, dataset, _, tokenizer = load_dataset_and_tokenizer(args.model, args.dataset, overrides)
    ensure_dir(args.output_dir)

    item2sid = {str(item): [int(token) for token in tokens] for item, tokens in tokenizer.item2tokens.items()}
    sid2item = {','.join(map(str, sid)): item for item, sid in item2sid.items()}
    sid_tokens = np.asarray([sid for _, sid in sorted(item2sid.items())], dtype=np.int64)

    level_token_ids = []
    for level in range(tokenizer.n_digit):
        tokens = sorted({sid[level] for sid in item2sid.values()})
        level_tensor = torch.tensor(tokens, dtype=torch.long)
        assert len(level_tensor) > 0
        assert not torch.isin(level_tensor, torch.tensor([0, tokenizer.eos_token] + tokenizer.coreason_token_ids)).any()
        level_token_ids.append(level_tensor)

    trie = {}
    for sid in item2sid.values():
        assert len(sid) == tokenizer.n_digit
        assert sid2item[','.join(map(str, sid))]
        for level in range(tokenizer.n_digit):
            prefix = ','.join(map(str, sid[:level]))
            trie.setdefault(prefix, set()).add(int(sid[level]))
    trie = {prefix: sorted(values) for prefix, values in trie.items()}

    np.save(os.path.join(args.output_dir, 'sid_tokens.npy'), sid_tokens)
    json.dump(item2sid, open(os.path.join(args.output_dir, 'item2sid.json'), 'w'))
    json.dump(sid2item, open(os.path.join(args.output_dir, 'sid2item.json'), 'w'))
    torch.save(level_token_ids, os.path.join(args.output_dir, 'level_token_ids.pt'))
    json.dump(trie, open(os.path.join(args.output_dir, 'valid_prefix_trie.json'), 'w'))
    json.dump({
        'dataset': args.dataset,
        'category': args.category,
        'vq_method': args.vq_method,
        'num_items': len(item2sid),
        'num_levels': tokenizer.n_digit,
        'codebook_sizes': tokenizer.codebook_sizes,
        'eos_token': int(tokenizer.eos_token),
        'coreason_token_ids': [int(x) for x in tokenizer.coreason_token_ids],
        'raw_num_items': int(dataset.n_items),
    }, open(os.path.join(args.output_dir, 'tokenizer_meta.json'), 'w'), indent=2)
    print(f'Exported SID artifacts to {args.output_dir}')


if __name__ == '__main__':
    main()
