"""Train the frozen SASRec teacher used to export collaborative preferences."""

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from model import SASRec
from utils import WarpSampler, data_partition, evaluate, save_eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--maxlen', type=int, default=128)
    parser.add_argument('--hidden_units', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--l2_emb', type=float, default=0.0)
    parser.add_argument('--device', default='0')
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(os.path.dirname(__file__))
    args.is_hpu = args.device == 'hpu'
    if args.is_hpu:
        import habana_frameworks.torch.core as htcore
        args.device = torch.device('hpu')
    elif args.device != 'cpu':
        args.device = f'cuda:{args.device}'

    data_root = os.path.abspath(os.path.join('..', f'data_{args.dataset}'))
    required = [os.path.join(data_root, f'{args.dataset}_{split}.txt') for split in ('train', 'valid', 'test')]
    if not all(os.path.isfile(path) for path in required):
        raise FileNotFoundError('Export teacher sequences first; missing: ' + ', '.join(path for path in required if not os.path.isfile(path)))

    dataset = data_partition(args.dataset, args)
    user_train, _, _, usernum, itemnum, _ = dataset
    model = SASRec(usernum, itemnum, args).to(args.device)
    for parameter in model.parameters():
        if parameter.ndim > 1:
            torch.nn.init.xavier_normal_(parameter)
    start_epoch = 1
    if args.state_dict_path:
        kwargs, weights = torch.load(args.state_dict_path, map_location='cpu', weights_only=False)
        kwargs['args'].device = args.device
        model = SASRec(**kwargs).to(args.device)
        model.load_state_dict(weights)
        if 'epoch=' in args.state_dict_path:
            start_epoch = int(args.state_dict_path.split('epoch=', 1)[1].split('.', 1)[0]) + 1

    if args.inference_only:
        for cutoff in (10, 20):
            ndcg, recall = evaluate(model, dataset, args, ranking=cutoff)
            print(f'test NDCG@{cutoff}: {ndcg:.4f}, HR@{cutoff}: {recall:.4f}')
        return

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    started = time.time()
    try:
        for epoch in tqdm(range(start_epoch, args.num_epochs + 1)):
            model.train()
            total_loss = 0.0
            num_batch = max(1, len(user_train) // args.batch_size)
            for _ in range(num_batch):
                user, sequence, positive, negative = map(np.asarray, sampler.next_batch())
                positive_logits, negative_logits = model(user, sequence, positive, negative)
                valid = np.where(positive != 0)
                loss = criterion(positive_logits[valid], torch.ones_like(positive_logits[valid]))
                loss += criterion(negative_logits[valid], torch.zeros_like(negative_logits[valid]))
                for parameter in model.item_emb.parameters():
                    loss += args.l2_emb * torch.norm(parameter)
                optimizer.zero_grad()
                loss.backward()
                if args.is_hpu:
                    htcore.mark_step()
                optimizer.step()
                if args.is_hpu:
                    htcore.mark_step()
                total_loss += loss.item()
            print(f'epoch={epoch} loss={total_loss / num_batch:.6f}', flush=True)

        output_dir = os.path.join(os.path.dirname(__file__), args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        checkpoint = os.path.join(output_dir, f'SASRec_saving.epoch={args.num_epochs}.lr={args.lr}.layer={args.num_blocks}.head={args.num_heads}.hidden={args.hidden_units}.maxlen={args.maxlen}.pth')
        torch.save([model.kwargs, model.state_dict()], checkpoint)
        save_eval(model, dataset, args)
        print(f'checkpoint={checkpoint} elapsed_seconds={time.time() - started:.1f}')
    finally:
        sampler.close()


if __name__ == '__main__':
    main()
