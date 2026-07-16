import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from SA import SequenceDataset, SemanticAlignmentModule, data_augmentation, data_partition, prepare_semantic_assets
from SeqRec.bert4rec.model import BERT4Rec
from SeqRec.bert4rec.train_utils import (
    append_eval_results,
    data_partition_si,
    prepare_device,
    run_evaluation,
    save_checkpoint,
    set_seed,
)


class SACPTrainingModule(nn.Module):
    def __init__(self, recsys_model: nn.Module, alignment_module: nn.Module):
        super().__init__()
        self.recsys_model = recsys_model
        self.alignment_module = alignment_module

    def forward(self, seq_unique_id, padded_seq, target, aug_seq1, aug_seq2, neighbor_seqs):
        rec_loss = self.recsys_model.calculate_loss(padded_seq, target)
        sa_loss, sa_info = self.alignment_module.compute_batch_losses(
            seq_unique_ids=seq_unique_id,
            padded_seq=padded_seq,
            aug_seq1=aug_seq1,
            aug_seq2=aug_seq2,
            neighbor_seqs=neighbor_seqs,
            allow_recsys_grad=True,
        )
        return rec_loss + sa_loss, rec_loss, sa_loss, sa_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a small recommendation teacher with SACP")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--sa_asset_root", type=str, default="./SA_assets")
    parser.add_argument("--sa_item_semantic", type=str, default=None)
    parser.add_argument("--sa_user_semantic", type=str, default=None)
    parser.add_argument("--sa_seq_keys", type=str, default=None)
    parser.add_argument("--sa_item_neighbors", type=str, default=None)
    parser.add_argument("--sa_user_neighbors", type=str, default=None)
    parser.add_argument("--sa_alpha", type=float, default=0.1)
    parser.add_argument("--sa_beta", type=float, default=0.1)
    parser.add_argument("--sa_temperature", type=float, default=0.2)
    parser.add_argument("--sa_mlm_probability", type=float, default=0.05)
    parser.add_argument("--sa_k_num", type=int, default=10)
    parser.add_argument("--sa_similarity", choices=["dot", "cos"], default="cos")
    parser.add_argument("--sa_repr_mode", choices=["mean"], default="mean")
    parser.add_argument("--sa_use_projection_head", action="store_true")
    parser.add_argument("--sa_proj_hidden_dim", type=int, default=0)
    parser.add_argument("--sa_proj_act", choices=["gelu", "relu"], default="gelu")
    parser.add_argument("--sa_contrast_norm", action="store_true")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--inner_size", type=int, default=256)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--attn_dropout_prob", type=float, default=0.2)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--bert_mask_prob", type=float, default=0.15)
    parser.add_argument("--bert_rec_objective", choices=["next_item"], default="next_item")
    parser.add_argument("--recsys_backbone", choices=["bert4rec"], default="bert4rec")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save_dir", type=str, default="./SeqRec/bert4rec")
    parser.add_argument("--save_prefix", type=str, default="cds_bertstyle_nextitem_full")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        args.device = torch.device(f"cuda:{local_rank}")
    else:
        args.device = prepare_device(args)
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    args.distributed = distributed
    return args.device


def is_main_process(args) -> bool:
    return getattr(args, "rank", 0) == 0


def main_print(args, *values, **kwargs):
    if is_main_process(args):
        print(*values, **kwargs)


def reduce_epoch_sums(args, values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if getattr(args, "distributed", False):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def main():
    args = parse_args()
    args.enable_semantic_module = True
    set_seed(args.seed)
    device = setup_distributed(args)

    user_train, _, _, usernum, itemnum, _ = data_partition(args.dataset, root_dir=args.data_root)
    si_data = data_partition_si(args.dataset, data_root=args.data_root)
    _, _, _, si_usernum, si_itemnum = si_data
    if si_usernum != usernum or si_itemnum != itemnum:
        raise ValueError("Training split and evaluation split are inconsistent.")

    prepare_semantic_assets(args, user_train)
    uid_list, item_list, target_list, item_list_length, seq_key_list = data_augmentation(
        user_train,
        args.maxlen,
        args.seq_keys_to_int,
    )
    train_dataset = SequenceDataset(
        args,
        uid_list,
        item_list,
        target_list,
        item_list_length,
        seq_key_list,
        args.maxlen,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
    ) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
    )

    model = BERT4Rec(usernum, itemnum, args).to(device)
    alignment_module = SemanticAlignmentModule(args, model, detach_recsys_grad=False).to(device)
    training_module = SACPTrainingModule(model, alignment_module).to(device)
    active_module = DDP(
        training_module,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
    ) if args.distributed else training_module
    optimizer = torch.optim.Adam(
        [
            {"params": [p for p in training_module.parameters() if p.requires_grad], "lr": args.learning_rate},
        ],
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    target_dir = os.path.join(args.save_dir, args.dataset, args.save_prefix)
    results_val_path = os.path.join(target_dir, "results_val.txt")
    results_test_path = os.path.join(target_dir, "results_test.txt")
    best_metric = -1.0
    best_loss = float("inf")
    best_loss_epoch = 0
    best_state = None
    patience_counter = 0

    main_print(
        args,
        f"Training small recommendation model with SACP: dataset={args.dataset}, users={usernum}, items={itemnum}, "
        f"examples={len(train_dataset)}, alpha={args.sa_alpha}, beta={args.sa_beta}, "
        f"world_size={args.world_size}"
    )

    for epoch in range(1, args.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        active_module.train()
        epoch_rec = 0.0
        epoch_sa = 0.0
        epoch_total = 0.0
        epoch_item_cl = 0.0
        epoch_user_cl = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", disable=not is_main_process(args))
        for batch in pbar:
            seq_unique_id, padded_seq, target, _, aug_seq1, aug_seq2, neighbor_seqs, _ = batch
            padded_seq = padded_seq.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            loss, rec_loss, sa_loss, sa_info = active_module(
                seq_unique_id,
                padded_seq,
                target,
                aug_seq1,
                aug_seq2,
                neighbor_seqs,
            )
            loss.backward()
            optimizer.step()

            epoch_rec += float(rec_loss.item())
            epoch_sa += float(sa_loss.item())
            epoch_total += float(loss.item())
            epoch_item_cl += float(sa_info.get("item_cl_loss", 0.0))
            epoch_user_cl += float(sa_info.get("user_cl_loss", 0.0))
            if is_main_process(args):
                pbar.set_postfix({"rec": f"{rec_loss.item():.3f}", "sa": f"{sa_loss.item():.3f}"})

        steps = max(len(train_loader), 1)
        epoch_rec, epoch_sa, epoch_total, epoch_item_cl, epoch_user_cl, total_steps = reduce_epoch_sums(
            args,
            [epoch_rec, epoch_sa, epoch_total, epoch_item_cl, epoch_user_cl, steps],
            device,
        )
        avg_total = epoch_total / max(total_steps, 1.0)
        if is_main_process(args):
            print(
                f"Epoch {epoch:03d}: total={avg_total:.4f}, rec={epoch_rec / total_steps:.4f}, "
                f"sa={epoch_sa / total_steps:.4f}, item_cl={epoch_item_cl / total_steps:.4f}, "
                f"user_cl={epoch_user_cl / total_steps:.4f}"
            )

            if avg_total < best_loss:
                best_loss = avg_total
                best_loss_epoch = epoch
                save_checkpoint(training_module.recsys_model, os.path.join(target_dir, "model_best.pth"))

        is_last_epoch = epoch == args.num_epochs
        should_eval = args.eval_every > 0 and (epoch % args.eval_every == 0 or is_last_epoch)
        should_stop = False
        if should_eval and is_main_process(args):
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(42)
            np.random.seed(42)
            val_metrics = run_evaluation(training_module.recsys_model, si_data, args, split="val")
            test_metrics = run_evaluation(training_module.recsys_model, si_data, args, split="test")
            random.setstate(py_state)
            np.random.set_state(np_state)

            append_eval_results(results_val_path, epoch, val_metrics, split="val")
            append_eval_results(results_test_path, epoch, test_metrics, split="test")
            print(f"Validation: {val_metrics}")
            print(f"Test: {test_metrics}")

            current_metric = val_metrics.get("NDCG@10", 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                best_state = {"epoch": epoch, "val": val_metrics, "test": test_metrics}
                best_path = os.path.join(target_dir, "model_metric_best.pth")
                save_checkpoint(training_module.recsys_model, best_path)
                print(f"Best validation NDCG@10={current_metric:.4f}; saved {best_path}")
            else:
                patience_counter += 1
                print(f"Patience {patience_counter}/{args.patience}; best NDCG@10={best_metric:.4f}")
            should_stop = patience_counter >= args.patience

        if args.distributed:
            stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int64, device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_eval and should_stop:
            if is_main_process(args):
                save_checkpoint(training_module.recsys_model, os.path.join(target_dir, "model.pth"))
                print(f"Early stopping at epoch {epoch}")
            break

        if is_last_epoch and is_main_process(args):
            save_checkpoint(training_module.recsys_model, os.path.join(target_dir, "model.pth"))

    if is_main_process(args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Small recommendation model training finished: {target_dir}")
        print(f"Best loss epoch={best_loss_epoch}, loss={best_loss:.4f}")
        if best_state is not None:
            print(f"Best metric epoch={best_state['epoch']}, val={best_state['val']}, test={best_state['test']}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
