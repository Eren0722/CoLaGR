import argparse
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from SeqRec.bert4rec.model import BERT4Rec
from SeqRec.gru4rec.model import GRU4Rec
from SeqRec.sasrec.model import SASRec
from SeqRec.sasrec.echorec_sa_backbone import EchoRecSABackbone
from SA import (
    SemanticAlignmentModule,
    SequenceDataset,
    data_augmentation,
    data_partition,
    prepare_semantic_assets,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Semantic alignment pretraining for sequential recommenders")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., Movies_and_TV")
    parser.add_argument("--data_root", type=str, default="./SeqRec", help="Root directory holding data_*/ files")
    parser.add_argument("--sa_asset_root", type=str, default="./SA_assets", help="Semantic asset root")
    parser.add_argument("--sa_item_semantic", type=str, default=None)
    parser.add_argument("--sa_user_semantic", type=str, default=None)
    parser.add_argument("--sa_seq_keys", type=str, default=None)
    parser.add_argument("--sa_item_neighbors", type=str, default=None)
    parser.add_argument("--sa_user_neighbors", type=str, default=None)
    parser.add_argument("--sa_alpha", type=float, default=0.1)
    parser.add_argument("--sa_beta", type=float, default=0.1)
    parser.add_argument("--sa_temperature", type=float, default=1.0)
    parser.add_argument("--sa_mlm_probability", type=float, default=0.2)
    parser.add_argument("--sa_k_num", type=int, default=10)
    parser.add_argument("--bert_mask_prob", type=float, default=0.15)
    parser.add_argument("--bert_rec_objective", choices=["masked", "next_item"], default="masked")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--backbone_learning_rate",
        type=float,
        default=None,
        help="Optional learning rate for backbone parameters. Defaults to --learning_rate.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument(
        "--recsys_backbone",
        choices=["sasrec", "gru4rec", "bert4rec", "echorec_sa", "sa_transformer"],
        default="sasrec",
    )
    parser.add_argument(
        "--gru_output_head",
        choices=["linear", "tied"],
        default="linear",
        help="GRU4Rec scoring head. linear matches the original/IADSR-style GRU item classifier; tied keeps SASRec-style item-embedding dot scores.",
    )
    parser.add_argument("--inner_size", type=int, default=256)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--attn_dropout_prob", type=float, default=0.5)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--device", type=str, default="0", help="cuda index, 'cpu', or 'hpu'")
    parser.add_argument("--nn_parameter", action="store_true", help="Enable nn.Parameter embeddings for HPU")

    parser.add_argument("--save_dir", type=str, default="./SeqRec/sasrec", help="Directory to store checkpoints")
    parser.add_argument("--save_prefix", type=str, default="sa_pretrain", help="Subfolder name under save_dir")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to pretrained teacher checkpoint")
    parser.add_argument(
        "--sa_warmup_epochs",
        type=int,
        default=0,
        help="When loading a pretrained checkpoint, train only the semantic-alignment module for the first N epochs.",
    )
    parser.add_argument("--eval_every", type=int, default=1, help="Run val/test every N epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def _init_distributed_environment(args: argparse.Namespace):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
        return torch.device(f"cuda:{args.local_rank}")
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    args.distributed = False
    return None


def prepare_device(args: argparse.Namespace) -> torch.device:
    raw_device = args.device
    args.is_hpu = False
    distributed_device = _init_distributed_environment(args)
    if distributed_device is not None:
        args.device = distributed_device
        return distributed_device

    if raw_device.lower() == "cpu":
        args.device = "cpu"
    elif raw_device.lower() == "hpu":
        args.device = torch.device("hpu")
        args.nn_parameter = True
        args.is_hpu = True
        try:
            import habana_frameworks.torch.core as htcore  # type: ignore

            args._htcore = htcore
        except ImportError:
            args._htcore = None
        return args.device
    else:
        args.device = f"cuda:{raw_device}"
    args.device = torch.device(args.device)
    return args.device


def _item_embedding_matrix(model: nn.Module) -> torch.Tensor:
    if hasattr(model, "get_item_matrix"):
        return model.get_item_matrix()
    start = int(getattr(model, "real_item_start", 1))
    end = getattr(model, "real_item_end", None)
    if hasattr(model.item_emb, "weight"):
        return model.item_emb.weight[start:end]
    return model.item_emb[start:end]


def compute_rec_loss(
    model: nn.Module,
    seq_batch: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    if isinstance(model, BERT4Rec):
        return model.calculate_loss(seq_batch, targets)
    seq_np = seq_batch.detach().cpu().numpy().astype(np.int32)
    seq_output = model.log2feats(seq_np)[:, -1, :]
    item_matrix = _item_embedding_matrix(model)
    logits = torch.matmul(seq_output, item_matrix.transpose(0, 1))
    labels = targets.to(logits.device) - 1
    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=logits.device)
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    return criterion(logits, labels)




def build_recsys_model(usernum: int, itemnum: int, args):
    backbone = getattr(args, "recsys_backbone", "sasrec")
    if backbone in {"echorec_sa", "sa_transformer"}:
        model_cls = EchoRecSABackbone
    elif backbone == "bert4rec":
        model_cls = BERT4Rec
    elif backbone == "gru4rec":
        model_cls = GRU4Rec
    else:
        model_cls = SASRec
    return model_cls(usernum, itemnum, args)


def data_partition_si(dataset_name, data_root="./SeqRec"):
    base = os.path.join(data_root, f"data_{dataset_name}")
    usernum = 0
    itemnum = 0
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    for split in ["train", "valid", "test"]:
        fpath = os.path.join(base, f"{dataset_name}_{split}.txt")
        with open(fpath, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                u, i = line.rstrip().split(" ")
                u, i = int(u), int(i)
                usernum = max(u, usernum)
                itemnum = max(i, itemnum)
                if split == "train":
                    user_train[u].append(i)
                elif split == "valid":
                    user_valid[u].append(i)
                else:
                    user_test[u].append(i)

    return user_train, user_valid, user_test, usernum, itemnum


def build_seq_for_val(user_train, uid, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    for item in reversed(user_train[uid]):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break
    return seq


def build_seq_for_test(user_train, user_valid, uid, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    idx = maxlen - 1
    try:
        seq[idx] = user_valid[uid][0]
        idx -= 1
    except (KeyError, IndexError):
        pass
    for item in reversed(user_train[uid]):
        seq[idx] = item
        idx -= 1
        if idx == -1:
            break
    return seq


def make_candidate_si(seq, target_id, itemnum, num_neg=99):
    history = set(seq[seq > 0].tolist())
    history.add(0)
    history.add(target_id)

    neg_ids = []
    while len(neg_ids) < num_neg:
        cand = np.random.randint(1, itemnum + 1)
        if cand not in history and cand not in neg_ids:
            neg_ids.append(cand)

    random.shuffle(neg_ids)
    return [target_id] + neg_ids[:num_neg]


def evaluate_sampled_si(
    model,
    user_list,
    user_train,
    user_valid,
    user_test,
    itemnum,
    maxlen,
    split="test",
    top_k=10,
    num_neg=99,
):
    model.eval()
    hr_list, ndcg_list = [], []
    hr20_list, ndcg20_list = [], []

    with torch.no_grad():
        for uid in user_list:
            if split == "test":
                if len(user_test.get(uid, [])) < 1:
                    continue
                seq = build_seq_for_test(user_train, user_valid, uid, maxlen)
                target = user_test[uid][0]
            else:
                if len(user_valid.get(uid, [])) < 1:
                    continue
                seq = build_seq_for_val(user_train, uid, maxlen)
                target = user_valid[uid][0]

            if target <= 0:
                continue

            candidates = make_candidate_si(seq, target, itemnum, num_neg)
            candidates_arr = np.array([candidates], dtype=np.int64)
            user_arr = np.array([uid], dtype=np.int64)
            seq_arr = np.array([seq], dtype=np.int64)

            scores = model.predict(user_arr, seq_arr, candidates_arr)
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            rank = int((scores[0] > scores[0, 0]).sum())

            if rank < top_k:
                hr_list.append(1)
                ndcg_list.append(1.0 / np.log2(rank + 2))
            else:
                hr_list.append(0)
                ndcg_list.append(0.0)

            if rank < 20:
                hr20_list.append(1)
                ndcg20_list.append(1.0 / np.log2(rank + 2))
            else:
                hr20_list.append(0)
                ndcg20_list.append(0.0)

    n = len(hr_list) if hr_list else 1
    return {
        f"HR@{top_k}": round(np.sum(hr_list) / n, 4),
        f"NDCG@{top_k}": round(np.sum(ndcg_list) / n, 4),
        "HR@20": round(np.sum(hr20_list) / n, 4),
        "NDCG@20": round(np.sum(ndcg20_list) / n, 4),
        "users": n,
    }


def run_evaluation(model, si_data, args, split):
    si_user_train, si_user_valid, si_user_test, _, si_itemnum = si_data
    if split == "val":
        user_list = [u for u in si_user_valid if len(si_user_valid[u]) >= 1]
    else:
        user_list = [u for u in si_user_test if len(si_user_test[u]) >= 1]
    return evaluate_sampled_si(
        model,
        user_list,
        si_user_train,
        si_user_valid,
        si_user_test,
        si_itemnum,
        args.maxlen,
        split=split,
        top_k=10,
        num_neg=99,
    )


def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = [model.kwargs, model.state_dict()]
    torch.save(ckpt, path)


def append_eval_results(results_path: str, epoch: int, metrics: Dict[str, float], split: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    metric_str = "\t".join([f"{k}={v:.6f}" for k, v in metrics.items()])
    line = f"epoch={epoch}\t{split}\t{metric_str}\t{ts}\n"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.enable_semantic_module = True
    args.rec_pre_trained_data = args.dataset

    device = prepare_device(args)
    if not hasattr(args, "_htcore"):
        args._htcore = None

    dataset = data_partition(args.dataset, root_dir=args.data_root)
    user_train, user_valid, user_test, usernum, itemnum, _ = dataset

    max_item = itemnum
    for seqs in (user_valid, user_test):
        for seq in seqs.values():
            if isinstance(seq, (list, tuple)) and len(seq) > 0:
                max_item = max(max_item, max(seq))
    itemnum = max_item

    prepare_semantic_assets(args, user_train)
    args.item_num = itemnum

    uid_list, item_list, target_list, item_list_length, seq_unique_key_list = data_augmentation(
        user_train, args.maxlen, args.seq_keys_to_int
    )

    train_dataset = SequenceDataset(
        args,
        uid_list,
        item_list,
        target_list,
        item_list_length,
        seq_unique_key_list,
        args.maxlen,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    si_data = data_partition_si(args.dataset, data_root=args.data_root)
    if args.rank == 0:
        _, si_uv, si_ute, si_un, si_in = si_data
        print(
            f"SI eval data loaded: usernum={si_un}, itemnum={si_in}, "
            f"val_users={sum(1 for u in si_uv if len(si_uv[u]) >= 1)}, "
            f"test_users={sum(1 for u in si_ute if len(si_ute[u]) >= 1)}"
        )

    model = build_recsys_model(usernum, itemnum, args).to(device)
    core_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    if args.pretrained_ckpt is not None:
        if args.rank == 0:
            print(f"Loading pretrained checkpoint from {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
        if isinstance(ckpt, (list, tuple)) and len(ckpt) == 2:
            _, state_dict = ckpt
        elif isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        missing, unexpected = core_model.load_state_dict(state_dict, strict=False)
        if args.rank == 0:
            if missing:
                print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
            if unexpected:
                print(
                    f"  Unexpected keys: {unexpected[:5]}..."
                    if len(unexpected) > 5
                    else f"  Unexpected keys: {unexpected}"
                )
            print("  Loaded pretrained weights successfully")

    alignment_module = SemanticAlignmentModule(
        args,
        core_model,
        detach_recsys_grad=False,
    ).to(device)

    alignment_params = [p for p in alignment_module.parameters() if p.requires_grad]
    if args.distributed and alignment_params:
        for param in alignment_params:
            dist.broadcast(param.data, src=0)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    core_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    backbone_lr = args.learning_rate if args.backbone_learning_rate is None else args.backbone_learning_rate
    optimizer_param_groups = [{"params": list(model.parameters()), "lr": backbone_lr}]
    if alignment_params:
        optimizer_param_groups.append({"params": alignment_params, "lr": args.learning_rate})
    optimizer = torch.optim.Adam(optimizer_param_groups, weight_decay=args.weight_decay)
    ce_criterion = torch.nn.CrossEntropyLoss()

    best_metric = -1.0
    best_loss = float("inf")
    best_loss_epoch = 1
    patience_counter = 0
    best_state: Dict[str, Any] = {}
    target_dir = os.path.join(args.save_dir, args.dataset, args.save_prefix)
    results_val_path = os.path.join(target_dir, "results_val.txt")
    results_test_path = os.path.join(target_dir, "results_test.txt")

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        freeze_backbone = bool(
            args.pretrained_ckpt
            and args.sa_warmup_epochs > 0
            and ((args.sa_alpha > 0) or (args.sa_beta > 0))
            and epoch <= args.sa_warmup_epochs
        )
        for param in model.parameters():
            param.requires_grad_(not freeze_backbone)

        epoch_rec = 0.0
        epoch_sa = 0.0
        epoch_item_cl = 0.0
        epoch_user_cl = 0.0
        epoch_total = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.num_epochs}",
            disable=args.distributed and args.rank != 0,
        )

        for batch in pbar:
            seq_unique_id, padded_seq, target, _, aug_seq1, aug_seq2, neighbor_seqs, _ = batch
            optimizer.zero_grad()

            rec_loss = compute_rec_loss(core_model, padded_seq, target, ce_criterion)
            sa_loss, sa_info = alignment_module.compute_batch_losses(
                seq_unique_id,
                padded_seq,
                aug_seq1,
                aug_seq2,
                neighbor_seqs,
                allow_recsys_grad=not freeze_backbone,
            )
            loss = sa_loss if freeze_backbone else (rec_loss + sa_loss)
            loss.backward()

            if args.distributed and alignment_params:
                for param in alignment_params:
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= args.world_size

            optimizer.step()
            if getattr(args, "is_hpu", False) and args._htcore is not None:
                args._htcore.mark_step()

            epoch_rec += rec_loss.item()
            epoch_sa += sa_loss.item()
            epoch_item_cl += float(sa_info.get("item_cl_loss", 0.0))
            epoch_user_cl += float(sa_info.get("user_cl_loss", 0.0))
            epoch_total += loss.item()
            pbar.set_postfix({"rec": f"{rec_loss.item():.3f}", "sa": f"{sa_loss.item():.3f}"})

        avg_rec = epoch_rec / max(len(train_loader), 1)
        avg_sa = epoch_sa / max(len(train_loader), 1)
        avg_item_cl = epoch_item_cl / max(len(train_loader), 1)
        avg_user_cl = epoch_user_cl / max(len(train_loader), 1)
        avg_total = epoch_total / max(len(train_loader), 1)
        if args.rank == 0:
            print(
                f"Epoch [{epoch}/{args.num_epochs}] Total: {avg_total:.4f} | "
                f"SA_rec: {avg_rec:.4f} | SA_sem: {avg_sa:.4f} | "
                f"item_cl: {avg_item_cl:.4f} | user_cl: {avg_user_cl:.4f} | "
                f"freeze_backbone={freeze_backbone}"
            )

        is_last_epoch = epoch == args.num_epochs
        if avg_total < best_loss:
            best_loss = avg_total
            best_loss_epoch = epoch
            if (not args.distributed) or args.rank == 0:
                best_path = os.path.join(target_dir, "model_best.pth")
                save_checkpoint(core_model, best_path)
                if args.rank == 0:
                    print(f"  Best loss model saved (loss={best_loss:.4f}) -> {best_path}")

        if is_last_epoch and ((not args.distributed) or args.rank == 0):
            final_path = os.path.join(target_dir, "model.pth")
            if os.path.exists(final_path):
                os.remove(final_path)
            save_checkpoint(core_model, final_path)
            if args.rank == 0:
                print(f"  Final model saved at epoch {epoch} -> {final_path}")
                print(f"    (best_loss epoch was {best_loss_epoch}, loss={best_loss:.4f})")

        do_eval = args.eval_every > 0 and ((epoch % args.eval_every == 0) or is_last_epoch)
        run_eval = do_eval and ((not args.distributed) or args.rank == 0)

        if run_eval:
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(42)
            np.random.seed(42)

            val_metrics = run_evaluation(core_model, si_data, args, split="val")
            test_metrics = run_evaluation(core_model, si_data, args, split="test")

            random.setstate(py_state)
            np.random.set_state(np_state)

            if args.rank == 0:
                print(f"  Val:  {val_metrics}")
                print(f"  Test: {test_metrics}")
            append_eval_results(results_val_path, epoch, val_metrics, split="val")
            append_eval_results(results_test_path, epoch, test_metrics, split="test")

            current_metric = val_metrics.get("NDCG@10", 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                best_state = {"epoch": epoch, "val": val_metrics, "test": test_metrics}
                if (not args.distributed) or args.rank == 0:
                    best_path = os.path.join(target_dir, "model_metric_best.pth")
                    save_checkpoint(core_model, best_path)
                    if args.rank == 0:
                        print(f"  Best metric model saved (val NDCG@10={current_metric:.4f}) -> {best_path}")
            else:
                patience_counter += 1
                if args.rank == 0:
                    print(f"  metric patience {patience_counter}/{args.patience} (best NDCG@10={best_metric:.4f})")

        should_stop = do_eval and patience_counter >= args.patience
        if args.distributed and do_eval:
            stop_tensor = torch.zeros(1, device=args.device, dtype=torch.int32)
            if args.rank == 0:
                stop_tensor.fill_(1 if should_stop else 0)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if args.distributed:
            dist.barrier()

        if should_stop:
            if args.rank == 0:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            if (not args.distributed) or args.rank == 0:
                es_path = os.path.join(target_dir, "model.pth")
                if os.path.exists(es_path):
                    os.remove(es_path)
                save_checkpoint(core_model, es_path)
                if args.rank == 0:
                    print(f"  Early-stopped model saved at epoch {epoch} -> {es_path}")
            break

    if args.distributed:
        dist.barrier()

    if args.rank == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Training finished. Final model saved at {target_dir}/model.pth")
        print(f"  Best loss: epoch {best_loss_epoch}, loss={best_loss:.4f}")
        if best_state:
            print(f"  Best metric: epoch {best_state['epoch']}, val={best_state.get('val')}, test={best_state.get('test')}")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
