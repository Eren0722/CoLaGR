import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from SA import SequenceDataset, SemanticAlignmentModule, data_augmentation, data_partition, prepare_semantic_assets
from SA.module import info_nce
from SeqRec.sasrec.train_sa import build_recsys_model
from SeqRec.sasrec.train_utils import (
    PrefixDataset,
    SASRecBatchSampler,
    append_eval_results,
    build_prefix_examples,
    data_partition_si,
    prepare_device,
    run_evaluation,
    sasrec_bce_loss,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("One-stage raw sequential recommender + SACP training")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Movies_and_TV")
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
    parser.add_argument("--sa_use_projection_head", action="store_true")
    parser.add_argument("--sa_proj_hidden_dim", type=int, default=0)
    parser.add_argument("--sa_proj_act", choices=["gelu", "relu"], default="gelu")
    parser.add_argument("--sa_contrast_norm", action="store_true")
    parser.add_argument("--sa_similarity", choices=["dot", "cos"], default="dot")
    parser.add_argument(
        "--sa_repr_mode",
        choices=["mean"],
        default="mean",
        help="Sequence-level pooling used by SACP. last pooling has been removed; mean pooling is the supported path.",
    )
    parser.add_argument(
        "--sacp_preset",
        choices=["raw_sasrec"],
        default="raw_sasrec",
        help="Compatibility flag. Only the raw_sasrec teacher path is supported.",
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument(
        "--backbone_learning_rate",
        type=float,
        default=None,
        help="Optional learning rate for backbone parameters. Defaults to --learning_rate.",
    )
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--inner_size", type=int, default=256)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--attn_dropout_prob", type=float, default=0.2)
    parser.add_argument("--hidden_act", type=str, default="gelu")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--recsys_backbone", choices=["sasrec", "gru4rec", "bert4rec"], default="sasrec")
    parser.add_argument(
        "--gru_output_head",
        choices=["linear", "tied"],
        default="linear",
        help="GRU4Rec scoring head. linear matches the original/IADSR-style GRU item classifier; tied keeps SASRec-style item-embedding dot scores.",
    )
    parser.add_argument(
        "--gru_hidden_units",
        type=int,
        default=0,
        help="Optional GRU hidden size before the dense projection. 0 means reuse --hidden_units.",
    )
    parser.add_argument(
        "--gru_sacp_layer_norm",
        dest="gru_sacp_layer_norm",
        action="store_true",
        help="Apply a LayerNorm mapper to GRU features before external SACP, matching SASRec/EchoRecBackbone feature geometry.",
    )
    parser.add_argument(
        "--gru_no_sacp_layer_norm",
        dest="gru_sacp_layer_norm",
        action="store_false",
        help="Disable the GRU SACP LayerNorm mapper.",
    )
    parser.set_defaults(gru_sacp_layer_norm=True)
    parser.add_argument(
        "--gru_output_layer_norm",
        dest="gru_output_layer_norm",
        action="store_true",
        help="Apply LayerNorm inside GRU log2feats so rec, eval, and SACP share the same normalized sequence space.",
    )
    parser.add_argument(
        "--gru_no_output_layer_norm",
        dest="gru_output_layer_norm",
        action="store_false",
        help="Disable GRU output LayerNorm.",
    )
    parser.set_defaults(gru_output_layer_norm=False)
    parser.add_argument(
        "--gru_enable_user_cl",
        dest="gru_enable_user_cl",
        action="store_true",
        help="Enable user-neighbor CL for GRU4Rec SACP. GRU4Rec now uses the same external SACP module as SASRec.",
    )
    parser.add_argument(
        "--gru_disable_user_cl",
        dest="gru_enable_user_cl",
        action="store_false",
        help="Disable user-neighbor CL for GRU4Rec SACP ablation.",
    )
    parser.set_defaults(gru_enable_user_cl=True)
    parser.add_argument(
        "--gru_user_cl_objective",
        choices=["single", "symmetric"],
        default="single",
        help="Deprecated compatibility flag. External SACP uses the same symmetric InfoNCE path as SASRec.",
    )
    parser.add_argument(
        "--gru_user_cl_similarity",
        choices=["cos", "dot"],
        default="cos",
        help="Similarity used by the GRU external SACP user branch. Item-CL still follows --sa_similarity.",
    )
    parser.add_argument(
        "--gru_user_cl_temperature",
        type=float,
        default=0.2,
        help="Temperature used only by the GRU external SACP user branch. Lower values sharpen cosine InfoNCE.",
    )
    parser.add_argument(
        "--gru_user_cl_post_norm",
        dest="gru_user_cl_post_norm",
        action="store_true",
        help="Normalize GRU user-CL anchor and semantic-neighbor aggregate after neighbor pooling.",
    )
    parser.add_argument(
        "--gru_no_user_cl_post_norm",
        dest="gru_user_cl_post_norm",
        action="store_false",
        help="Disable post-pooling normalization for GRU user-CL.",
    )
    parser.set_defaults(gru_user_cl_post_norm=True)
    parser.add_argument(
        "--gru_user_cl_detach_neighbors",
        dest="gru_user_cl_detach_neighbors",
        action="store_true",
        help="Deprecated compatibility flag. External SACP follows SASRec and does not detach neighbor features.",
    )
    parser.add_argument(
        "--gru_user_cl_train_neighbors",
        dest="gru_user_cl_detach_neighbors",
        action="store_false",
        help="Deprecated compatibility flag. External SACP follows SASRec and trains through neighbor features.",
    )
    parser.set_defaults(gru_user_cl_detach_neighbors=True)
    parser.add_argument(
        "--rec_objective",
        choices=["bce", "ce"],
        default="bce",
        help="SASRec supports sampled BCE or prefix next-item CE. GRU4Rec/BERT4Rec(next_item) always use CE-style next-item supervision.",
    )
    parser.add_argument(
        "--bce_mode",
        choices=["sampler"],
        default="sampler",
        help="Compatibility flag. One-stage raw SACP uses the original sampler-style BCE path.",
    )
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="0 means auto infer")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="0", help="cuda index, 'cpu', or 'hpu'")
    parser.add_argument("--nn_parameter", action="store_true")

    parser.add_argument("--save_dir", type=str, default="./SeqRec/sacp_sasrec")
    parser.add_argument("--save_prefix", type=str, default="sacp_sasrec")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def use_raw_sasrec_training_prior(args: argparse.Namespace) -> bool:
    return getattr(args, "recsys_backbone", "sasrec") == "sasrec"


def use_gru4rec_native_objective(args: argparse.Namespace) -> bool:
    return getattr(args, "recsys_backbone", "sasrec") == "gru4rec"


def use_sasrec_next_item_objective(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "recsys_backbone", "sasrec") == "sasrec"
        and getattr(args, "rec_objective", "bce") == "ce"
    )


def use_bert4rec_masked_objective(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "recsys_backbone", "sasrec") == "bert4rec"
        and getattr(args, "bert_rec_objective", "masked") == "masked"
    )


def use_bert4rec_next_item_objective(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "recsys_backbone", "sasrec") == "bert4rec"
        and getattr(args, "bert_rec_objective", "masked") == "next_item"
    )


def use_native_ce_objective(args: argparse.Namespace) -> bool:
    return (
        use_sasrec_next_item_objective(args)
        or use_gru4rec_native_objective(args)
        or use_bert4rec_next_item_objective(args)
    )


def describe_rec_objective(args: argparse.Namespace) -> str:
    if use_sasrec_next_item_objective(args):
        return "sasrec_next_item_ce"
    if use_bert4rec_masked_objective(args):
        return "bert_masked_ce"
    if use_bert4rec_next_item_objective(args):
        return "bert_next_item_ce"
    if use_native_ce_objective(args):
        return "native_ce"
    return args.rec_objective


def infer_raw_sasrec_reference_steps(user_train, batch_size: int, maxlen: int) -> int:
    """Match raw-SASRec teacher epoch length to the prefix-example count used by SACP mode."""
    uid_list, _, _, _ = build_prefix_examples(user_train, maxlen)
    return max((len(uid_list) + batch_size - 1) // batch_size, 1)


def maybe_init_raw_sasrec_like_main(model: torch.nn.Module, args: argparse.Namespace) -> None:
    """Match the original SASRec main.py initialization for raw SASRec training."""
    if not use_raw_sasrec_training_prior(args):
        return

    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    print("Applied original SASRec Xavier re-initialization before one-stage training.")


def _item_embedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "get_item_matrix"):
        return model.get_item_matrix()
    start = int(getattr(model, "real_item_start", 1))
    end = getattr(model, "real_item_end", None)
    if hasattr(model.item_emb, "weight"):
        return model.item_emb.weight[start:end]
    return model.item_emb[start:end]


def native_ce_loss(
    model: torch.nn.Module,
    seq_batch: torch.Tensor,
    target_batch: torch.Tensor,
    precomputed_final_feat: torch.Tensor | None = None,
) -> torch.Tensor:
    if precomputed_final_feat is None and hasattr(model, "calculate_loss"):
        return model.calculate_loss(seq_batch, target_batch)

    if precomputed_final_feat is None:
        seq_output = model.log2feats(seq_batch)[:, -1, :]
    else:
        seq_output = precomputed_final_feat
    item_matrix = _item_embedding_matrix(model)
    logits = torch.matmul(seq_output, item_matrix.transpose(0, 1))
    labels = target_batch.to(logits.device, dtype=torch.long) - 1
    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return torch.tensor(0.0, device=logits.device)
    return torch.nn.functional.cross_entropy(logits[valid_mask], labels[valid_mask])


class FullSequenceDataset(Dataset):
    """User-level full-sequence dataset for masked BERT4Rec training."""

    def __init__(self, user_train, maxlen: int):
        self.maxlen = int(maxlen)
        self.examples = []
        for uid, seq in user_train.items():
            if len(seq) < 1:
                continue
            padded_seq = np.zeros(self.maxlen, dtype=np.int32)
            clipped_seq = seq[-self.maxlen :]
            padded_seq[-len(clipped_seq) :] = clipped_seq
            self.examples.append(
                (
                    int(uid),
                    torch.tensor(padded_seq, dtype=torch.long),
                    torch.tensor(len(clipped_seq), dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def build_alignment_module(args: argparse.Namespace, model: torch.nn.Module) -> SemanticAlignmentModule:
    return SemanticAlignmentModule(
        args,
        model,
        feature_mapper=None,
        detach_recsys_grad=False,
    ).to(args.device)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = prepare_device(args)
    args.rec_pre_trained_data = args.dataset
    args.enable_semantic_module = (args.sa_alpha > 0) or (args.sa_beta > 0)
    train_data = data_partition(args.dataset, root_dir=args.data_root)
    user_train, _, _, usernum, itemnum, _ = train_data
    si_data = data_partition_si(args.dataset, data_root=args.data_root)
    _, si_user_valid, si_user_test, si_usernum, si_itemnum = si_data

    if itemnum <= 0 or usernum <= 0:
        raise RuntimeError(f"Invalid dataset stats: usernum={usernum}, itemnum={itemnum}")

    if args.enable_semantic_module:
        prepare_semantic_assets(args, user_train)

    model = build_recsys_model(usernum, itemnum, args).to(device)
    maybe_init_raw_sasrec_like_main(model, args)

    alignment_module = None
    semantic_loader = None
    semantic_iter = None
    bert_loader = None
    bert_iter = None
    prefix_loader = None
    prefix_iter = None

    if args.enable_semantic_module:
        uid_list, item_list, target_list, item_list_length, seq_unique_key_list = data_augmentation(
            user_train,
            args.maxlen,
            args.seq_keys_to_int,
        )
        semantic_dataset = SequenceDataset(
            args,
            uid_list,
            item_list,
            target_list,
            item_list_length,
            seq_unique_key_list,
            args.maxlen,
        )
        semantic_loader = DataLoader(
            semantic_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        alignment_module = build_alignment_module(args, model)
    elif use_bert4rec_masked_objective(args):
        bert_dataset = FullSequenceDataset(user_train, args.maxlen)
        bert_loader = DataLoader(
            bert_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
    elif use_native_ce_objective(args):
        uid_list, item_list, target_list, item_list_length = build_prefix_examples(user_train, args.maxlen)
        prefix_dataset = PrefixDataset(
            uid_list,
            item_list,
            target_list,
            item_list_length,
            args.maxlen,
        )
        prefix_loader = DataLoader(
            prefix_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

    backbone_lr = args.learning_rate if args.backbone_learning_rate is None else args.backbone_learning_rate
    optimizer_groups = [{"params": list(model.parameters()), "lr": backbone_lr}]
    if alignment_module is not None:
        alignment_params = [p for p in alignment_module.parameters() if p.requires_grad]
        if alignment_params:
            optimizer_groups.append({"params": alignment_params, "lr": args.learning_rate})
    adam_betas = (0.9, 0.98) if use_raw_sasrec_training_prior(args) else (0.9, 0.999)
    optimizer = torch.optim.Adam(
        optimizer_groups,
        betas=adam_betas,
        weight_decay=args.weight_decay,
    )
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    sampler = None
    if (not use_native_ce_objective(args)) and (not use_bert4rec_masked_objective(args)):
        sampler = SASRecBatchSampler(
            user_train=user_train,
            usernum=usernum,
            itemnum=itemnum,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
        )

    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch
    elif semantic_loader is not None:
        steps_per_epoch = max(len(semantic_loader), 1)
    elif bert_loader is not None:
        steps_per_epoch = max(len(bert_loader), 1)
    elif prefix_loader is not None:
        steps_per_epoch = max(len(prefix_loader), 1)
    elif args.recsys_backbone == "sasrec" and args.sacp_preset == "raw_sasrec":
        steps_per_epoch = infer_raw_sasrec_reference_steps(user_train, args.batch_size, args.maxlen)
    else:
        steps_per_epoch = max(len(user_train) // args.batch_size, 1)

    target_dir = os.path.join(args.save_dir, args.dataset, args.save_prefix)
    results_val_path = os.path.join(target_dir, "results_val.txt")
    results_test_path = os.path.join(target_dir, "results_test.txt")

    best_metric = -1.0
    best_loss = float("inf")
    best_loss_epoch = 1
    best_state: Dict[str, Any] = {}
    patience_counter = 0

    if args.enable_semantic_module:
        effective_repr = args.sa_repr_mode
        if alignment_module is not None and hasattr(alignment_module, "_fusion_info"):
            effective_repr = alignment_module._fusion_info.get("repr_mode", effective_repr)
        print(
            f"One-stage raw {args.recsys_backbone} + SACP enabled: alpha={args.sa_alpha}, beta={args.sa_beta}, "
            f"temperature={args.sa_temperature}, mlm={args.sa_mlm_probability}, steps={steps_per_epoch}, "
            f"rec_objective={describe_rec_objective(args)}, "
            f"preset={args.sacp_preset}, backbone={args.recsys_backbone}, "
            f"sacp_impl=external_semantic_module, "
            f"heads={args.num_heads}, dropout={args.dropout_rate}, "
            f"repr={effective_repr}, bce_mode={args.bce_mode}, "
            f"proj_head={bool(args.sa_use_projection_head)}, similarity={args.sa_similarity}, "
            f"contrast_norm={bool(args.sa_contrast_norm)}, "
            f"gru_output_head={getattr(args, 'gru_output_head', 'n/a')}, "
            f"gru_hidden_units={getattr(args, 'gru_hidden_units', 0) or args.hidden_units}, "
            f"gru_sacp_layer_norm={bool(getattr(args, 'gru_sacp_layer_norm', True))}, "
            f"gru_output_layer_norm={bool(getattr(args, 'gru_output_layer_norm', False))}, "
            f"gru_user_cl={bool(getattr(args, 'gru_enable_user_cl', False))}, "
            f"gru_user_cl_similarity={getattr(args, 'gru_user_cl_similarity', 'cos')}, "
            f"gru_user_cl_temperature={float(getattr(args, 'gru_user_cl_temperature', 0.2))}, "
            f"gru_user_cl_post_norm={bool(getattr(args, 'gru_user_cl_post_norm', True))}"
        )
    else:
        print(
            f"Semantic mode disabled: running raw {args.recsys_backbone} baseline "
            f"(steps={steps_per_epoch}, "
            f"rec_objective={describe_rec_objective(args)})"
        )

    print(
        f"SI eval data loaded: usernum={si_usernum}, itemnum={si_itemnum}, "
        f"val_users={sum(1 for u in si_user_valid if len(si_user_valid[u]) >= 1)}, "
        f"test_users={sum(1 for u in si_user_test if len(si_user_test[u]) >= 1)}"
    )

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if alignment_module is not None:
            alignment_module.train()
            semantic_iter = iter(semantic_loader)
        if bert_loader is not None:
            bert_iter = iter(bert_loader)
        if prefix_loader is not None:
            prefix_iter = iter(prefix_loader)

        epoch_rec = 0.0
        epoch_sa = 0.0
        epoch_item_cl = 0.0
        epoch_user_cl = 0.0
        epoch_total = 0.0
        last_sa_status = "disabled"

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.num_epochs}")
        for _ in pbar:
            optimizer.zero_grad()
            rec_loss = torch.tensor(0.0, device=device)

            sa_loss = torch.tensor(0.0, device=device)
            item_cl_loss = 0.0
            user_cl_loss = 0.0
            if alignment_module is not None:
                try:
                    sa_batch = next(semantic_iter)
                except StopIteration:
                    semantic_iter = iter(semantic_loader)
                    sa_batch = next(semantic_iter)

                seq_unique_id, padded_seq, _, _, aug_seq1, aug_seq2, neighbor_seqs, _ = sa_batch
                if use_gru4rec_native_objective(args):
                    target_batch = sa_batch[2]
                    base_seq_feats = model.log2feats(padded_seq)
                    rec_loss = native_ce_loss(model, padded_seq, target_batch, precomputed_final_feat=base_seq_feats[:, -1, :])
                elif use_sasrec_next_item_objective(args):
                    target_batch = sa_batch[2]
                    rec_loss = native_ce_loss(model, padded_seq, target_batch)
                elif use_bert4rec_next_item_objective(args):
                    target_batch = sa_batch[2]
                    rec_loss = model.calculate_loss(padded_seq, target_batch)
                elif use_bert4rec_masked_objective(args):
                    rec_loss = model.calculate_loss(padded_seq, None)
                else:
                    rec_user_ids, rec_seq_batch, rec_pos_batch, rec_neg_batch = sampler.next_batch()
                    rec_loss = sasrec_bce_loss(
                        model=model,
                        user_ids=rec_user_ids,
                        seq_batch=rec_seq_batch,
                        pos_batch=rec_pos_batch,
                        neg_batch=rec_neg_batch,
                        device=device,
                        bce_criterion=bce_criterion,
                        l2_emb=args.l2_emb,
                    )
                sa_loss, sa_info = alignment_module.compute_batch_losses(
                    seq_unique_ids=seq_unique_id,
                    padded_seq=padded_seq,
                    aug_seq1=aug_seq1,
                    aug_seq2=aug_seq2,
                    neighbor_seqs=neighbor_seqs,
                    allow_recsys_grad=True,
                )
                item_cl_loss = float(sa_info.get("item_cl_loss", 0.0))
                user_cl_loss = float(sa_info.get("user_cl_loss", 0.0))
                last_sa_status = str(sa_info.get("sa_status", "enabled"))
            elif use_bert4rec_masked_objective(args):
                try:
                    _, rec_padded_seq, _ = next(bert_iter)
                except StopIteration:
                    bert_iter = iter(bert_loader)
                    _, rec_padded_seq, _ = next(bert_iter)
                rec_loss = model.calculate_loss(rec_padded_seq, None)
            elif use_native_ce_objective(args):
                try:
                    _, rec_padded_seq, rec_target_batch, _ = next(prefix_iter)
                except StopIteration:
                    prefix_iter = iter(prefix_loader)
                    _, rec_padded_seq, rec_target_batch, _ = next(prefix_iter)
                rec_loss = native_ce_loss(model, rec_padded_seq, rec_target_batch)
            else:
                rec_user_ids, rec_seq_batch, rec_pos_batch, rec_neg_batch = sampler.next_batch()
                rec_loss = sasrec_bce_loss(
                    model=model,
                    user_ids=rec_user_ids,
                    seq_batch=rec_seq_batch,
                    pos_batch=rec_pos_batch,
                    neg_batch=rec_neg_batch,
                    device=device,
                    bce_criterion=bce_criterion,
                    l2_emb=args.l2_emb,
                )
            loss = rec_loss + sa_loss
            loss.backward()
            optimizer.step()
            if getattr(args, "is_hpu", False) and getattr(args, "_htcore", None) is not None:
                args._htcore.mark_step()

            epoch_rec += rec_loss.item()
            epoch_sa += sa_loss.item()
            epoch_item_cl += item_cl_loss
            epoch_user_cl += user_cl_loss
            epoch_total += loss.item()
            pbar.set_postfix({"rec": f"{rec_loss.item():.3f}", "sa": f"{sa_loss.item():.3f}"})

        avg_rec = epoch_rec / max(steps_per_epoch, 1)
        avg_sa = epoch_sa / max(steps_per_epoch, 1)
        avg_item_cl = epoch_item_cl / max(steps_per_epoch, 1)
        avg_user_cl = epoch_user_cl / max(steps_per_epoch, 1)
        avg_total = epoch_total / max(steps_per_epoch, 1)

        if alignment_module is not None:
            print(
                f"Epoch [{epoch}/{args.num_epochs}] Total: {avg_total:.4f} | "
                f"SA_rec: {avg_rec:.4f} | SA_sem: {avg_sa:.4f} | "
                f"item_cl: {avg_item_cl:.4f} | user_cl: {avg_user_cl:.4f} | "
                f"sa_mode={last_sa_status}"
            )
        else:
            print(
                f"Epoch [{epoch}/{args.num_epochs}] Total: {avg_total:.4f} | "
                f"SA_rec: {avg_rec:.4f} | SA_sem: {avg_sa:.4f}"
            )

        if avg_total < best_loss:
            best_loss = avg_total
            best_loss_epoch = epoch
            best_path = os.path.join(target_dir, "model_best.pth")
            save_checkpoint(model, best_path)
            print(f"  Best loss model saved (loss={best_loss:.4f}) -> {best_path}")

        is_last_epoch = epoch == args.num_epochs
        do_eval = args.eval_every > 0 and ((epoch % args.eval_every == 0) or is_last_epoch)
        if do_eval:
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(42)
            np.random.seed(42)

            val_metrics = run_evaluation(model, si_data, args, split="val")
            test_metrics = run_evaluation(model, si_data, args, split="test")

            random.setstate(py_state)
            np.random.set_state(np_state)

            print(f"  Val:  {val_metrics}")
            print(f"  Test: {test_metrics}")
            append_eval_results(results_val_path, epoch, val_metrics, split="val")
            append_eval_results(results_test_path, epoch, test_metrics, split="test")

            current_metric = val_metrics.get("NDCG@10", 0.0)
            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                best_state = {"epoch": epoch, "val": val_metrics, "test": test_metrics}
                best_path = os.path.join(target_dir, "model_metric_best.pth")
                save_checkpoint(model, best_path)
                print(f"  Best metric model saved (val NDCG@10={current_metric:.4f}) -> {best_path}")
            else:
                patience_counter += 1
                print(f"  metric patience {patience_counter}/{args.patience} (best NDCG@10={best_metric:.4f})")

        if do_eval and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            final_path = os.path.join(target_dir, "model.pth")
            save_checkpoint(model, final_path)
            print(f"  Early-stopped model saved at epoch {epoch} -> {final_path}")
            break

        if is_last_epoch:
            final_path = os.path.join(target_dir, "model.pth")
            save_checkpoint(model, final_path)
            print(f"  Final model saved at epoch {epoch} -> {final_path}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Training finished. Final model saved under {target_dir}")
    print(f"  Best loss: epoch {best_loss_epoch}, loss={best_loss:.4f}")
    if best_state:
        print(f"  Best metric: epoch {best_state['epoch']}, val={best_state['val']}, test={best_state['test']}")


if __name__ == "__main__":
    main()
