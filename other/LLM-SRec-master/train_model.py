import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models.seqllm_model import *
from SeqRec.sasrec.utils import (
    SeqDataset,
    SeqDataset_Inference,
    SeqDataset_Validation,
    data_partition,
)


def _is_hpu_device(device):
    if isinstance(device, torch.device):
        return device.type == "hpu"
    return str(device) == "hpu"


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _dist_barrier(rank, args):
    if _is_hpu_device(args.device):
        dist.barrier()
    else:
        dist.barrier(device_ids=[rank])


def _reset_eval_state(model):
    model.users = 0.0
    model.NDCG = 0.0
    model.HT = 0.0
    model.NDCG_20 = 0.0
    model.HIT_20 = 0.0
    model.all_embs = None


def _metric_tuple(model):
    if model.users == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        model.NDCG / model.users,
        model.HT / model.users,
        model.NDCG_20 / model.users,
        model.HIT_20 / model.users,
    )


def _sample_eval_users(candidate_users, target_dict, max_users=10000):
    users = list(candidate_users)
    if len(users) > max_users:
        users = random.sample(users, max_users)
    return [u for u in users if len(target_dict[u]) >= 1]


def _build_test_loader(user_train, user_valid, user_test, eval_set, itemnum, args):
    user_list = _sample_eval_users(eval_set[1], user_test)
    dataset = SeqDataset_Inference(
        user_train, user_valid, user_test, user_list, itemnum, args.maxlen
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size_infer,
        pin_memory=True,
        shuffle=False,
    )


def _build_valid_loader(user_train, user_valid, eval_set, itemnum, args):
    user_list = _sample_eval_users(eval_set[0], user_valid)
    dataset = SeqDataset_Validation(
        user_train, user_valid, user_list, itemnum, args.maxlen
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size_infer,
        pin_memory=True,
        shuffle=False,
    )


def _run_eval(model, data_loader, rank):
    eval_model = _unwrap_model(model)
    _reset_eval_state(eval_model)
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u, seq, pos, neg, rank, None, "original"], mode="generate_batch")
    return _metric_tuple(eval_model)


def _write_result_file(args, epoch, metrics):
    out_dir = f"./models/{args.save_dir}/"
    out_dir = out_dir[:-1] + "best/"
    out_dir += f"{args.rec_pre_trained_data}_"
    out_dir += f"{args.llm}_{epoch}_results.txt"

    ndcg10, hr10, ndcg20, hr20 = metrics
    with open(out_dir, "a", encoding="utf-8") as f:
        f.write(f"NDCG: {ndcg10}, HR: {hr10}\n")
        f.write(f"NDCG20: {ndcg20}, HR20: {hr20}\n")


def setup_ddp(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["ID"] = str(rank)

    if _is_hpu_device(args.device):
        import habana_frameworks.torch.distributed.hccl

        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def train_model(args):
    print("LLMRec strat train\n")
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(train_model_, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_model_(0, 0, args)


def inference(args):
    print("LLMRec start inference\n")
    if args.multi_gpu:
        world_size = args.world_size
        mp.spawn(inference_, args=(world_size, args), nprocs=world_size, join=True)
    else:
        inference_(0, 0, args)


def train_model_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size, args)
        if _is_hpu_device(args.device):
            args.device = torch.device("hpu")
        else:
            args.device = f"cuda:{rank}"

    random.seed(0)

    model = llmrec_model(args).to(args.device)

    dataset = data_partition(
        args.rec_pre_trained_data,
        args,
        path=f"./SeqRec/data_{args.rec_pre_trained_data}/{args.rec_pre_trained_data}",
    )
    [user_train, user_valid, user_test, usernum, itemnum, eval_set] = dataset

    if rank == 0:
        print("user num:", usernum, "item num:", itemnum)

    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    if rank == 0:
        print("average sequence length: %.2f" % (cc / len(user_train)))

    train_dataset = SeqDataset(user_train, len(user_train.keys()), itemnum, args.maxlen)

    if args.multi_gpu:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=DistributedSampler(train_dataset, shuffle=True),
            pin_memory=True,
        )
        model = DDP(model, static_graph=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=True,
        )
    num_batch = len(train_loader)

    adam_optimizer = torch.optim.Adam(
        model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98)
    )
    scheduler = LambdaLR(adam_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    epoch_start_idx = 1
    best_perform = 0.0
    early_stop = 0
    early_thres = 5
    t0 = time.time()

    run_eval_on_rank = (not args.multi_gpu) or rank == 0
    inference_loader = None
    if run_eval_on_rank:
        inference_loader = _build_test_loader(
            user_train, user_valid, user_test, eval_set, itemnum, args
        )

    epoch_iter = tqdm(range(epoch_start_idx, args.num_epochs + 1)) if rank == 0 else range(epoch_start_idx, args.num_epochs + 1)
    for epoch in epoch_iter:
        model.train()
        if args.multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        for step, data in enumerate(train_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model(
                [u, seq, pos, neg],
                optimizer=adam_optimizer,
                batch_iter=[epoch, args.num_epochs, step, num_batch],
                mode="phase2",
            )

        scheduler.step()

        if args.multi_gpu:
            _dist_barrier(rank, args)

        stop_training = False
        if run_eval_on_rank:
            if torch.cuda.is_available() and not _is_hpu_device(args.device):
                torch.cuda.empty_cache()
            valid_loader = _build_valid_loader(user_train, user_valid, eval_set, itemnum, args)
            print(f"Validation, early stop: {early_stop}")
            valid_metrics = _run_eval(model, valid_loader, rank)
            print(args.save_dir, args.rec_pre_trained_data)
            print(
                "valid (NDCG@10: %.4f, HR@10: %.4f)"
                % (valid_metrics[0], valid_metrics[1])
            )
            print(
                "valid (NDCG@20: %.4f, HR@20: %.4f)"
                % (valid_metrics[2], valid_metrics[3])
            )

            perform = valid_metrics[1]
            save_model = _unwrap_model(model)

            if perform >= best_perform:
                best_perform = perform
                save_model.save_model(args, epoch2=epoch, best=True)

                print("Testing")
                test_metrics = _run_eval(model, inference_loader, rank)
                print(args.save_dir, args.rec_pre_trained_data)
                print(
                    "test (NDCG@10: %.4f, HR@10: %.4f)"
                    % (test_metrics[0], test_metrics[1])
                )
                print(
                    "test (NDCG@20: %.4f, HR@20: %.4f)"
                    % (test_metrics[2], test_metrics[3])
                )
                _write_result_file(args, epoch, test_metrics)
                early_stop = 0
            else:
                save_model.save_model(args, epoch2=epoch)
                early_stop += 1

            stop_training = early_stop >= early_thres

        if args.multi_gpu:
            stop_tensor = torch.zeros(1, device=args.device)
            if rank == 0:
                stop_tensor.fill_(1 if stop_training else 0)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
            _dist_barrier(rank, args)

        if stop_training:
            if rank == 0:
                print("Terminating Train")
            break

    if rank == 0:
        print("train time :", time.time() - t0)

    if args.multi_gpu:
        dist.destroy_process_group()
