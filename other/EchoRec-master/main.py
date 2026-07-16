import argparse
import os
import random

import numpy as np
import torch

from train_si import train_si


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("EchoRec sequence-injection training")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--local-rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--disable_model_saving", action="store_true", default=False)

    parser.add_argument("--recsys", type=str, default="bert4rec")
    parser.add_argument("--rec_pre_trained_data", type=str, default="CDs_and_Vinyl")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="cds_small_model_si")
    parser.add_argument("--recsys_ckpt_path", type=str, required=True)

    parser.add_argument("--llm", type=str, default="llama-3b")
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--hf_local_only", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--hf_endpoint", type=str, default="")
    parser.add_argument("--hf_mirror_endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--hf_access_token", type=str, default="")
    parser.add_argument("--hf_use_mirror", action="store_true")
    parser.add_argument("--llm_max_length", type=int, default=896)
    parser.add_argument("--token", action="store_true")
    parser.add_argument("--nn_parameter", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--batch_size_infer", type=int, default=20)
    parser.add_argument("--train_num_workers", type=int, default=0)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--num_epochs", type=int, default=25)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--match_weight", type=float, default=1.0)
    parser.add_argument("--train_candidate_num", type=int, default=4)
    parser.add_argument("--candidate_chunk_size", type=int, default=80)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_log_users", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--min_epochs_before_early_stop", type=int, default=10)
    parser.add_argument("--inference_chunk_size", type=int, default=8)
    parser.add_argument("--eval_item_batch", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=0)
    parser.add_argument("--eval_min_length", type=int, default=0)
    parser.add_argument("--candidate_chunk_threshold", type=int, default=36)
    parser.add_argument("--min_candidate_chunk_size", type=int, default=20)
    parser.add_argument("--sequence_chunk_threshold", type=int, default=50)
    parser.add_argument("--sequence_chunk_size", type=int, default=5)
    parser.add_argument("--min_sequence_chunk_size", type=int, default=3)
    return parser


def configure_runtime(args):
    set_random_seed(args.seed)
    if args.multi_gpu:
        os.environ.setdefault("NCCL_TIMEOUT", "1200")
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(f"cuda:{args.local_rank}")
    else:
        args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


def main():
    parser = build_parser()
    args = parser.parse_args()
    configure_runtime(args)
    if not args.train:
        parser.error("Specify --train")
    train_si(args)


if __name__ == "__main__":
    main()
