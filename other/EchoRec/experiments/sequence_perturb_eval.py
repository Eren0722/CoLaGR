import argparse
import csv
import gc
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import build_parser, set_random_seed
from models.echorec_si import EchoRecSIModel
from SeqRec.sasrec.utils import data_partition


def seed_everything(seed: int) -> None:
    set_random_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detect_best_epoch(model_dir: Path, dataset: str, llm: str, subdir: str = "best") -> int:
    target_dir = model_dir / subdir
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Best checkpoint directory not found: {target_dir}")

    pattern = re.compile(rf"^{re.escape(dataset)}_{re.escape(llm)}_(\d+)_item_proj\.pt$")
    epochs: List[int] = []
    for file in target_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            epochs.append(int(match.group(1)))

    if not epochs:
        raise FileNotFoundError(f"No checkpoint files matching {dataset}_{llm}_*_item_proj.pt under {target_dir}")
    return max(epochs)


def right_align_sequence(items: Sequence[int], maxlen: int) -> np.ndarray:
    seq = np.zeros([maxlen], dtype=np.int32)
    items = list(items)[-maxlen:]
    if items:
        seq[-len(items):] = np.array(items, dtype=np.int32)
    return seq


def perturb_history(items: Sequence[int], perturbation: str, seed: int, user_id: int) -> List[int]:
    items = [int(x) for x in items if int(x) > 0]
    if perturbation == "original" or len(items) <= 1:
        return items

    if perturbation == "shuffle":
        rng = random.Random(seed * 1000003 + user_id * 9176 + 11)
        shuffled = items[:]
        rng.shuffle(shuffled)
        return shuffled

    if perturbation == "reverse":
        return list(reversed(items))

    if perturbation == "drop_recent":
        return items[:-1] if len(items) > 1 else items

    if perturbation == "swap_last2":
        swapped = items[:]
        swapped[-2], swapped[-1] = swapped[-1], swapped[-2]
        return swapped

    raise ValueError(f"Unsupported perturbation: {perturbation}")


class PerturbedInferenceDataset(Dataset):
    def __init__(
        self,
        user_train,
        user_valid,
        user_test,
        users: Sequence[int],
        num_item: int,
        max_len: int,
        split: str,
        perturbation: str,
        seed: int,
    ):
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.users = list(users)
        self.num_item = num_item
        self.max_len = max_len
        self.split = split
        self.perturbation = perturbation
        self.seed = seed

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        if self.split == "test":
            visible_history = list(self.user_train[user_id]) + list(self.user_valid[user_id])
            target_item = int(self.user_test[user_id][0])
        elif self.split == "valid":
            visible_history = list(self.user_train[user_id])
            target_item = int(self.user_valid[user_id][0])
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        perturbed = perturb_history(visible_history, self.perturbation, self.seed, user_id)
        seq = right_align_sequence(perturbed, self.max_len)
        neg = np.zeros([1], dtype=np.int32)
        return int(user_id), seq, np.int32(target_item), neg


@dataclass
class ModelSpec:
    label: str
    save_dir: str
    recsys_ckpt_path: str
    epoch: int | None = None


def make_base_args(cli_args) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    args.multi_gpu = False
    args.local_rank = -1
    args.world_size = 1
    args.train = False
    args.extract = False
    args.train_student = False
    args.token = False
    args.nn_parameter = False
    args.use_amp = False
    args.disable_model_saving = True

    args.recsys = cli_args.recsys
    args.rec_pre_trained_data = cli_args.dataset
    args.maxlen = cli_args.maxlen
    args.llm = cli_args.llm
    args.llm_path = cli_args.llm_path
    args.hf_local_only = cli_args.hf_local_only
    args.hf_cache_dir = cli_args.hf_cache_dir or cli_args.llm_path
    args.batch_size_infer = cli_args.batch_size_infer
    args.eval_item_batch = cli_args.eval_item_batch
    args.eval_max_length = cli_args.eval_max_length
    args.eval_min_length = cli_args.eval_min_length
    args.llm_max_length = cli_args.llm_max_length
    args.inference_chunk_size = cli_args.inference_chunk_size
    args.candidate_chunk_size = cli_args.candidate_chunk_size
    args.candidate_chunk_threshold = cli_args.candidate_chunk_threshold
    args.min_candidate_chunk_size = cli_args.min_candidate_chunk_size
    args.train_history_window = cli_args.history_window
    args.student_eval_candidates = cli_args.student_eval_candidates
    args.save_dir = ""
    args.device = torch.device(f"cuda:{cli_args.device}" if torch.cuda.is_available() else "cpu")
    return args


def pick_eval_users(dataset, split: str, max_users: int | None, seed: int) -> List[int]:
    user_train, user_valid, user_test, usernum, itemnum, eval_set = dataset
    if split == "test":
        users = [u for u in eval_set[1] if len(user_test[u]) >= 1]
    else:
        users = [u for u in eval_set[0] if len(user_valid[u]) >= 1]

    users = sorted(users)
    if max_users is not None and len(users) > max_users:
        rng = random.Random(seed)
        users = rng.sample(users, max_users)
        users.sort()
    return users


def evaluate_one_model(
    base_args: argparse.Namespace,
    dataset,
    eval_users: Sequence[int],
    spec: ModelSpec,
    perturbations: Sequence[str],
    split: str,
    seed: int,
) -> List[Dict[str, float]]:
    args = argparse.Namespace(**vars(base_args))
    args.save_dir = spec.save_dir
    args.recsys_ckpt_path = spec.recsys_ckpt_path

    model_dir = Path("./models") / args.rec_pre_trained_data / args.save_dir
    epoch = spec.epoch if spec.epoch is not None else detect_best_epoch(model_dir, args.rec_pre_trained_data, args.llm)

    print(f"\n===== Loading {spec.label} =====")
    print(f"save_dir={args.save_dir}")
    print(f"teacher_ckpt={args.recsys_ckpt_path}")
    print(f"best_epoch={epoch}")

    model = EchoRecSIModel(args).to(args.device)
    model.load_model(args, phase2_epoch=epoch, subdir="best")
    model.eval()

    # Candidate-side item embeddings depend on the loaded SI checkpoint,
    # but not on user-history perturbations. Build them once per model.
    print(f"Precomputing item embeddings once for {spec.label} ...")
    with torch.no_grad():
        model._ensure_item_embeddings_ready(desc=f"Building item embeddings [{spec.label}]")

    results: List[Dict[str, float]] = []
    original_metrics: Dict[str, float] | None = None

    for perturbation in perturbations:
        seed_everything(seed)

        inference_dataset = PerturbedInferenceDataset(
            user_train=dataset[0],
            user_valid=dataset[1],
            user_test=dataset[2],
            users=eval_users,
            num_item=dataset[4],
            max_len=args.maxlen,
            split=split,
            perturbation=perturbation,
            seed=seed,
        )
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size_infer,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
        )

        model.users = 0.0
        model.NDCG = 0.0
        model.HT = 0.0
        model.NDCG_20 = 0.0
        model.HIT_20 = 0.0

        with torch.no_grad():
            for data in inference_loader:
                u, seq, pos, neg = data
                model(
                    [u.numpy(), seq.numpy(), pos.numpy(), neg.numpy(), 0, None, perturbation],
                    mode="generate_batch",
                )

        users = max(float(model.users), 1.0)
        metrics = {
            "model": spec.label,
            "perturbation": perturbation,
            "users": float(model.users),
            "hr10": float(model.HT / users),
            "ndcg10": float(model.NDCG / users),
            "hr20": float(model.HIT_20 / users),
            "ndcg20": float(model.NDCG_20 / users),
        }

        if original_metrics is None:
            original_metrics = metrics.copy()

        metrics["delta_hr10"] = metrics["hr10"] - original_metrics["hr10"]
        metrics["delta_ndcg10"] = metrics["ndcg10"] - original_metrics["ndcg10"]
        metrics["delta_hr20"] = metrics["hr20"] - original_metrics["hr20"]
        metrics["delta_ndcg20"] = metrics["ndcg20"] - original_metrics["ndcg20"]
        results.append(metrics)

        print(
            f"[{spec.label}] {perturbation:<12} "
            f"HR@10={metrics['hr10']:.4f} NDCG@10={metrics['ndcg10']:.4f} "
            f"HR@20={metrics['hr20']:.4f} NDCG@20={metrics['ndcg20']:.4f}"
        )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def write_outputs(output_dir: Path, dataset: str, split: str, all_results: List[Dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{dataset.lower()}_{split}_sequence_perturb_{ts}.csv"
    txt_path = output_dir / f"{dataset.lower()}_{split}_sequence_perturb_{ts}.txt"

    fieldnames = [
        "model",
        "perturbation",
        "users",
        "hr10",
        "ndcg10",
        "hr20",
        "ndcg20",
        "delta_hr10",
        "delta_ndcg10",
        "delta_hr20",
        "delta_ndcg20",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    grouped: Dict[str, List[Dict[str, float]]] = {}
    for row in all_results:
        grouped.setdefault(row["model"], []).append(row)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Sequence perturbation evaluation\n")
        f.write(f"dataset={dataset}, split={split}\n\n")
        for model_name, rows in grouped.items():
            f.write(f"[{model_name}]\n")
            f.write("perturbation      HR@10    dHR@10   NDCG@10  dNDCG@10 HR@20    dHR@20   NDCG@20  dNDCG@20\n")
            for row in rows:
                f.write(
                    f"{row['perturbation']:<15} "
                    f"{row['hr10']:.4f}  {row['delta_hr10']:+.4f}  "
                    f"{row['ndcg10']:.4f}  {row['delta_ndcg10']:+.4f}  "
                    f"{row['hr20']:.4f}  {row['delta_hr20']:+.4f}  "
                    f"{row['ndcg20']:.4f}  {row['delta_ndcg20']:+.4f}\n"
                )
            f.write("\n")

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved TXT: {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser("Sequence perturbation evaluation for two SI checkpoints")
    parser.add_argument("--dataset", type=str, default="Electronics")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--history_window", type=int, default=10)
    parser.add_argument("--batch_size_infer", type=int, default=8)
    parser.add_argument("--eval_item_batch", type=int, default=32)
    parser.add_argument("--eval_max_length", type=int, default=1024)
    parser.add_argument("--eval_min_length", type=int, default=1024)
    parser.add_argument("--llm_max_length", type=int, default=1024)
    parser.add_argument("--inference_chunk_size", type=int, default=8)
    parser.add_argument("--candidate_chunk_size", type=int, default=80)
    parser.add_argument("--candidate_chunk_threshold", type=int, default=36)
    parser.add_argument("--min_candidate_chunk_size", type=int, default=20)
    parser.add_argument("--student_eval_candidates", type=int, default=100)
    parser.add_argument("--recsys", type=str, default="sasrec")
    parser.add_argument("--llm", type=str, default="llama-3b")
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--hf_local_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./analysis/sequence_perturb")

    parser.add_argument("--pure_label", type=str, default="Pure-SI")
    parser.add_argument("--pure_save_dir", type=str, required=True)
    parser.add_argument("--pure_recsys_ckpt", type=str, required=True)
    parser.add_argument("--pure_epoch", type=int, default=None)

    parser.add_argument("--echo_label", type=str, default="EchoRec")
    parser.add_argument("--echo_save_dir", type=str, required=True)
    parser.add_argument("--echo_recsys_ckpt", type=str, required=True)
    parser.add_argument("--echo_epoch", type=int, default=None)

    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["original", "shuffle", "reverse", "drop_recent", "swap_last2"],
        choices=["original", "shuffle", "reverse", "drop_recent", "swap_last2"],
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    seed_everything(cli_args.seed)

    base_args = make_base_args(cli_args)
    dataset = data_partition(
        cli_args.dataset,
        base_args,
        path=f"./SeqRec/data_{cli_args.dataset}/{cli_args.dataset}",
    )
    eval_users = pick_eval_users(dataset, cli_args.split, cli_args.max_users, cli_args.seed)
    print(f"Evaluation users: {len(eval_users)} (split={cli_args.split})")

    specs = [
        ModelSpec(cli_args.pure_label, cli_args.pure_save_dir, cli_args.pure_recsys_ckpt, cli_args.pure_epoch),
        ModelSpec(cli_args.echo_label, cli_args.echo_save_dir, cli_args.echo_recsys_ckpt, cli_args.echo_epoch),
    ]

    all_results: List[Dict[str, float]] = []
    for spec in specs:
        all_results.extend(
            evaluate_one_model(
                base_args=base_args,
                dataset=dataset,
                eval_users=eval_users,
                spec=spec,
                perturbations=cli_args.perturbations,
                split=cli_args.split,
                seed=cli_args.seed,
            )
        )

    write_outputs(Path(cli_args.output_dir) / cli_args.dataset, cli_args.dataset, cli_args.split, all_results)


if __name__ == "__main__":
    main()
