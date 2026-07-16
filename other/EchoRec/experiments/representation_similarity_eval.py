import argparse
import csv
import gc
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
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
        seq[-len(items):] = np.asarray(items, dtype=np.int32)
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


class PrefixDataset(Dataset):
    def __init__(self, user_ids: np.ndarray, seqs: np.ndarray):
        self.user_ids = user_ids.astype(np.int64)
        self.seqs = seqs.astype(np.int32)

    def __len__(self) -> int:
        return int(self.user_ids.shape[0])

    def __getitem__(self, idx: int):
        user_id = np.int64(self.user_ids[idx])
        seq = self.seqs[idx]
        pos = np.zeros([1], dtype=np.int32)
        neg = np.zeros([1], dtype=np.int32)
        return user_id, seq, pos, neg


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
    user_train, user_valid, user_test, _, _, eval_set = dataset
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


def build_seq_arrays(dataset, users: Sequence[int], split: str, perturbation: str, maxlen: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    user_train, user_valid, user_test = dataset[0], dataset[1], dataset[2]
    user_ids: List[int] = []
    seq_arrays: List[np.ndarray] = []

    for user_id in users:
        if split == "test":
            visible_history = list(user_train[user_id]) + list(user_valid[user_id])
        else:
            visible_history = list(user_train[user_id])
        perturbed = perturb_history(visible_history, perturbation, seed, int(user_id))
        user_ids.append(int(user_id))
        seq_arrays.append(right_align_sequence(perturbed, maxlen))

    return np.asarray(user_ids, dtype=np.int64), np.asarray(seq_arrays, dtype=np.int32)


def extract_llm_embeddings(model: EchoRecSIModel, user_ids: np.ndarray, seq_arrays: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(
        PrefixDataset(user_ids, seq_arrays),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    model.extract_embs_list = []
    with torch.no_grad():
        for u, seq, pos, neg in loader:
            model((u, seq, pos, neg, seq, None, None), mode="extract")
    return torch.cat(model.extract_embs_list, dim=0).numpy().astype(np.float32, copy=False)


def rowwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_t = torch.from_numpy(left)
    right_t = torch.from_numpy(right)
    return F.cosine_similarity(left_t, right_t, dim=1).cpu().numpy().astype(np.float32, copy=False)


def evaluate_one_model(
    base_args: argparse.Namespace,
    dataset,
    eval_users: Sequence[int],
    spec: ModelSpec,
    split: str,
    seed: int,
    alt_perturbation: str,
) -> Dict[str, float]:
    args = argparse.Namespace(**vars(base_args))
    args.save_dir = spec.save_dir
    args.recsys_ckpt_path = spec.recsys_ckpt_path

    model_dir = PROJECT_ROOT / "models" / args.rec_pre_trained_data / args.save_dir
    epoch = spec.epoch if spec.epoch is not None else detect_best_epoch(model_dir, args.rec_pre_trained_data, args.llm)

    print(f"\n===== Loading {spec.label} =====")
    print(f"save_dir={args.save_dir}")
    print(f"teacher_ckpt={args.recsys_ckpt_path}")
    print(f"best_epoch={epoch}")

    model = EchoRecSIModel(args).to(args.device)
    model.load_model(args, phase2_epoch=epoch, subdir="best")
    model.eval()

    user_ids, seq_original = build_seq_arrays(dataset, eval_users, split, "original", args.maxlen, seed)
    _, seq_alt = build_seq_arrays(dataset, eval_users, split, alt_perturbation, args.maxlen, seed)

    with torch.no_grad():
        emb_original = extract_llm_embeddings(model, user_ids, seq_original, args.batch_size_infer)
        emb_alt = extract_llm_embeddings(model, user_ids, seq_alt, args.batch_size_infer)

    cosine = rowwise_cosine(emb_original, emb_alt)
    result = {
        "model": spec.label,
        "users": float(len(user_ids)),
        "perturbation": alt_perturbation,
        "cosine_mean": float(np.mean(cosine)),
        "cosine_std": float(np.std(cosine)),
        "cosine_min": float(np.min(cosine)),
        "cosine_max": float(np.max(cosine)),
    }

    print(
        f"[{spec.label}] original-vs-{alt_perturbation:<10} "
        f"cosine_mean={result['cosine_mean']:.4f} "
        f"cosine_std={result['cosine_std']:.4f}"
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_outputs(output_dir: Path, dataset: str, split: str, perturbation: str, all_results: List[Dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{dataset.lower()}_{split}_repr_similarity_{perturbation}_{ts}"
    csv_path = output_dir / f"{stem}.csv"
    txt_path = output_dir / f"{stem}.txt"

    fieldnames = ["model", "users", "perturbation", "cosine_mean", "cosine_std", "cosine_min", "cosine_max"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Representation similarity evaluation\n")
        f.write(f"dataset={dataset}, split={split}, perturbation={perturbation}\n\n")
        f.write("model           cosine_mean  cosine_std   cosine_min   cosine_max   users\n")
        for row in all_results:
            f.write(
                f"{row['model']:<15} "
                f"{row['cosine_mean']:.4f}       "
                f"{row['cosine_std']:.4f}       "
                f"{row['cosine_min']:.4f}       "
                f"{row['cosine_max']:.4f}       "
                f"{int(row['users'])}\n"
            )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved TXT: {txt_path}")


def parse_args():
    parser = argparse.ArgumentParser("Representation similarity evaluation for SI checkpoints")
    parser.add_argument("--dataset", type=str, default="CDs_and_Vinyl")
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
    parser.add_argument("--output_dir", type=str, default="./analysis/repr_similarity")
    parser.add_argument(
        "--perturbation",
        type=str,
        default="shuffle",
        choices=["shuffle", "reverse", "drop_recent", "swap_last2"],
    )

    parser.add_argument("--pure_label", type=str, default="Pure-SI")
    parser.add_argument("--pure_save_dir", type=str, required=True)
    parser.add_argument("--pure_recsys_ckpt", type=str, required=True)
    parser.add_argument("--pure_epoch", type=int, default=None)

    parser.add_argument("--echo_label", type=str, default="EchoRec")
    parser.add_argument("--echo_save_dir", type=str, required=True)
    parser.add_argument("--echo_recsys_ckpt", type=str, required=True)
    parser.add_argument("--echo_epoch", type=int, default=None)
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
        all_results.append(
            evaluate_one_model(
                base_args=base_args,
                dataset=dataset,
                eval_users=eval_users,
                spec=spec,
                split=cli_args.split,
                seed=cli_args.seed,
                alt_perturbation=cli_args.perturbation,
            )
        )

    write_outputs(Path(cli_args.output_dir) / cli_args.dataset, cli_args.dataset, cli_args.split, cli_args.perturbation, all_results)


if __name__ == "__main__":
    main()
