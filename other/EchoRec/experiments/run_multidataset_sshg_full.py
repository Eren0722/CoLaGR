import argparse
import getpass
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    dataset: str
    output_dir: str
    pure_save_dir: str
    echo_save_dir: str
    wo_item_save_dir: str
    wo_user_save_dir: str
    wo_match_save_dir: str
    pure_teacher_ckpt: str
    echo_teacher_ckpt: str
    wo_item_teacher_ckpt: str
    wo_user_teacher_ckpt: str

    @property
    def wo_match_teacher_ckpt(self) -> str:
        return self.echo_teacher_ckpt


DATASETS: Dict[str, DatasetConfig] = {
    "movies": DatasetConfig(
        key="movies",
        dataset="Movies_and_TV",
        output_dir="./analysis/movies_sshg_full",
        pure_save_dir="movies_pure_sasrec_si_5090",
        echo_save_dir="movies_si_5090",
        wo_item_save_dir="movies_wo_itemcl_si_5090",
        wo_user_save_dir="movies_wo_usercl_si_5090",
        wo_match_save_dir="movies_wo_match_si_5090",
        pure_teacher_ckpt="./SeqRec/sasrec/Movies_and_TV/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        echo_teacher_ckpt="./SeqRec/sasrec/Movies_and_TV/movies_sa_teacher/model_metric_best.pth",
        wo_item_teacher_ckpt="./SeqRec/sasrec/Movies_and_TV/movies_wo_itemcl_teacher/model_metric_best.pth",
        wo_user_teacher_ckpt="./SeqRec/sasrec/Movies_and_TV/movies_wo_usercl_teacher/model_metric_best.pth",
    ),
    "scientific": DatasetConfig(
        key="scientific",
        dataset="Industrial_and_Scientific",
        output_dir="./analysis/scientific_sshg_full",
        pure_save_dir="scientific_pure_sasrec_si_5090",
        echo_save_dir="scientific_si_5090",
        wo_item_save_dir="scientific_wo_itemcl_si_5090",
        wo_user_save_dir="scientific_wo_usercl_si_5090",
        wo_match_save_dir="scientific_wo_match_si_5090",
        pure_teacher_ckpt="./SeqRec/sasrec/Industrial_and_Scientific/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        echo_teacher_ckpt="./SeqRec/sasrec/Industrial_and_Scientific/scientific_sa_teacher/model_metric_best.pth",
        wo_item_teacher_ckpt="./SeqRec/sasrec/Industrial_and_Scientific/scientific_wo_itemcl_teacher/model_metric_best.pth",
        wo_user_teacher_ckpt="./SeqRec/sasrec/Industrial_and_Scientific/scientific_wo_usercl_teacher/model_metric_best.pth",
    ),
    "electronics": DatasetConfig(
        key="electronics",
        dataset="Electronics",
        output_dir="./analysis/electronics_sshg_full",
        pure_save_dir="electronics_pure_sasrec_si_5090",
        echo_save_dir="electronics_si_5090",
        wo_item_save_dir="electronics_wo_itemcl_si_5090",
        wo_user_save_dir="electronics_wo_usercl_si_5090",
        wo_match_save_dir="electronics_wo_match_si_5090",
        pure_teacher_ckpt="./SeqRec/sasrec/Electronics/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        echo_teacher_ckpt="./SeqRec/sasrec/Electronics/electronics_sa_teacher/model_metric_best.pth",
        wo_item_teacher_ckpt="./SeqRec/sasrec/Electronics/electronics_wo_itemcl_teacher/model_metric_best.pth",
        wo_user_teacher_ckpt="./SeqRec/sasrec/Electronics/electronics_wo_usercl_teacher/model_metric_best.pth",
    ),
    "cds": DatasetConfig(
        key="cds",
        dataset="CDs_and_Vinyl",
        output_dir="./analysis/cds_sshg_full",
        pure_save_dir="cds_pure_sasrec_si_cand4_5090",
        echo_save_dir="cds_si_cand4_5090",
        wo_item_save_dir="cds_wo_itemcl_si_cand4_5090",
        wo_user_save_dir="cds_wo_usercl_si_cand4_5090",
        wo_match_save_dir="cds_wo_match_si_cand4_5090",
        pure_teacher_ckpt="./SeqRec/sasrec/CDs_and_Vinyl/SASRec_saving.epoch=200.lr=0.001.layer=2.head=1.hidden=64.maxlen=128.pth",
        echo_teacher_ckpt="./SeqRec/sasrec/CDs_and_Vinyl/cds_sa_teacher/model_metric_best.pth",
        wo_item_teacher_ckpt="./SeqRec/sasrec/CDs_and_Vinyl/cds_wo_itemcl_teacher/model_metric_best.pth",
        wo_user_teacher_ckpt="./SeqRec/sasrec/CDs_and_Vinyl/cds_wo_usercl_teacher/model_metric_best.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run full SSHG diagnosis for all dataset--variant pairs")
    parser.add_argument("--datasets", nargs="+", default=["movies", "scientific", "electronics", "cds"], choices=sorted(DATASETS))
    parser.add_argument("--llm_path", type=str, default="")
    parser.add_argument("--llm", type=str, default="llama-3b")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--asset_root", type=str, default="./SA_assets")
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--sample_size", type=int, default=2000)
    parser.add_argument("--teacher_batch_size", type=int, default=1024)
    parser.add_argument("--batch_size_infer", type=int, default=8)
    parser.add_argument("--neighbor_ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--transfer_k", type=int, default=20)
    parser.add_argument("--rbo_p", type=float, default=0.9)
    parser.add_argument("--hf_local_only", action="store_true")
    parser.add_argument("--hf_cache_dir", type=str, default="./.cache/huggingface")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_llm_path(raw_path: str) -> str:
    raw_path = raw_path.strip()
    if raw_path:
        return raw_path

    user = getpass.getuser()
    candidates = [
        Path.home() / "cyx" / "models" / "llama3_3b",
        Path(f"/home/{user}/cyx/models/llama3_3b"),
        Path.home() / "models" / "llama3_3b",
        Path(f"/home/{user}/models/llama3_3b"),
        Path("/home/cyx/models/llama3_3b"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not resolve --llm_path automatically. Searched:\n"
        f"{searched}\n"
        "Set it manually, e.g. --llm_path /home/cyx/cyx/models/llama3_3b"
    )


def resolve_save_dir(dataset: str, save_dir: str) -> str:
    """Prefer the configured save_dir, but support nested server layouts.

    Some archived runs were copied under models/<dataset>/<dataset>/<dataset>/,
    while others live under top-level or dataset-archive directories such as
    <dataset>/<dataset>/<dataset>/ or CDs_and_Vinyl/Electronics/Electronics/Electronics/.
    The underlying loader joins PROJECT_ROOT/models/<dataset>/<save_dir>, so we
    can pass a slash-containing or ..-containing save_dir when those layouts are
    present.
    """
    model_root = PROJECT_ROOT / "models" / dataset
    candidates = {
        save_dir: model_root / save_dir,
        f"{dataset}/{save_dir}": model_root / dataset / save_dir,
        f"{dataset}/{dataset}/{save_dir}": model_root / dataset / dataset / save_dir,
        f"../../{dataset}/{save_dir}": PROJECT_ROOT / dataset / save_dir,
        f"../../{dataset}/{dataset}/{save_dir}": PROJECT_ROOT / dataset / dataset / save_dir,
        f"../../{dataset}/{dataset}/{dataset}/{save_dir}": PROJECT_ROOT / dataset / dataset / dataset / save_dir,
        f"../../CDs_and_Vinyl/{dataset}/{save_dir}": PROJECT_ROOT / "CDs_and_Vinyl" / dataset / save_dir,
        f"../../CDs_and_Vinyl/{dataset}/{dataset}/{save_dir}": PROJECT_ROOT / "CDs_and_Vinyl" / dataset / dataset / save_dir,
        f"../../CDs_and_Vinyl/{dataset}/{dataset}/{dataset}/{save_dir}": PROJECT_ROOT / "CDs_and_Vinyl" / dataset / dataset / dataset / save_dir,
    }
    for candidate, path in candidates.items():
        if (path / "best").is_dir():
            if candidate != save_dir:
                print(f"[info] resolved save_dir {save_dir} -> {candidate}", flush=True)
            return candidate
    return save_dir


def build_command(config: DatasetConfig, args: argparse.Namespace) -> List[str]:
    pure_save_dir = resolve_save_dir(config.dataset, config.pure_save_dir)
    echo_save_dir = resolve_save_dir(config.dataset, config.echo_save_dir)
    wo_item_save_dir = resolve_save_dir(config.dataset, config.wo_item_save_dir)
    wo_user_save_dir = resolve_save_dir(config.dataset, config.wo_user_save_dir)
    wo_match_save_dir = resolve_save_dir(config.dataset, config.wo_match_save_dir)

    cmd = [
        sys.executable,
        "experiments/cds_sshg_full_table.py",
        "--dataset",
        config.dataset,
        "--asset_root",
        args.asset_root,
        "--maxlen",
        str(args.maxlen),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--sample_size",
        str(args.sample_size),
        "--teacher_batch_size",
        str(args.teacher_batch_size),
        "--batch_size_infer",
        str(args.batch_size_infer),
        "--llm",
        args.llm,
        "--llm_path",
        args.llm_path,
        "--neighbor_ks",
        *(str(k) for k in args.neighbor_ks),
        "--transfer_k",
        str(args.transfer_k),
        "--rbo_p",
        str(args.rbo_p),
        "--output_dir",
        config.output_dir,
        "--include_wo_match",
        "--pure_save_dir",
        pure_save_dir,
        "--pure_teacher_ckpt",
        config.pure_teacher_ckpt,
        "--echo_save_dir",
        echo_save_dir,
        "--echo_teacher_ckpt",
        config.echo_teacher_ckpt,
        "--wo_item_save_dir",
        wo_item_save_dir,
        "--wo_item_teacher_ckpt",
        config.wo_item_teacher_ckpt,
        "--wo_user_save_dir",
        wo_user_save_dir,
        "--wo_user_teacher_ckpt",
        config.wo_user_teacher_ckpt,
        "--wo_match_save_dir",
        wo_match_save_dir,
        "--wo_match_teacher_ckpt",
        config.wo_match_teacher_ckpt,
    ]
    if args.hf_local_only:
        cmd.append("--hf_local_only")
    if args.hf_cache_dir:
        cmd.extend(["--hf_cache_dir", args.hf_cache_dir])
    return cmd


def main() -> None:
    args = parse_args()
    if "/path/to/your" in args.llm_path or args.llm_path.strip() in {"", "DUMMY"}:
        if args.llm_path.strip() in {"", "DUMMY"}:
            args.llm_path = ""
        else:
            raise ValueError("--llm_path must be the real local LLM directory, not a placeholder")
    args.llm_path = resolve_llm_path(args.llm_path)
    for dataset_key in args.datasets:
        config = DATASETS[dataset_key]
        cmd = build_command(config, args)
        printable = " ".join(shlex.quote(part) for part in cmd)
        print(f"[run] {dataset_key}: {printable}", flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
