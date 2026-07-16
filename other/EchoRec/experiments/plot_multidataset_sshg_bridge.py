import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_COLOR = "#14213D"
PURE_COLOR = "#B8D5E5"
ECHO_COLOR = "#0E3A66"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot cross-dataset SSHG bridge scatter")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs in the form DatasetName=./analysis/run_dir",
    )
    parser.add_argument("--out_dir", type=str, default="./analysis/sshg_bridge")
    parser.add_argument("--paper_figure_dir", type=str, default="./paper/figure")
    parser.add_argument("--scatter_name", type=str, default="sshg_cross_dataset_bridge")
    return parser.parse_args()


def parse_input_pairs(items: List[str]) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid input pair: {item}")
        name, path_str = item.split("=", 1)
        pairs.append((name, (PROJECT_ROOT / path_str).resolve()))
    return pairs


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dataset_summary(name: str, summary_rows: List[Dict[str, str]]) -> Dict[str, object]:
    by_model = {row["model"]: row for row in summary_rows}
    pure = by_model["Pure-SI"]
    echo = by_model["EchoRec"]
    return {
        "dataset": name,
        "delta_semantic_alignment_j20": to_float(echo, "pre_jaccard@20_mean") - to_float(pure, "pre_jaccard@20_mean"),
        "delta_transfer_j20": to_float(echo, "post_jaccard@20_mean") - to_float(pure, "post_jaccard@20_mean"),
        "delta_rbo20": to_float(echo, "post_rbo@20_mean") - to_float(pure, "post_rbo@20_mean"),
        "delta_ndcg10": to_float(echo, "test_ndcg10") - to_float(pure, "test_ndcg10"),
    }


def seed_summary(name: str, per_seed_rows: List[Dict[str, str]], dataset_delta_ndcg10: float) -> List[Dict[str, object]]:
    pure_rows = {int(row["seed"]): row for row in per_seed_rows if row["model"] == "Pure-SI"}
    echo_rows = {int(row["seed"]): row for row in per_seed_rows if row["model"] == "EchoRec"}
    rows: List[Dict[str, object]] = []
    for seed in sorted(set(pure_rows) & set(echo_rows)):
        pure = pure_rows[seed]
        echo = echo_rows[seed]
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "delta_semantic_alignment_j20": to_float(echo, "pre_jaccard@20") - to_float(pure, "pre_jaccard@20"),
                "delta_transfer_j20": to_float(echo, "post_jaccard@20") - to_float(pure, "post_jaccard@20"),
                "delta_rbo20": to_float(echo, "post_rbo@20") - to_float(pure, "post_rbo@20"),
                "dataset_delta_ndcg10": dataset_delta_ndcg10,
            }
        )
    return rows


def plot_scatter(rows: List[Dict[str, object]], target_path: Path) -> float:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.weight": "bold",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )

    x = np.array([float(row["delta_semantic_alignment_j20"]) for row in rows], dtype=np.float64)
    y = np.array([float(row["delta_transfer_j20"]) for row in rows], dtype=np.float64)
    c = np.array([float(row["dataset_delta_ndcg10"]) for row in rows], dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(rows) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(5.6, 4.35))
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color(TEXT_COLOR)
    ax.tick_params(axis="both", which="major", direction="out", length=3.5, width=1.2, color=TEXT_COLOR)

    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="Blues",
        s=88,
        edgecolors=TEXT_COLOR,
        linewidths=0.9,
        zorder=3,
    )

    if len(rows) >= 2:
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.min(x)) * 0.95, float(np.max(x)) * 1.05, 100)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, color=ECHO_COLOR, linewidth=1.8, alpha=0.9, zorder=2)

    for row in rows:
        ax.annotate(
            f'{row["dataset"]}-{row["seed"]}',
            (float(row["delta_semantic_alignment_j20"]), float(row["delta_transfer_j20"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ax.set_xlabel("Pre-injection semantic alignment improvement")
    ax.set_ylabel("Post-injection transfer fidelity improvement")
    ax.set_title(f"Cross-dataset mechanism bridge (Pearson = {pearson:.3f})")
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label("Dataset-level NDCG@10 improvement", fontweight="bold")

    x_pad = max(0.002, (float(np.max(x)) - float(np.min(x))) * 0.12)
    y_pad = max(0.004, (float(np.max(y)) - float(np.min(y))) * 0.12)
    ax.set_xlim(float(np.min(x)) - x_pad, float(np.max(x)) + x_pad)
    ax.set_ylim(float(np.min(y)) - y_pad, float(np.max(y)) + y_pad)

    fig.tight_layout()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return pearson


def main() -> None:
    args = parse_args()
    pairs = parse_input_pairs(args.inputs)

    dataset_rows: List[Dict[str, object]] = []
    seed_rows: List[Dict[str, object]] = []

    for name, input_dir in pairs:
        summary_rows = read_csv(input_dir / "summary_metrics.csv")
        per_seed_rows = read_csv(input_dir / "per_seed_metrics.csv")
        ds_row = dataset_summary(name, summary_rows)
        dataset_rows.append(ds_row)
        seed_rows.extend(seed_summary(name, per_seed_rows, float(ds_row["delta_ndcg10"])))

    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    write_csv(out_dir / "dataset_bridge_summary.csv", dataset_rows)
    write_csv(out_dir / "dataset_seed_bridge.csv", seed_rows)

    pearson = plot_scatter(seed_rows, out_dir / args.scatter_name)
    plot_scatter(seed_rows, (PROJECT_ROOT / args.paper_figure_dir).resolve() / args.scatter_name)

    with (out_dir / "correlation.txt").open("w", encoding="utf-8") as f:
        f.write(f"Pearson(delta semantic alignment, delta transfer fidelity) = {pearson:.6f}\n")

    print(f"[done] saved bridge assets to {out_dir}")


if __name__ == "__main__":
    main()
