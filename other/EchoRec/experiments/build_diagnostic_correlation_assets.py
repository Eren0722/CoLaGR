import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_INPUTS: Sequence[Tuple[str, str]] = (
    ("Movies", "./analysis/movies_sshg_full"),
    ("Scientific", "./analysis/scientific_sshg_full"),
    ("Electronics", "./analysis/electronics_sshg_full"),
    ("CDs", "./analysis/cds_sshg_full"),
)

VARIANT_ORDER = {
    "Pure-SI": 0,
    "w/o Item-CL": 1,
    "w/o User-CL": 2,
    "EchoRec": 3,
    "w/o Match": 4,
}

REQUIRED_VARIANTS = set(VARIANT_ORDER)
REQUIRED_COLUMNS = {
    "model",
    "pre_jaccard@20_mean",
    "post_jaccard@20_mean",
    "post_rbo@20_mean",
    "test_ndcg10",
}

METRIC_COLUMNS = {
    "PreAlign20": "pre_jaccard@20_mean",
    "Jaccard20": "post_jaccard@20_mean",
    "RBO20": "post_rbo@20_mean",
    "NDCG10": "test_ndcg10",
}

CORRELATION_PAIRS = (
    ("d_PreAlign20", "d_Jaccard20", r"$\Delta$PreAlign@20", r"$\Delta$Jaccard@20"),
    ("d_PreAlign20", "d_RBO20", r"$\Delta$PreAlign@20", r"$\Delta$RBO@20"),
    ("d_Jaccard20", "d_NDCG10", r"$\Delta$Jaccard@20", r"$\Delta$NDCG@10"),
    ("d_RBO20", "d_NDCG10", r"$\Delta$RBO@20", r"$\Delta$NDCG@10"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build diagnostic-correlation CSV, LaTeX table, and scatter figure")
    parser.add_argument("--out_dir", type=str, default="./analysis/diagnostic_correlation")
    parser.add_argument("--paper_figure_dir", type=str, default="./paper/figure")
    parser.add_argument("--pure_label", type=str, default="Pure-SI")
    parser.add_argument("--allow_partial", action="store_true", help="Allow missing dataset inputs for exploratory local checks")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_value(row: Dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite value for {key}: {row}")
    return value


def validate_summary(dataset: str, rows: Sequence[Dict[str, str]], allow_partial: bool) -> None:
    if not rows:
        raise ValueError(f"{dataset}: empty summary_metrics.csv")
    missing_columns = REQUIRED_COLUMNS.difference(rows[0].keys())
    if missing_columns:
        raise ValueError(f"{dataset}: missing columns {sorted(missing_columns)}")
    variants = {str(row["model"]) for row in rows}
    missing_variants = REQUIRED_VARIANTS.difference(variants)
    if missing_variants and not allow_partial:
        raise ValueError(f"{dataset}: missing variants {sorted(missing_variants)}")


def load_by_variant(allow_partial: bool) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    missing_inputs: List[str] = []
    for dataset, rel_dir in DATASET_INPUTS:
        summary_path = PROJECT_ROOT / rel_dir / "summary_metrics.csv"
        if not summary_path.exists():
            missing_inputs.append(f"{dataset}: {summary_path}")
            continue
        summary_rows = read_csv(summary_path)
        validate_summary(dataset, summary_rows, allow_partial)
        for row in sorted(summary_rows, key=lambda item: VARIANT_ORDER.get(str(item["model"]), 99)):
            variant = str(row["model"])
            if variant not in VARIANT_ORDER:
                continue
            output_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "PreAlign20": float_value(row, METRIC_COLUMNS["PreAlign20"]),
                    "Jaccard20": float_value(row, METRIC_COLUMNS["Jaccard20"]),
                    "RBO20": float_value(row, METRIC_COLUMNS["RBO20"]),
                    "NDCG10": float_value(row, METRIC_COLUMNS["NDCG10"]),
                }
            )

    if missing_inputs and not allow_partial:
        raise FileNotFoundError("Missing required full diagnostic summaries:\n" + "\n".join(missing_inputs))
    return output_rows


def build_delta_rows(rows: Sequence[Dict[str, object]], pure_label: str) -> List[Dict[str, object]]:
    baseline: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["variant"] == pure_label:
            baseline[str(row["dataset"])] = {
                metric: float(row[metric])
                for metric in ("PreAlign20", "Jaccard20", "RBO20", "NDCG10")
            }

    delta_rows: List[Dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        if variant == pure_label:
            continue
        if dataset not in baseline:
            raise ValueError(f"{dataset}: missing {pure_label} baseline")
        base = baseline[dataset]
        delta_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "d_PreAlign20": float(row["PreAlign20"]) - base["PreAlign20"],
                "d_Jaccard20": float(row["Jaccard20"]) - base["Jaccard20"],
                "d_RBO20": float(row["RBO20"]) - base["RBO20"],
                "d_NDCG10": float(row["NDCG10"]) - base["NDCG10"],
            }
        )
    return delta_rows


def rankdata(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.shape[0], dtype=float)
    sorted_values = arr[order]
    start = 0
    while start < arr.shape[0]:
        end = start + 1
        while end < arr.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman(x_values: Iterable[float], y_values: Iterable[float]) -> float:
    x_rank = rankdata(x_values)
    y_rank = rankdata(y_values)
    if x_rank.size < 2:
        raise ValueError("Spearman correlation requires at least two points")
    rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
    if not math.isfinite(rho):
        raise ValueError("Spearman correlation is non-finite")
    return rho


def build_correlation_rows(delta_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for x_key, y_key, x_label, y_label in CORRELATION_PAIRS:
        rho = spearman((float(row[x_key]) for row in delta_rows), (float(row[y_key]) for row in delta_rows))
        rows.append(
            {
                "pair": f"{x_label} vs. {y_label}",
                "x_key": x_key,
                "y_key": y_key,
                "rho": rho,
            }
        )
    return rows


def latex_float(value: float) -> str:
    return f"{value:.3f}"


def write_latex_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Diagnostic correlation over dataset--variant pairs. Pure-SI is excluded after serving as the within-dataset baseline.}",
        "\\label{tab:diagnostic_correlation}",
        "\\small",
        "\\setlength{\\tabcolsep}{7pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Correlation pair & Spearman's $\\rho$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(f"{row['pair']} & {latex_float(float(row['rho']))} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_scatter(path: Path, delta_rows: Sequence[Dict[str, object]], correlation_rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset_colors = {
        "Movies": "#4C78A8",
        "Scientific": "#F58518",
        "Electronics": "#54A24B",
        "CDs": "#B279A2",
    }
    variant_markers = {
        "w/o Item-CL": "o",
        "w/o User-CL": "s",
        "EchoRec": "^",
        "w/o Match": "D",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for axis, pair, corr in zip(axes.flatten(), CORRELATION_PAIRS, correlation_rows):
        x_key, y_key, x_label, y_label = pair
        for row in delta_rows:
            dataset = str(row["dataset"])
            variant = str(row["variant"])
            axis.scatter(
                float(row[x_key]),
                float(row[y_key]),
                s=42,
                marker=variant_markers.get(variant, "o"),
                color=dataset_colors.get(dataset, "#666666"),
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(rf"Spearman $\rho$={float(corr['rho']):.3f}", fontsize=10)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=color, markeredgecolor="black", markersize=6)
        for name, color in dataset_colors.items()
    ]
    variant_handles = [
        plt.Line2D([0], [0], marker=marker, color="black", label=name, linestyle="None", markersize=6)
        for name, marker in variant_markers.items()
    ]
    fig.legend(handles=dataset_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.legend(handles=variant_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.065))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def validate_counts(rows: Sequence[Dict[str, object]], delta_rows: Sequence[Dict[str, object]], allow_partial: bool) -> None:
    if allow_partial:
        return
    if len(rows) != 20:
        raise ValueError(f"Expected 20 dataset--variant rows, got {len(rows)}")
    if len(delta_rows) != 16:
        raise ValueError(f"Expected 16 non-baseline delta rows, got {len(delta_rows)}")


def main() -> None:
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    paper_figure_dir = (PROJECT_ROOT / args.paper_figure_dir).resolve()

    by_variant_rows = load_by_variant(args.allow_partial)
    delta_rows = build_delta_rows(by_variant_rows, args.pure_label)
    validate_counts(by_variant_rows, delta_rows, args.allow_partial)
    correlation_rows = build_correlation_rows(delta_rows)

    write_csv(
        out_dir / "diagnostic_by_variant.csv",
        by_variant_rows,
        ["dataset", "variant", "PreAlign20", "Jaccard20", "RBO20", "NDCG10"],
    )
    write_csv(
        out_dir / "diagnostic_delta_points.csv",
        delta_rows,
        ["dataset", "variant", "d_PreAlign20", "d_Jaccard20", "d_RBO20", "d_NDCG10"],
    )
    write_csv(out_dir / "diagnostic_correlation.csv", correlation_rows, ["pair", "x_key", "y_key", "rho"])
    write_latex_table(paper_figure_dir / "diagnostic_correlation_table.tex", correlation_rows)
    plot_scatter(paper_figure_dir / "diagnostic_correlation_scatter.pdf", delta_rows, correlation_rows)

    for row in correlation_rows:
        print(f"{row['pair']}: rho={float(row['rho']):.3f}")
    print(f"[write] {out_dir / 'diagnostic_by_variant.csv'}")
    print(f"[write] {out_dir / 'diagnostic_delta_points.csv'}")
    print(f"[write] {paper_figure_dir / 'diagnostic_correlation_table.tex'}")
    print(f"[write] {paper_figure_dir / 'diagnostic_correlation_scatter.pdf'}")


if __name__ == "__main__":
    main()
