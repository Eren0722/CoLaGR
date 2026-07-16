import argparse
import csv
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build paper-ready SSHG assets from existing diagnosis outputs")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="./analysis/cds_sshg_full/summary_metrics.csv",
    )
    parser.add_argument(
        "--curve_csv",
        type=str,
        default="./analysis/cds_sshg_full/jaccard_curve_summary.csv",
    )
    parser.add_argument(
        "--per_seed_csv",
        type=str,
        default="./analysis/cds_sshg_full/per_seed_metrics.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./analysis/cds_sshg_full/paper_ready",
    )
    parser.add_argument(
        "--paper_figure_dir",
        type=str,
        default="./paper/figure",
    )
    parser.add_argument("--pure_label", type=str, default="Pure-SI")
    parser.add_argument("--echo_label", type=str, default="EchoRec")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_row(rows: List[Dict[str, str]], model: str) -> Dict[str, str]:
    for row in rows:
        if row["model"] == model:
            return row
    raise KeyError(f"Missing model row: {model}")


def metric_value(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def rel_impr(base: float, new: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return 100.0 * (new - base) / base


def collect_available_ks(row: Dict[str, str], prefix: str) -> List[int]:
    suffix = "_mean"
    ks: List[int] = []
    for key in row.keys():
        if not key.startswith(prefix) or not key.endswith(suffix):
            continue
        k_str = key[len(prefix) : -len(suffix)]
        ks.append(int(k_str))
    return sorted(set(ks))


def build_table_rows(pure: Dict[str, str], echo: Dict[str, str]) -> List[Dict[str, object]]:
    pre_ks = collect_available_ks(pure, "pre_jaccard@")
    post_ks = collect_available_ks(pure, "post_jaccard@")
    specs: List[tuple[str, str]] = []
    for k in pre_ks:
        specs.append((f"Semantic alignment (Jaccard@{k})", f"pre_jaccard@{k}_mean"))
    for k in post_ks:
        specs.append((f"Structure transfer (Jaccard@{k})", f"post_jaccard@{k}_mean"))

    if "post_rbo@20_mean" in pure:
        specs.append(("Structure transfer (RBO@20)", "post_rbo@20_mean"))

    specs.extend(
        [
            ("Test NDCG@10", "test_ndcg10"),
            ("Test HR@10", "test_hr10"),
        ]
    )

    rows: List[Dict[str, object]] = []
    for label, key in specs:
        pure_val = metric_value(pure, key)
        echo_val = metric_value(echo, key)
        rows.append(
            {
                "metric": label,
                "Pure-SI": f"{pure_val:.4f}",
                "EchoRec": f"{echo_val:.4f}",
                "Rel. Impr.": f"{rel_impr(pure_val, echo_val):+.1f}%",
            }
        )
    return rows


def build_curve_rows(curve_rows: List[Dict[str, str]], pure_label: str, echo_label: str) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    order = {pure_label: 0, echo_label: 1}
    for row in curve_rows:
        if row["model"] not in order:
            continue
        kept.append(
            {
                "model": row["model"],
                "view": row["view"],
                "k": int(row["k"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
        )
    kept.sort(key=lambda item: (item["view"], order[item["model"]], item["k"]))
    return kept


def build_seed_curve_rows(per_seed_rows: List[Dict[str, str]], pure_label: str, echo_label: str) -> List[Dict[str, object]]:
    kept: List[Dict[str, object]] = []
    order = {pure_label: 0, echo_label: 1}
    for row in per_seed_rows:
        model = row["model"]
        if model not in order:
            continue
        seed = int(row["seed"])
        for key, value in row.items():
            if key.startswith("pre_jaccard@"):
                kept.append(
                    {
                        "seed": seed,
                        "model": model,
                        "view": "semantic_alignment",
                        "k": int(key.split("@", 1)[1]),
                        "value": float(value),
                    }
                )
            elif key.startswith("post_jaccard@"):
                kept.append(
                    {
                        "seed": seed,
                        "model": model,
                        "view": "structure_transfer",
                        "k": int(key.split("@", 1)[1]),
                        "value": float(value),
                    }
                )
    kept.sort(key=lambda item: (item["view"], order[item["model"]], item["seed"], item["k"]))
    return kept


def plot_curves(
    rows: List[Dict[str, object]],
    seed_rows: List[Dict[str, object]],
    pure_label: str,
    echo_label: str,
    out_dir: Path,
    paper_figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    palette = {
        pure_label: "#a9d6f5",
        echo_label: "#0e3e87",
    }
    panels = [
        (
            "semantic_alignment",
            "Pre-Injection: Sequential Space vs Frozen Semantic Space",
            "cds_sshg_semantic",
        ),
        (
            "structure_transfer",
            "Post-Injection: Sequential Space vs Injected LLM Space",
            "cds_sshg_transfer",
        ),
    ]

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.weight": "bold",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.labelweight": "bold",
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.dpi": 220,
            "savefig.dpi": 400,
        }
    )

    def draw_panel(
        ax,
        panel_idx: int,
        view_key: str,
        title: str,
        *,
        add_title: bool,
        add_panel_tag: bool,
    ) -> None:
        subset = [row for row in rows if row["view"] == view_key]
        subset_by_model = {
            model: [row for row in subset if row["model"] == model]
            for model in [pure_label, echo_label]
        }
        panel_markers = {
            pure_label: "D",
            echo_label: "^",
        }
        pure_rows = subset_by_model[pure_label]
        echo_rows = subset_by_model[echo_label]
        seed_subset = [row for row in seed_rows if row["view"] == view_key]

        ks = np.asarray([row["k"] for row in pure_rows], dtype=np.int32)
        pure_means = np.asarray([row["mean"] for row in pure_rows], dtype=np.float32)
        pure_stds = np.asarray([row["std"] for row in pure_rows], dtype=np.float32)
        echo_means = np.asarray([row["mean"] for row in echo_rows], dtype=np.float32)
        echo_stds = np.asarray([row["std"] for row in echo_rows], dtype=np.float32)

        for model_name, means, stds in [
            (pure_label, pure_means, pure_stds),
            (echo_label, echo_means, echo_stds),
        ]:
            color = palette[model_name]
            ax.plot(
                ks.astype(float),
                means.astype(float),
                color=color,
                linewidth=16.0,
                alpha=0.08,
                solid_capstyle="round",
                zorder=1.5,
            )
            ax.fill_between(
                ks.astype(float),
                (means - stds).astype(float),
                (means + stds).astype(float),
                color=color,
                alpha=0.08,
                linewidth=0,
                zorder=2,
            )
            container = ax.plot(
                ks.astype(float),
                means.astype(float),
                color=color,
                linewidth=1.0,
                marker=panel_markers[model_name],
                markersize=8.8 if model_name == echo_label else 8.2,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                zorder=3,
            )[0]
            if panel_idx == 0:
                method_handles.append(container)
                method_labels.append(model_name)

        y_low_all = min(float(np.min(pure_means - pure_stds)), float(np.min(echo_means - echo_stds)))
        y_high_all = max(float(np.max(pure_means + pure_stds)), float(np.max(echo_means + echo_stds)))
        y_margin = 0.08 * (y_high_all - y_low_all)
        ax.set_ylim(y_low_all - y_margin, y_high_all + y_margin)
        if view_key == "semantic_alignment":
            ax.set_ylim(0.018, 0.096)
            ax.set_yticks([0.02, 0.04, 0.06, 0.08])
            ax.set_yticklabels(["0.02", "0.04", "0.06", "0.08"])
        elif view_key == "structure_transfer":
            ax.set_ylim(0.065, 0.295)
            ax.set_yticks([0.1, 0.2])
            ax.set_yticklabels(["0.1", "0.2"])

        if add_title:
            ax.set_title(title, pad=8)
        ax.set_xlabel("Neighborhood size $k$", labelpad=2.0)
        ax.set_ylabel("Mean Jaccard")
        ax.set_xticks(ks.tolist())
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["left"].set_linewidth(1.45)
        ax.spines["bottom"].set_linewidth(1.45)
        ax.spines["top"].set_linewidth(1.45)
        ax.spines["right"].set_linewidth(1.45)
        ax.tick_params(axis="both", which="major", width=1.25, length=3.6, direction="out", pad=1.2)
        ax.tick_params(axis="y", which="major", pad=0.8)
        ax.set_facecolor("white")
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        if 20 in ks.tolist():
            ann_idx = ks.tolist().index(20)
        else:
            ann_idx = len(ks) // 2
        gap = rel_impr(float(pure_means[ann_idx]), float(echo_means[ann_idx]))
        if view_key == "semantic_alignment":
            x_ann = 33.2
            y_ann = 0.030
        elif view_key == "structure_transfer":
            x_ann = 31.5
            y_ann = 0.083
        else:
            x_ann = float(ks[ann_idx]) + 1.4
            y_ann = 0.5 * (float(pure_means[ann_idx]) + float(echo_means[ann_idx]))
        ax.text(
            x_ann,
            y_ann,
            f"$\\Delta_{{20}}={gap:+.1f}\\%$",
            fontsize=11,
            color="#000000",
            va="center",
            ha="left",
            fontweight="bold",
        )
        if add_panel_tag:
            ax.text(
                0.02,
                0.96,
                f"({chr(ord('a') + panel_idx)})",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
                color="#111111",
            )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    fig.patch.set_facecolor("white")
    method_handles = []
    method_labels = []

    for panel_idx, (view_key, title, _) in enumerate(panels):
        draw_panel(
            axes[panel_idx],
            panel_idx,
            view_key,
            title,
            add_title=False,
            add_panel_tag=True,
        )

    legend_handles = [
        Line2D([0], [0], color=palette[pure_label], linewidth=1.0, marker="D", markersize=8.0, markerfacecolor=palette[pure_label], markeredgecolor=palette[pure_label], markeredgewidth=0.0),
        Line2D([0], [0], color=palette[echo_label], linewidth=1.0, marker="^", markersize=8.4, markerfacecolor=palette[echo_label], markeredgecolor=palette[echo_label], markeredgewidth=0.0),
    ]
    fig.legend(
        legend_handles,
        method_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        handlelength=2.1,
        handletextpad=0.55,
        columnspacing=1.2,
        borderaxespad=0.1,
        prop={"weight": "bold", "size": 11.5},
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84], w_pad=1.6, pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)

    for target_dir in [out_dir, paper_figure_dir]:
        fig.savefig(target_dir / "cds_sshg_pure_vs_echo_curves.pdf", bbox_inches="tight")
        fig.savefig(target_dir / "cds_sshg_pure_vs_echo_curves.png", bbox_inches="tight")
    plt.close(fig)

    for panel_idx, (view_key, title, stem) in enumerate(panels):
        single_fig, single_ax = plt.subplots(figsize=(2.75, 2.75))
        single_fig.patch.set_facecolor("white")
        draw_panel(
            single_ax,
            panel_idx,
            view_key,
            title,
            add_title=False,
            add_panel_tag=False,
        )
        single_ax.set_box_aspect(0.72)
        single_ax.set_ylabel("")
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=palette[pure_label],
                linewidth=1.0,
                marker="D",
                markersize=7.3,
                markerfacecolor=palette[pure_label],
                markeredgecolor=palette[pure_label],
                markeredgewidth=0.0,
            ),
            Line2D(
                [0],
                [0],
                color=palette[echo_label],
                linewidth=1.0,
                marker="^",
                markersize=7.8,
                markerfacecolor=palette[echo_label],
                markeredgecolor=palette[echo_label],
                markeredgewidth=0.0,
            ),
        ]
        single_ax.legend(
            legend_handles,
            [pure_label, echo_label],
            loc="upper left",
            ncol=2,
            frameon=True,
            handlelength=1.10,
            handletextpad=0.32,
            columnspacing=0.62,
            borderaxespad=0.10,
            prop={"weight": "bold", "size": 11},
        )
        legend = single_ax.get_legend()
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.9)
        legend.get_frame().set_boxstyle("round,pad=0.10")
        legend.set_bbox_to_anchor((0.02, 0.975))
        single_fig.subplots_adjust(left=0.18, right=0.995, top=0.93, bottom=0.22)
        for target_dir in [out_dir, paper_figure_dir]:
            single_fig.savefig(target_dir / f"{stem}.pdf")
            single_fig.savefig(target_dir / f"{stem}.png")
        plt.close(single_fig)


def main() -> None:
    args = parse_args()
    summary_path = (PROJECT_ROOT / args.summary_csv).resolve()
    curve_path = (PROJECT_ROOT / args.curve_csv).resolve()
    per_seed_path = (PROJECT_ROOT / args.per_seed_csv).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    paper_figure_dir = (PROJECT_ROOT / args.paper_figure_dir).resolve()

    summary_rows = read_csv(summary_path)
    curve_rows = read_csv(curve_path)
    per_seed_rows = read_csv(per_seed_path)
    pure_row = get_row(summary_rows, args.pure_label)
    echo_row = get_row(summary_rows, args.echo_label)

    table_rows = build_table_rows(pure_row, echo_row)
    write_csv(
        out_dir / "pure_vs_echo_summary.csv",
        table_rows,
        ["metric", "Pure-SI", "EchoRec", "Rel. Impr."],
    )

    curve_table_rows = build_curve_rows(curve_rows, args.pure_label, args.echo_label)
    write_csv(
        out_dir / "pure_vs_echo_multiscale_curves.csv",
        curve_table_rows,
        ["model", "view", "k", "mean", "std"],
    )
    seed_curve_rows = build_seed_curve_rows(per_seed_rows, args.pure_label, args.echo_label)
    write_csv(
        out_dir / "pure_vs_echo_seed_curves.csv",
        seed_curve_rows,
        ["seed", "model", "view", "k", "value"],
    )

    plot_curves(curve_table_rows, seed_curve_rows, args.pure_label, args.echo_label, out_dir, paper_figure_dir)
    print(f"[done] paper-ready SSHG assets saved to {out_dir}")


if __name__ == "__main__":
    main()
