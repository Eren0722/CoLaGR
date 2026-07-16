import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_STEMS = {
    "Movies": "movies",
    "Movies_and_TV": "movies",
    "Scientific": "scientific",
    "Industrial_and_Scientific": "scientific",
    "Electronics": "electronics",
    "CDs": "cds",
    "CDs_and_Vinyl": "cds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build multi-dataset SSHG appendix table and combined curve figure")
    parser.add_argument(
        "--summary_inputs",
        nargs="+",
        required=True,
        help="Pairs in the form DatasetName=./analysis/run_dir containing summary_metrics.csv",
    )
    parser.add_argument(
        "--curve_inputs",
        nargs="*",
        default=[],
        help="Pairs in the form DatasetName=./analysis/run_dir containing jaccard_curve_summary.csv for the main-text 4-panel figure",
    )
    parser.add_argument("--out_dir", type=str, default="./analysis/multidataset_sshg")
    parser.add_argument("--paper_figure_dir", type=str, default="./paper/figure")
    parser.add_argument("--pure_label", type=str, default="Pure-SI")
    parser.add_argument("--echo_label", type=str, default="EchoRec")
    parser.add_argument("--figure_name", type=str, default="multidataset_sshg_curves")
    return parser.parse_args()


def parse_pairs(items: Sequence[str]) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid pair: {item}")
        name, path_str = item.split("=", 1)
        pairs.append((name, (PROJECT_ROOT / path_str).resolve()))
    return pairs


def dataset_stem(name: str) -> str:
    return DATASET_STEMS.get(name, name.lower().replace(" ", "_"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def build_appendix_rows(summary_pairs: Sequence[Tuple[str, Path]], pure_label: str, echo_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_name, run_dir in summary_pairs:
        summary_rows = read_csv(run_dir / "summary_metrics.csv")
        by_model = {row["model"]: row for row in summary_rows}
        pure = by_model[pure_label]
        echo = by_model[echo_label]
        rows.append(
            {
                "dataset": dataset_name,
                "delta_prealign_j20": metric(echo, "pre_jaccard@20_mean") - metric(pure, "pre_jaccard@20_mean"),
                "delta_transfer_j20": metric(echo, "post_jaccard@20_mean") - metric(pure, "post_jaccard@20_mean"),
                "delta_rbo20": metric(echo, "post_rbo@20_mean") - metric(pure, "post_rbo@20_mean"),
                "delta_ndcg10": metric(echo, "test_ndcg10") - metric(pure, "test_ndcg10"),
            }
        )
    return rows


def plot_combined_curves(
    curve_pairs: Sequence[Tuple[str, Path]],
    pure_label: str,
    echo_label: str,
    out_dir: Path,
    paper_figure_dir: Path,
    figure_name: str,
) -> None:
    if len(curve_pairs) < 2:
        print("[info] skip combined curve figure: need at least two curve inputs")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.lines import Line2D
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.weight": "bold",
            "axes.titlesize": 12.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.6,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 11.2,
            "figure.dpi": 220,
            "savefig.dpi": 400,
        }
    )

    palette = {pure_label: "#a9d6f5", echo_label: "#0e3e87"}
    markers = {pure_label: "D", echo_label: "^"}
    views = [("semantic_alignment", "Semantic Alignment"), ("structure_transfer", "Structure Transfer")]

    dataset_rows: Dict[str, List[Dict[str, str]]] = {}
    view_limits: Dict[str, Tuple[float, float]] = {}
    for dataset_name, run_dir in curve_pairs[:2]:
        rows = read_csv(run_dir / "jaccard_curve_summary.csv")
        dataset_rows[dataset_name] = rows
    for view_key, _ in views:
        lows: List[float] = []
        highs: List[float] = []
        for dataset_name, _ in curve_pairs[:2]:
            rows = dataset_rows[dataset_name]
            subset = [row for row in rows if row["view"] == view_key and row["model"] in {pure_label, echo_label}]
            for row in subset:
                mean = float(row["mean"])
                std = float(row["std"])
                lows.append(mean - std)
                highs.append(mean + std)
        y_low_all = min(lows)
        y_high_all = max(highs)
        y_margin = 0.08 * (y_high_all - y_low_all)
        view_limits[view_key] = (y_low_all - y_margin, y_high_all + y_margin)

    def draw_panel(ax, dataset_name: str, rows: List[Dict[str, str]], view_key: str, view_title: str) -> None:
        subset = [row for row in rows if row["view"] == view_key and row["model"] in {pure_label, echo_label}]
        by_model = {model: [row for row in subset if row["model"] == model] for model in [pure_label, echo_label]}

        pure_rows = by_model[pure_label]
        echo_rows = by_model[echo_label]
        ks = np.array([int(row["k"]) for row in pure_rows], dtype=np.int32)
        pure_means = np.array([float(row["mean"]) for row in pure_rows], dtype=np.float32)
        pure_stds = np.array([float(row["std"]) for row in pure_rows], dtype=np.float32)
        echo_means = np.array([float(row["mean"]) for row in echo_rows], dtype=np.float32)
        echo_stds = np.array([float(row["std"]) for row in echo_rows], dtype=np.float32)

        for model_name, means, stds in [(pure_label, pure_means, pure_stds), (echo_label, echo_means, echo_stds)]:
            color = palette[model_name]
            ax.plot(ks, means, color=color, linewidth=16.0, alpha=0.08, solid_capstyle="round", zorder=1.5)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.10, linewidth=0, zorder=2)
            ax.plot(
                ks,
                means,
                color=color,
                linewidth=1.95,
                marker=markers[model_name],
                markersize=10.2 if model_name == echo_label else 9.6,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                zorder=3,
            )

        if 20 in ks.tolist():
            ann_idx = ks.tolist().index(20)
        else:
            ann_idx = len(ks) // 2
        gain = 0.0 if abs(float(pure_means[ann_idx])) < 1e-12 else 100.0 * (float(echo_means[ann_idx]) - float(pure_means[ann_idx])) / float(pure_means[ann_idx])

        ax.set_title(view_title, pad=3.5)
        ax.set_xlabel("Neighborhood size $k$")
        ax.set_ylabel("Mean Jaccard")
        ax.set_xticks(ks.tolist())
        ax.set_xlim(float(ks.min()) - 4.0, float(ks.max()) + 4.0)
        ymin, ymax = view_limits[view_key]
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
        ax.tick_params(axis="both", which="major", width=1.2, length=5.0, direction="out", pad=2.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.set_facecolor("white")
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        ann_pos = {
            ("CDs", "semantic_alignment"): (0.25, 0.24),
            ("CDs", "structure_transfer"): (0.57, 0.16),
            ("Scientific", "semantic_alignment"): (0.24, 0.24),
            ("Scientific", "structure_transfer"): (0.12, 0.82),
        }
        ann_x_frac, ann_y_frac = ann_pos.get((dataset_name, view_key), (0.74, 0.13))
        ax.text(
            ann_x_frac,
            ann_y_frac,
            f"$\\Delta_{{20}}={gain:+.1f}\\%$",
            transform=ax.transAxes,
            fontsize=10.8,
            color="#000000",
            va="center",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.86),
        )

    fig = plt.figure(figsize=(10.7, 3.22))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.12, 1.12, 0.42, 1.12, 1.12], wspace=0.16)
    axes = np.asarray(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[0, 4])],
        ]
    )
    legend_ax = fig.add_subplot(gs[0, 2])
    legend_ax.axis("off")
    fig.patch.set_facecolor("white")
    for dataset_idx, (dataset_name, _) in enumerate(curve_pairs[:2]):
        rows = dataset_rows[dataset_name]
        for view_idx, (view_key, view_title) in enumerate(views):
            draw_panel(axes[dataset_idx, view_idx], dataset_name, rows, view_key, view_title)
            axes[dataset_idx, view_idx].set_box_aspect(0.66)

    legend_handles = [
        Line2D([0], [0], color=palette[pure_label], marker=markers[pure_label], linewidth=1.55, markersize=7.2, label=pure_label),
        Line2D([0], [0], color=palette[echo_label], marker=markers[echo_label], linewidth=1.55, markersize=7.6, label=echo_label),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=1,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.55,
        prop={"weight": "bold", "size": 11.0},
    )
    fig.subplots_adjust(left=0.045, right=0.992, top=0.77, bottom=0.20)

    for dataset_idx, (dataset_name, _) in enumerate(curve_pairs[:2]):
        left_ax = axes[dataset_idx, 0]
        right_ax = axes[dataset_idx, 1]
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        x_center = 0.5 * (left_pos.x0 + right_pos.x1)
        y_top = max(left_pos.y1, right_pos.y1) + 0.032
        fig.text(
            x_center,
            y_top,
            f"({chr(ord('a') + dataset_idx)}) {dataset_name}",
            ha="center",
            va="bottom",
            fontsize=12.4,
            fontweight="bold",
            color="#111111",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    for target_dir in [out_dir, paper_figure_dir]:
        fig.savefig(target_dir / f"{figure_name}.pdf", bbox_inches="tight")
        fig.savefig(target_dir / f"{figure_name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_pair_curves(
    curve_pairs: Sequence[Tuple[str, Path]],
    pure_label: str,
    echo_label: str,
    out_dir: Path,
    paper_figure_dir: Path,
) -> None:
    if not curve_pairs:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.weight": "bold",
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.6,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "figure.dpi": 220,
            "savefig.dpi": 400,
        }
    )

    palette = {pure_label: "#a9d6f5", echo_label: "#0e3e87"}
    markers = {pure_label: "D", echo_label: "^"}
    views = [("semantic_alignment", "Semantic Alignment"), ("structure_transfer", "Structure Transfer")]

    def draw_panel(ax, rows: List[Dict[str, str]], view_key: str, show_ylabel: bool) -> None:
        subset = [row for row in rows if row["view"] == view_key and row["model"] in {pure_label, echo_label}]
        by_model = {model: [row for row in subset if row["model"] == model] for model in [pure_label, echo_label]}

        pure_rows = by_model[pure_label]
        echo_rows = by_model[echo_label]
        ks = np.array([int(row["k"]) for row in pure_rows], dtype=np.int32)
        pure_means = np.array([float(row["mean"]) for row in pure_rows], dtype=np.float32)
        pure_stds = np.array([float(row["std"]) for row in pure_rows], dtype=np.float32)
        echo_means = np.array([float(row["mean"]) for row in echo_rows], dtype=np.float32)
        echo_stds = np.array([float(row["std"]) for row in echo_rows], dtype=np.float32)

        for model_name, means, stds in [(pure_label, pure_means, pure_stds), (echo_label, echo_means, echo_stds)]:
            color = palette[model_name]
            ax.plot(ks, means, color=color, linewidth=13.0, alpha=0.08, solid_capstyle="round", zorder=1.5)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.10, linewidth=0, zorder=2)
            ax.plot(
                ks,
                means,
                color=color,
                linewidth=1.55,
                marker=markers[model_name],
                markersize=8.2 if model_name == echo_label else 7.8,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                zorder=3,
            )

        if 20 in ks.tolist():
            ann_idx = ks.tolist().index(20)
        else:
            ann_idx = len(ks) // 2
        gain = 0.0 if abs(float(pure_means[ann_idx])) < 1e-12 else 100.0 * (float(echo_means[ann_idx]) - float(pure_means[ann_idx])) / float(pure_means[ann_idx])

        ax.set_xlabel("Neighborhood size $k$")
        ax.set_ylabel("Mean Jaccard" if show_ylabel else "")
        ax.set_xticks(ks.tolist())
        ax.set_xlim(float(ks.min()) - 4.0, float(ks.max()) + 4.0)

        if view_key == "semantic_alignment":
            ax.set_ylim(0.018, 0.110)
            ax.yaxis.set_major_locator(FixedLocator([0.02, 0.04, 0.06, 0.08]))
            ann_xy = (0.74, 0.13)
        else:
            ax.set_ylim(0.045, 0.305)
            ax.yaxis.set_major_locator(FixedLocator([0.05, 0.10, 0.15, 0.20, 0.25]))
            ann_xy = (0.74, 0.13)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.labelpad = 0.5
        ax.tick_params(axis="both", which="major", width=1.15, length=4.2, direction="out", pad=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.set_facecolor("white")
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.text(
            ann_xy[0],
            ann_xy[1],
            fr"$\Delta_{{20}}={gain:+.1f}\%$",
            transform=ax.transAxes,
            fontsize=9.4,
            color="#000000",
            va="center",
            ha="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, run_dir in curve_pairs:
        rows = read_csv(run_dir / "jaccard_curve_summary.csv")
        fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.28))
        fig.patch.set_facecolor("white")
        for idx, (ax, (view_key, _view_title)) in enumerate(zip(axes, views)):
            draw_panel(ax, rows, view_key, show_ylabel=(idx == 0))
            ax.set_box_aspect(0.72)

        legend_handles = [
            Line2D([0], [0], color=palette[pure_label], marker=markers[pure_label], linewidth=1.45, markersize=6.6, label=pure_label),
            Line2D([0], [0], color=palette[echo_label], marker=markers[echo_label], linewidth=1.45, markersize=6.8, label=echo_label),
        ]
        for ax in axes:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=False,
                ncol=2,
                handlelength=1.0,
                handletextpad=0.32,
                columnspacing=0.62,
                borderaxespad=0.10,
                prop={"weight": "bold", "size": 8.9},
            )
            legend = ax.get_legend()
            legend.set_frame_on(True)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.9)
            legend.get_frame().set_boxstyle("round,pad=0.08")
            legend.set_bbox_to_anchor((0.015, 1.015))

        fig.subplots_adjust(left=0.08, right=0.992, top=0.92, bottom=0.39, wspace=0.16)
        for ax, (_, view_title), tag in zip(axes, views, ['(a)', '(b)']):
            pos = ax.get_position()
            fig.text(
                0.5 * (pos.x0 + pos.x1),
                pos.y0 - 0.108,
                f"{tag} {view_title}",
                ha="center",
                va="top",
                fontsize=10.6,
                fontweight="bold",
            )

        stem = dataset_stem(dataset_name)
        for target_dir in [out_dir, paper_figure_dir]:
            fig.savefig(target_dir / f"{stem}_sshg_pair.pdf")
            fig.savefig(target_dir / f"{stem}_sshg_pair.png", dpi=300)
        plt.close(fig)


def plot_dataset_single_panels(
    curve_pairs: Sequence[Tuple[str, Path]],
    pure_label: str,
    echo_label: str,
    out_dir: Path,
    paper_figure_dir: Path,
) -> None:
    if not curve_pairs:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
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

    palette = {pure_label: "#a9d6f5", echo_label: "#0e3e87"}
    markers = {pure_label: "D", echo_label: "^"}
    views = [("semantic_alignment", "Semantic Alignment"), ("structure_transfer", "Structure Transfer")]

    def draw_single(ax, rows: List[Dict[str, str]], view_key: str) -> None:
        subset = [row for row in rows if row["view"] == view_key and row["model"] in {pure_label, echo_label}]
        by_model = {model: [row for row in subset if row["model"] == model] for model in [pure_label, echo_label]}

        pure_rows = by_model[pure_label]
        echo_rows = by_model[echo_label]
        ks = np.array([int(row["k"]) for row in pure_rows], dtype=np.int32)
        pure_means = np.array([float(row["mean"]) for row in pure_rows], dtype=np.float32)
        pure_stds = np.array([float(row["std"]) for row in pure_rows], dtype=np.float32)
        echo_means = np.array([float(row["mean"]) for row in echo_rows], dtype=np.float32)
        echo_stds = np.array([float(row["std"]) for row in echo_rows], dtype=np.float32)

        for model_name, means, stds in [(pure_label, pure_means, pure_stds), (echo_label, echo_means, echo_stds)]:
            color = palette[model_name]
            ax.plot(ks, means, color=color, linewidth=16.0, alpha=0.08, solid_capstyle="round", zorder=1.5)
            ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.08, linewidth=0, zorder=2)
            ax.plot(
                ks,
                means,
                color=color,
                linewidth=1.0,
                marker=markers[model_name],
                markersize=8.8 if model_name == echo_label else 8.2,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.0,
                zorder=3,
            )

        ann_idx = ks.tolist().index(20) if 20 in ks.tolist() else len(ks) // 2
        gain = 0.0 if abs(float(pure_means[ann_idx])) < 1e-12 else 100.0 * (float(echo_means[ann_idx]) - float(pure_means[ann_idx])) / float(pure_means[ann_idx])

        ax.set_xlabel(r"Neighborhood size $\mathbf{k}$", labelpad=2.0)
        ax.xaxis.label.set_fontweight("bold")
        ax.set_xticks(ks.tolist())
        y_low = float(min((pure_means - pure_stds).min(), (echo_means - echo_stds).min()))
        y_high = float(max((pure_means + pure_stds).max(), (echo_means + echo_stds).max()))
        y_span = max(y_high - y_low, 1e-4)
        bottom_pad = 0.18 if view_key == "semantic_alignment" else 0.16
        y_min = max(0.0, y_low - bottom_pad * y_span)
        y_data_top = y_high + 0.05 * y_span
        y_max = y_min + (y_data_top - y_min) / 0.76
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", which="major", width=1.25, length=3.6, direction="out", pad=1.2)
        ax.tick_params(axis="y", which="major", pad=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.45)
        ax.set_facecolor("white")
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

        if gain >= 100.0:
            ann_x, ann_y, ann_ha, ann_fs = 0.98, 0.12, "right", 9.9
        elif gain >= 60.0:
            ann_x, ann_y, ann_ha, ann_fs = 0.95, 0.12, "right", 10.4
        else:
            ann_x, ann_y, ann_ha, ann_fs = 0.92, 0.13, "right", 10.8
        ax.text(
            ann_x,
            ann_y,
            f"$\\Delta_{{20}}={gain:+.1f}\\%$",
            transform=ax.transAxes,
            fontsize=ann_fs,
            color="#000000",
            va="center",
            ha=ann_ha,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.06", facecolor="white", edgecolor="none", alpha=0.88),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, run_dir in curve_pairs:
        rows = read_csv(run_dir / "jaccard_curve_summary.csv")
        stem = dataset_stem(dataset_name)
        for _idx, (view_key, _view_title) in enumerate(views):
            fig, ax = plt.subplots(figsize=(2.75, 2.75))
            fig.patch.set_facecolor("white")
            draw_single(ax, rows, view_key)
            ax.set_box_aspect(0.72)
            legend_handles = [
                Line2D([0], [0], color=palette[pure_label], marker=markers[pure_label], linewidth=1.0, markersize=7.3, label=pure_label),
                Line2D([0], [0], color=palette[echo_label], marker=markers[echo_label], linewidth=1.0, markersize=7.8, label=echo_label),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                frameon=False,
                ncol=2,
                handlelength=1.10,
                handletextpad=0.32,
                columnspacing=0.62,
                borderaxespad=0.10,
                prop={"weight": "bold", "size": 11},
            )
            legend = ax.get_legend()
            legend.set_frame_on(True)
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(0.9)
            legend.get_frame().set_boxstyle("round,pad=0.10")
            legend.set_bbox_to_anchor((0.02, 0.975))
            fig.subplots_adjust(left=0.18, right=0.995, top=0.93, bottom=0.22)
            suffix = "semantic" if view_key == "semantic_alignment" else "transfer"
            for target_dir in [out_dir, paper_figure_dir]:
                fig.savefig(target_dir / f"{stem}_sshg_{suffix}.pdf", bbox_inches=None, pad_inches=0.0)
                fig.savefig(target_dir / f"{stem}_sshg_{suffix}.png", dpi=300, bbox_inches=None, pad_inches=0.0)
            plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_pairs = parse_pairs(args.summary_inputs)
    curve_pairs = parse_pairs(args.curve_inputs) if args.curve_inputs else []
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    paper_figure_dir = (PROJECT_ROOT / args.paper_figure_dir).resolve()

    appendix_rows = build_appendix_rows(summary_pairs, args.pure_label, args.echo_label)
    write_csv(out_dir / "multidataset_appendix_delta_table.csv", appendix_rows)
    plot_combined_curves(curve_pairs, args.pure_label, args.echo_label, out_dir, paper_figure_dir, args.figure_name)
    plot_dataset_pair_curves(curve_pairs, args.pure_label, args.echo_label, out_dir, paper_figure_dir)
    plot_dataset_single_panels(curve_pairs, args.pure_label, args.echo_label, out_dir, paper_figure_dir)
    print(f"[done] saved multi-dataset SSHG assets to {out_dir}")


if __name__ == "__main__":
    main()
