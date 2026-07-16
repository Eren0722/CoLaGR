import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    KMeans = None
    PCA = None
    TSNE = None


PURE_COLOR = "#B8D5E5"
ECHO_COLOR = "#0E3A66"
GRID_COLOR = "#D7DEE6"
TEXT_COLOR = "#14213D"
MODEL_ORDER = {"Pure-SI": 0, "EchoRec": 1}
MODEL_COLOR = {"Pure-SI": PURE_COLOR, "EchoRec": ECHO_COLOR}
BORDER_LW = 1.2
TICK_LW = 1.3
POINT_SIZE = 140
MEAN_MARKER_SIZE = 175
CONNECT_LW = 2.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Plot CDs SSHG diagnosis")
    parser.add_argument("--input_dir", type=str, default="./analysis/cds_sshg")
    parser.add_argument("--output_dir", type=str, default="./analysis/cds_sshg/figure")
    parser.add_argument("--tsne_perplexity", type=float, default=36.0)
    parser.add_argument("--tsne_seed", type=int, default=42)
    parser.add_argument("--semantic_clusters", type=int, default=5)
    return parser.parse_args()


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.weight": "bold",
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
        }
    )


def read_summary(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] not in MODEL_ORDER:
                continue
            parsed = {"model": row["model"]}
            for key, value in row.items():
                if key == "model":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda x: MODEL_ORDER.get(x["model"], 99))
    return rows


def read_per_seed(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] not in MODEL_ORDER:
                continue
            parsed = {
                "seed": int(row["seed"]),
                "model": row["model"],
            }
            for key, value in row.items():
                if key in {"seed", "model", "teacher_kind"}:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    rows.sort(key=lambda x: (x["seed"], MODEL_ORDER.get(x["model"], 99)))
    return rows


def style_axis(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(BORDER_LW)
        spine.set_color(TEXT_COLOR)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.0,
        width=TICK_LW,
        color=TEXT_COLOR,
        top=False,
        right=False,
    )


def save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def pair_values(summary_rows, key: str):
    values = {row["model"]: float(row[key]) for row in summary_rows}
    return values["Pure-SI"], values["EchoRec"]


def metric_bounds(values, floor_zero: bool = True):
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, max(abs(vmax), 1e-3) * 0.12, 0.015)
    lower = vmin - span * 0.45
    upper = vmax + span * 0.70
    if floor_zero:
        lower = max(0.0, lower)
    return lower, upper


def draw_single_metric_panel(ax, summary_rows, key: str, title: str, connect: bool = False) -> None:
    style_axis(ax)
    pure_val, echo_val = pair_values(summary_rows, key)
    y_low, y_high = metric_bounds([pure_val, echo_val])
    y_span = y_high - y_low

    x_positions = np.array([0.0, 1.0], dtype=np.float32)
    values = [pure_val, echo_val]
    colors = [PURE_COLOR, ECHO_COLOR]
    labels = ["Pure-SI", "EchoRec"]

    if connect:
        ax.plot(
            x_positions,
            values,
            color=TEXT_COLOR,
            linewidth=CONNECT_LW,
            alpha=0.9,
            zorder=1,
        )

    for x, val, color in zip(x_positions, values, colors):
        ax.scatter([x], [val], s=POINT_SIZE * 1.35, color=color, edgecolors=TEXT_COLOR, linewidths=1.1, zorder=3)
        ax.text(x, val + y_span * 0.06, f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    if pure_val > 0:
        gain = (echo_val - pure_val) / pure_val * 100.0
        ax.text(
            0.5,
            max(values) + y_span * 0.22,
            f"+{gain:.1f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=ECHO_COLOR,
        )

    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(y_low, max(y_high, max(values) + y_span * 0.32))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_title(title, pad=10, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")


def draw_grouped_bar_panel(ax, summary_rows, metrics, title: str) -> None:
    style_axis(ax)
    x_positions = np.arange(len(metrics), dtype=np.float32)
    bar_width = 0.26
    offsets = np.array([-bar_width / 2.0 - 0.015, bar_width / 2.0 + 0.015], dtype=np.float32)
    all_values = []
    for _, key in metrics:
        all_values.extend(pair_values(summary_rows, key))

    y_low, y_high = metric_bounds(all_values)
    y_span = y_high - y_low
    max_value = max(all_values)

    for idx, (label, key) in enumerate(metrics):
        pure_val, echo_val = pair_values(summary_rows, key)
        ax.bar(x_positions[idx] + offsets[0], pure_val, width=bar_width, color=PURE_COLOR, edgecolor=TEXT_COLOR, linewidth=1.0, zorder=2)
        ax.bar(x_positions[idx] + offsets[1], echo_val, width=bar_width, color=ECHO_COLOR, edgecolor=TEXT_COLOR, linewidth=1.0, zorder=2)
        ax.text(x_positions[idx] + offsets[0], pure_val + y_span * 0.04, f"{pure_val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(x_positions[idx] + offsets[1], echo_val + y_span * 0.04, f"{echo_val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if pure_val > 0:
            gain = (echo_val - pure_val) / pure_val * 100.0
            ax.text(
                x_positions[idx],
                max(pure_val, echo_val) + y_span * 0.18,
                f"+{gain:.1f}%",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=ECHO_COLOR,
            )
            max_value = max(max_value, max(pure_val, echo_val) + y_span * 0.24)

    ax.set_xlim(-0.45, len(metrics) - 0.55 if len(metrics) > 1 else 0.45)
    ax.set_ylim(y_low, max(y_high, max_value + y_span * 0.10))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _ in metrics], fontweight="bold")
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_title(title, pad=10, fontweight="bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")


def legend_handles():
    return [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PURE_COLOR, markeredgecolor=TEXT_COLOR, markeredgewidth=1.0, markersize=8, label="Pure-SI"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=ECHO_COLOR, markeredgecolor=TEXT_COLOR, markeredgewidth=1.0, markersize=8, label="EchoRec"),
    ]


def plot_bridge_panel(summary_rows, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.1))
    draw_grouped_bar_panel(
        axes[0],
        summary_rows,
        [("Jaccard@20", "pre_jaccard@20_mean")],
        "SRec-Semantic Agreement",
    )
    draw_grouped_bar_panel(
        axes[1],
        summary_rows,
        [("Jaccard@20", "post_jaccard@20_mean"), ("RBO@20", "post_rbo@20_mean")],
        "SRec-to-LLM Fidelity",
    )
    draw_grouped_bar_panel(
        axes[2],
        summary_rows,
        [("NDCG@10", "test_ndcg10"), ("HR@10", "test_hr10")],
        "Downstream Ranking",
    )

    fig.legend(
        handles=legend_handles(),
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89], w_pad=1.0)
    save_fig(fig, out_dir, "cds_sshg_bridge")


def plot_intro_teaser(summary_rows, out_dir: Path) -> None:
    teaser_metrics = [
        ("SRec-Semantic\nJaccard@20", "pre_jaccard@20_mean"),
        ("SRec-to-LLM\nJaccard@20", "post_jaccard@20_mean"),
        ("Test\nNDCG@10", "test_ndcg10"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.7, 3.3))
    for ax, (title, key) in zip(axes, teaser_metrics):
        draw_single_metric_panel(ax, summary_rows, key, title, connect=True)

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=1.0)
    save_fig(fig, out_dir, "cds_intro_teaser")


def compute_tsne_inputs(payload_path: Path, perplexity: float, seed: int, num_clusters: int):
    payload = np.load(payload_path)
    semantic = payload["semantic"]
    raw = payload["teacher_raw"]
    sacp = payload["teacher_sacp"]

    if KMeans is None or PCA is None or TSNE is None:
        labels = np.arange(semantic.shape[0]) % max(num_clusters, 1)
        return labels, raw[:, :2], sacp[:, :2]

    cluster_source = semantic
    if cluster_source.shape[1] > 64:
        cluster_source = PCA(n_components=64, random_state=seed).fit_transform(cluster_source)
    labels = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10).fit_predict(cluster_source)

    def run_tsne(emb):
        emb_ = emb
        if emb_.shape[1] > 50:
            emb_ = PCA(n_components=50, random_state=seed).fit_transform(emb_)
        model = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5.0, emb_.shape[0] / 8.0)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return model.fit_transform(emb_)

    return labels, run_tsne(raw), run_tsne(sacp)


def plot_teacher_tsne(payload_path: Path, out_dir: Path, perplexity: float, seed: int, num_clusters: int) -> None:
    labels, raw_2d, sacp_2d = compute_tsne_inputs(payload_path, perplexity, seed, num_clusters)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    cmap = plt.get_cmap("tab10")

    for ax, coords, title in zip(axes, [raw_2d, sacp_2d], ["Raw SRec", "SACP SRec"]):
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.85,
                color=cmap(int(cluster_id) % 10),
                linewidths=0,
            )
        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color(TEXT_COLOR)
        ax.set_facecolor("white")

    fig.suptitle("SRec Geometry Under Semantic Pseudo-Clusters", y=1.02, fontsize=15)
    fig.tight_layout()
    save_fig(fig, out_dir, "cds_teacher_tsne")


def plot_seed_consistency(per_seed_rows, out_dir: Path) -> None:
    metric_specs = [
        ("SRec-Semantic\nJaccard@20", "pre_jaccard@20"),
        ("SRec-to-LLM\nJaccard@20", "post_jaccard@20"),
        ("SRec-to-LLM\nRBO@20", "post_rbo@20"),
    ]
    seeds = sorted({int(row["seed"]) for row in per_seed_rows})
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))

    for ax, (title, key) in zip(axes, metric_specs):
        style_axis(ax)
        all_values = [float(row[key]) for row in per_seed_rows]
        y_low, y_high = metric_bounds(all_values)
        y_span = y_high - y_low
        ax.set_ylim(y_low, y_high)
        ax.set_xlim(-0.15, 1.15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pure-SI", "EchoRec"], fontweight="bold")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(title, pad=10, fontweight="bold")

        seed_offsets = np.linspace(-0.05, 0.05, num=max(len(seeds), 2))
        pure_values = [float(row[key]) for row in per_seed_rows if row["model"] == "Pure-SI"]
        echo_values = [float(row[key]) for row in per_seed_rows if row["model"] == "EchoRec"]
        for idx, value in enumerate(pure_values):
            ax.scatter([0.0 + seed_offsets[idx]], [value], s=86, color=PURE_COLOR, edgecolors=TEXT_COLOR, linewidths=0.9, alpha=0.95, zorder=3)
        for idx, value in enumerate(echo_values):
            ax.scatter([1.0 + seed_offsets[idx]], [value], s=86, color=ECHO_COLOR, edgecolors=TEXT_COLOR, linewidths=0.9, alpha=0.95, zorder=3)

        pure_mean = float(np.mean(pure_values))
        echo_mean = float(np.mean(echo_values))
        ax.scatter([0.0], [pure_mean], s=MEAN_MARKER_SIZE, marker="D", color=PURE_COLOR, edgecolors=TEXT_COLOR, linewidths=1.2, zorder=4)
        ax.scatter([1.0], [echo_mean], s=MEAN_MARKER_SIZE, marker="D", color=ECHO_COLOR, edgecolors=TEXT_COLOR, linewidths=1.2, zorder=4)
        ax.text(0.0, pure_mean + y_span * 0.06, f"{pure_mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(1.0, echo_mean + y_span * 0.06, f"{echo_mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        for tick in ax.get_yticklabels():
            tick.set_fontweight("bold")

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=1.0)
    save_fig(fig, out_dir, "cds_seed_consistency")


def main() -> None:
    cli_args = parse_args()
    set_plot_style()

    input_dir = Path(cli_args.input_dir).resolve()
    output_dir = Path(cli_args.output_dir).resolve()
    summary_rows = read_summary(input_dir / "summary_metrics.csv")
    per_seed_rows = read_per_seed(input_dir / "per_seed_metrics.csv")

    plot_intro_teaser(summary_rows, output_dir)
    plot_bridge_panel(summary_rows, output_dir)
    plot_teacher_tsne(input_dir / "embedding_payload.npz", output_dir, cli_args.tsne_perplexity, cli_args.tsne_seed, cli_args.semantic_clusters)
    plot_seed_consistency(per_seed_rows, output_dir)
    print(f"Saved SSHG diagnosis figures to: {output_dir}")


if __name__ == "__main__":
    main()
