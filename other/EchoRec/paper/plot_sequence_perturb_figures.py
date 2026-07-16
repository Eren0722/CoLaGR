import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "sequence_perturb"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figure" / "rq3"

PERTURBATION_ORDER = ["original", "shuffle", "reverse", "drop_recent"]
PERTURBATION_LABELS = {
    "original": "Orig.",
    "shuffle": "Shuf.",
    "reverse": "Rev.",
    "drop_recent": "Drop\nrec.",
}
MODEL_ORDER = ["Pure-SI", "EchoRec"]

DATASET_META = {
    "Movies_and_TV": {"display": "Movies", "stem": "movies"},
    "Industrial_and_Scientific": {"display": "Scientific", "stem": "scientific"},
    "Electronics": {"display": "Electronics", "stem": "electronics"},
    "CDs_and_Vinyl": {"display": "CDs", "stem": "cds"},
    "Movies": {"display": "Movies", "stem": "movies"},
    "Scientific": {"display": "Scientific", "stem": "scientific"},
    "CDs": {"display": "CDs", "stem": "cds"},
}

LEGACY_DATA = {
    "CDs": {
        "Pure-SI": {
            "hr10": [0.6023, 0.5714, 0.5473, 0.5771],
            "ndcg10": [0.3757, 0.3469, 0.3260, 0.3547],
        },
        "EchoRec": {
            "hr10": [0.6656, 0.6378, 0.6153, 0.6437],
            "ndcg10": [0.4312, 0.4088, 0.3875, 0.4093],
        },
    },
    "Scientific": {
        "Pure-SI": {
            "hr10": [0.4840, 0.4660, 0.4280, 0.4740],
            "ndcg10": [0.2856, 0.2732, 0.2479, 0.2794],
        },
        "EchoRec": {
            "hr10": [0.5785, 0.5640, 0.5360, 0.5660],
            "ndcg10": [0.3698, 0.3581, 0.3356, 0.3607],
        },
    },
}

PURE_COLOR = "#bbe1f8"
ECHO_COLOR = "#0e3e87"
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 15
X_TICK_SIZE = 14
ANNOTATION_SIZE = 11
LEGEND_SIZE = 14
BORDER_LW = 1.45
TICK_LW = 1.25
BAR_WIDTH = 0.30
BAR_OFFSET = 0.19
GROUP_STEP = 1.14


def parse_args():
    parser = argparse.ArgumentParser("Plot RQ3 sequence-perturbation figures from evaluation CSVs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CDs_and_Vinyl", "Industrial_and_Scientific"],
        help="Dataset names matching analysis/sequence_perturb/<dataset>.",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Root directory containing per-dataset sequence perturbation CSVs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for paper-ready RQ3 figures.",
    )
    parser.add_argument(
        "--panel_datasets",
        nargs="*",
        default=None,
        help="Optional two datasets for the 4-panel combined figure. Defaults to the first two datasets.",
    )
    return parser.parse_args()


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
            "font.weight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": 220,
            "savefig.dpi": 400,
            "hatch.linewidth": 0.42,
        }
    )


def dataset_meta(name: str) -> dict:
    if name in DATASET_META:
        return DATASET_META[name]
    return {"display": name.replace("_", " "), "stem": name.lower().replace("_", "_")}


def csv_is_complete(path: Path) -> bool:
    try:
        rows = load_csv_rows(path)
    except Exception:
        return False
    keys = {(row.get("model"), row.get("perturbation")) for row in rows}
    required = {(model, perturbation) for model in MODEL_ORDER for perturbation in PERTURBATION_ORDER}
    return required.issubset(keys)


def latest_csv(dataset_dir: Path) -> Path:
    candidates = sorted(dataset_dir.glob("*_sequence_perturb_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No sequence perturbation CSV found under: {dataset_dir}")
    complete = [path for path in candidates if csv_is_complete(path)]
    if complete:
        return complete[-1]
    return candidates[-1]


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_to_dataset(rows, dataset_name: str):
    meta = dataset_meta(dataset_name)
    data = {model: {"hr10": [], "ndcg10": []} for model in MODEL_ORDER}
    indexed = {(row["model"], row["perturbation"]): row for row in rows}

    for model in MODEL_ORDER:
        for perturbation in PERTURBATION_ORDER:
            key = (model, perturbation)
            if key not in indexed:
                raise KeyError(f"Missing row for {dataset_name}: {model} / {perturbation}")
            row = indexed[key]
            data[model]["hr10"].append(float(row["hr10"]))
            data[model]["ndcg10"].append(float(row["ndcg10"]))

    return {
        "raw_name": dataset_name,
        "display": meta["display"],
        "stem": meta["stem"],
        "data": data,
    }


def load_dataset_payload(dataset_name: str, analysis_dir: Path):
    meta = dataset_meta(dataset_name)
    dataset_dir = analysis_dir / dataset_name
    if dataset_dir.is_dir():
        csv_path = latest_csv(dataset_dir)
        rows = load_csv_rows(csv_path)
        payload = rows_to_dataset(rows, dataset_name)
        payload["source_csv"] = str(csv_path)
        return payload

    if meta["display"] in LEGACY_DATA:
        return {
            "raw_name": dataset_name,
            "display": meta["display"],
            "stem": meta["stem"],
            "data": LEGACY_DATA[meta["display"]],
            "source_csv": "legacy_embedded_values",
        }

    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")


def style_axis(ax):
    ax.grid(False)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(BORDER_LW)
        ax.spines[side].set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.6,
        width=TICK_LW,
        pad=1.2,
        color="black",
        top=False,
        right=False,
    )
    ax.tick_params(axis="y", which="major", pad=0.8)
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=2.2,
        width=TICK_LW * 0.9,
        color="black",
        top=False,
        right=False,
    )


def draw_metric_panel(ax, payload, metric_name, panel_title, *, xlabel=""):
    x = np.arange(len(PERTURBATION_ORDER)) * GROUP_STEP
    tick_labels = [PERTURBATION_LABELS[p] for p in PERTURBATION_ORDER]

    pure_vals = np.array(payload["data"]["Pure-SI"][metric_name])
    echo_vals = np.array(payload["data"]["EchoRec"][metric_name])

    ax.bar(
        x - BAR_OFFSET,
        pure_vals,
        width=BAR_WIDTH,
        color=PURE_COLOR,
        edgecolor="black",
        linewidth=1.3,
        hatch="//",
    )
    ax.bar(
        x + BAR_OFFSET,
        echo_vals,
        width=BAR_WIDTH,
        color=ECHO_COLOR,
        edgecolor="black",
        linewidth=1.3,
    )

    style_axis(ax)
    ax.margins(x=0.16)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontweight="bold", fontsize=X_TICK_SIZE)
    ax.set_title(panel_title, pad=7.5, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10.0)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")

    if metric_name == "ndcg10":
        y_top = max(0.78, float(np.max(echo_vals)) + 0.06)
        ax.set_ylim(0.0, y_top)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        text_pad = 0.018
    else:
        y_top = max(1.10, float(np.max(echo_vals)) + 0.10)
        ax.set_ylim(0.0, y_top)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
        text_pad = 0.026
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for idx, (x_pos, pure_v, echo_v) in enumerate(zip(x, pure_vals, echo_vals)):
        improvement = (echo_v / pure_v - 1.0) * 100.0
        y_offset = text_pad * (1.1 if idx % 2 == 0 else 3.45)
        if idx == len(x) - 1:
            x_offset = -0.06
        else:
            x_offset = -0.15 if idx % 2 == 0 else 0.15
        text_y = min(echo_v + y_offset, y_top - text_pad * 1.6)
        ax.text(
            x_pos + BAR_OFFSET + x_offset,
            text_y,
            f"+{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color="black",
        )


def save_single(payload, metric_name, out_dir: Path):
    fig, ax = plt.subplots(figsize=(2.75, 2.75))
    draw_metric_panel(ax, payload, metric_name, "", xlabel="")
    ax.set_box_aspect(0.72)
    handles = [
        Patch(facecolor=PURE_COLOR, edgecolor="black", linewidth=1.3, hatch="//", label="Pure-SI"),
        Patch(facecolor=ECHO_COLOR, edgecolor="black", linewidth=1.3, label="EchoRec"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        handlelength=1.10,
        handletextpad=0.32,
        columnspacing=0.62,
        borderaxespad=0.10,
        prop={"weight": "bold", "size": LEGEND_SIZE - 3},
    )
    legend = ax.get_legend()
    legend.set_frame_on(True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_boxstyle("round,pad=0.10")
    legend.set_bbox_to_anchor((0.02, 0.975))
    fig.subplots_adjust(left=0.18, right=0.995, top=0.93, bottom=0.22)

    stem = payload["stem"]
    metric_stem = metric_name.lower()
    for suffix in ["", "_crop"]:
        fig.savefig(out_dir / f"sequence_perturb_{stem}_{metric_stem}{suffix}.pdf")
    plt.close(fig)


def save_panel(payloads, out_dir: Path, filename: str):
    fig = plt.figure(figsize=(11.8, 3.18))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 0.42, 1.0, 1.0], wspace=0.14)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    legend_ax = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 3])
    ax3 = fig.add_subplot(gs[0, 4])
    legend_ax.axis("off")

    draw_metric_panel(ax0, payloads[0], "ndcg10", "NDCG@10")
    draw_metric_panel(ax1, payloads[0], "hr10", "HR@10")
    draw_metric_panel(ax2, payloads[1], "ndcg10", "NDCG@10")
    draw_metric_panel(ax3, payloads[1], "hr10", "HR@10")

    handles = [
        Patch(facecolor=PURE_COLOR, edgecolor="black", linewidth=1.3, hatch="//", label="w/o SACP (Pure-SI)"),
        Patch(facecolor=ECHO_COLOR, edgecolor="black", linewidth=1.3, label="w/ SACP (EchoRec)"),
    ]
    legend_ax.legend(
        handles=handles,
        frameon=False,
        loc="center",
        ncol=1,
        handlelength=1.35,
        handletextpad=0.4,
        borderaxespad=0.0,
        labelspacing=0.58,
        prop={"weight": "bold", "size": LEGEND_SIZE - 1},
    )

    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_box_aspect(0.72)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.79, bottom=0.24)

    left_pos = ax0.get_position()
    left_pos_2 = ax1.get_position()
    right_pos = ax2.get_position()
    right_pos_2 = ax3.get_position()
    fig.text(
        0.5 * (left_pos.x0 + left_pos_2.x1),
        max(left_pos.y1, left_pos_2.y1) + 0.028,
        f"(a) {payloads[0]['display']}",
        ha="center",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    fig.text(
        0.5 * (right_pos.x0 + right_pos_2.x1),
        max(right_pos.y1, right_pos_2.y1) + 0.028,
        f"(b) {payloads[1]['display']}",
        ha="center",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )

    fig.savefig(out_dir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_plot_style()

    analysis_dir = Path(args.analysis_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = [load_dataset_payload(dataset_name, analysis_dir) for dataset_name in args.datasets]
    for payload in payloads:
        save_single(payload, "ndcg10", out_dir)
        save_single(payload, "hr10", out_dir)

    panel_names = args.panel_datasets if args.panel_datasets is not None else args.datasets[:2]
    if len(panel_names) == 2:
        panel_payloads = [load_dataset_payload(name, analysis_dir) for name in panel_names]
        panel_filename = f"sequence_perturb_panel_{panel_payloads[0]['stem']}_{panel_payloads[1]['stem']}"
        save_panel(panel_payloads, out_dir, panel_filename)
        if {panel_payloads[0]["display"], panel_payloads[1]["display"]} == {"CDs", "Scientific"}:
            save_panel(panel_payloads, out_dir, "sequence_perturb_panel")

    for payload in payloads:
        print(f"{payload['display']}: {payload['source_csv']}")
    print(f"Saved sequence perturbation figures to: {out_dir}")


if __name__ == "__main__":
    main()
