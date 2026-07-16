import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch


PURE_COLOR = "#bbe1f8"
ECHO_COLOR = "#0e3e87"
TITLE_SIZE = 18
TICK_SIZE = 15
X_TICK_SIZE = 12
ANNOTATION_SIZE = 11
LEGEND_SIZE = 14
BORDER_LW = 1.45
TICK_LW = 1.25
BAR_WIDTH = 0.50
BAR_OFFSET = 0.31


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "mathtext.default": "bf",
            "font.weight": "bold",
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.dpi": 220,
            "savefig.dpi": 400,
            "hatch.linewidth": 0.42,
        }
    )


def style_axis(ax) -> None:
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
        color="black",
        top=False,
        right=False,
    )


def draw_metric_panel(
    ax,
    labels,
    pure_vals,
    echo_vals,
    metric_name: str,
    annotation_lift: float,
    y_bottom_lift: float,
) -> None:
    x = np.array([0.0, 1.72, 3.44], dtype=np.float32)

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
    ax.set_xlim(-0.78, 4.22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold", fontsize=X_TICK_SIZE)
    if metric_name == "NDCG@10":
        y_bottom, y_top = 0.25 - y_bottom_lift, 0.58
        ax.set_ylim(y_bottom, y_top)
        ax.set_yticks([0.30, 0.40, 0.50])
        text_pad = 0.004
    else:
        y_bottom, y_top = 0.45 - y_bottom_lift, 0.78
        ax.set_ylim(y_bottom, y_top)
        ax.set_yticks([0.50, 0.60, 0.70])
        text_pad = 0.004
    ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()])

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    for x_pos, pure_v, echo_v in zip(x, pure_vals, echo_vals):
        improvement = (echo_v / pure_v - 1.0) * 100.0
        text_y = min(max(pure_v, echo_v) + text_pad + annotation_lift, y_top - text_pad * 1.6)
        ax.text(
            x_pos,
            text_y,
            f"+{improvement:.1f}%",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="bold",
            color="black",
        )


def save_single_panel(
    out_dir: Path,
    filename: str,
    labels,
    pure_vals,
    echo_vals,
    metric_name: str,
    annotation_lift: float = 0.0,
    y_bottom_lift: float = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(2.75, 2.75))
    draw_metric_panel(
        ax,
        labels,
        np.asarray(pure_vals),
        np.asarray(echo_vals),
        metric_name,
        annotation_lift,
        y_bottom_lift,
    )
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
    fig.savefig(out_dir / filename)
    plt.close(fig)


def main() -> None:
    set_plot_style()

    out_dir = Path("d:/New/EchoRec/paper/figure")
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        {
            "prefix": "model_agnostic_cds",
            "labels": ["BERT4Rec", "SASRec", "GRU4Rec"],
            "ndcg_pure": [0.3548622680659638, 0.3655934144856504, 0.3805695426806859],
            "ndcg_echo": [0.4277102803267731, 0.4169341086253188, 0.4192563570199631],
            "hr_pure": [0.5846, 0.5888, 0.6030],
            "hr_echo": [0.6570, 0.6474, 0.6602],
        },
        {
            "prefix": "model_agnostic_movies",
            "labels": ["BERT4Rec", "SASRec", "GRU4Rec"],
            "ndcg_pure": [0.3649589836914872, 0.3649483085178021, 0.3557034774200897],
            "ndcg_echo": [0.3990252311427773, 0.3955558943684408, 0.3924720070540223],
            "hr_pure": [0.5632029994643813, 0.5567755757900374, 0.5538296732726299],
            "hr_echo": [0.6057846813069094, 0.5977504017139796, 0.5918585966791644],
        },
    ]

    for item in plots:
        save_single_panel(
            out_dir,
            f"{item['prefix']}_ndcg10.pdf",
            item["labels"],
            item["ndcg_pure"],
            item["ndcg_echo"],
            "NDCG@10",
            0.012,
            0.06 if item["prefix"].endswith("_movies") else 0.0,
        )
        save_single_panel(
            out_dir,
            f"{item['prefix']}_hr10.pdf",
            item["labels"],
            item["hr_pure"],
            item["hr_echo"],
            "HR@10",
            0.012,
            0.06 if item["prefix"].endswith("_movies") else 0.0,
        )


if __name__ == "__main__":
    main()
