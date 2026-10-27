"""Draw the two decision trees in gait-identifier/python/model.json as PNGs.

    .venv/bin/python training/plot_trees.py

Writes training/results/model1_peak_tree.png and model2_imu_tree.png.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "gait-identifier" / "python" / "model.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#c9c8c3"
# Same activity colors as the phone page
ACTIVITY_COLORS = {"stand": "#2a78d6", "walk": "#eb6834", "run": "#1baf7a", "bike": "#eda100"}
ACTIVITY_TEXT = {"stand": "white", "walk": "white", "run": "white", "bike": INK}

# Plain-language names for the features the trees use
FEATURE_TEXT = {
    "ax_max": ("Peak acceleration up the leg (X)", "g"),
    "ay_min": ("Lowest forward/back acceleration (Y)", "g"),
    "az_max": ("Peak sideways acceleration (Z)", "g"),
    "gx_max": ("Peak rotation about the leg (X)", "°/s"),
    "gz_std": ("Spread of leg-swing rotation (Z)", "°/s"),
}

TITLES = {
    "peak": ("Model 1: peak acceleration only", "Decision tree on one feature, depth 2 (96.3% on test windows)"),
    "imu": ("Model 2: full IMU", "Decision tree on all features, depth limit 4 (98.2% on test windows)"),
}


def layout(tree):
    """x, y for every node: leaves spread evenly left to right, parents centered over children."""
    pos, next_x = {}, [0]

    def place(n, depth):
        if tree["left"][n] == -1:
            pos[n] = (next_x[0], -depth)
            next_x[0] += 1
        else:
            place(tree["left"][n], depth + 1)
            place(tree["right"][n], depth + 1)
            pos[n] = ((pos[tree["left"][n]][0] + pos[tree["right"][n]][0]) / 2, -depth)

    place(0, 0)
    return pos


def draw(tree, name, path):
    pos = layout(tree)
    n_leaves = sum(1 for l in tree["left"] if l == -1)
    depth = -min(y for _, y in pos.values())
    fig, ax = plt.subplots(figsize=(max(9, 2.6 * n_leaves), 2.0 + 1.9 * depth))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    box_w, box_h = 0.96, 0.5          # leaves sit 1 unit apart

    for n, (x, y) in pos.items():
        for child, answer in ((tree["left"][n], "yes"), (tree["right"][n], "no")):
            if child == -1:
                continue
            cx, cy = pos[child]
            ax.annotate("", xy=(cx, cy + box_h / 2), xytext=(x, y - box_h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=GRID, lw=1.6))
            ax.text((x + cx) / 2, (y + cy) / 2, answer, ha="center", va="center", fontsize=10,
                    color=INK_2, bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))

    for n, (x, y) in pos.items():
        if tree["left"][n] == -1:
            act = tree["classes"][tree["leaf_class"][n]]
            fc, ec, tc = ACTIVITY_COLORS[act], ACTIVITY_COLORS[act], ACTIVITY_TEXT[act]
            label, size, weight = act.upper(), 13, "bold"
        else:
            feat = tree["features"][n]
            text, unit = FEATURE_TEXT.get(feat, (feat, ""))
            fc, ec, tc = "white", INK_2, INK
            label = f"{text}\n{feat} ≤ {tree['thresholds'][n]:.2f} {unit}?"
            size, weight = 9, "normal"
        w = box_w if tree["left"][n] != -1 else box_w * 0.75
        ax.add_patch(FancyBboxPatch((x - w / 2, y - box_h / 2), w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=size, color=tc, weight=weight)

    xs = [x for x, _ in pos.values()]
    ax.set_xlim(min(xs) - box_w * 0.6, max(xs) + box_w * 0.6)
    ax.set_ylim(-depth - 0.6, 0.6)
    title, subtitle = TITLES[name]
    fig.suptitle(title, fontsize=15, weight="bold", color=INK, x=0.03, ha="left", y=0.98)
    fig.text(0.03, 0.905, subtitle + ". Each 2 s window starts at the top; "
             "follow yes/no until you reach an activity.", fontsize=10, color=INK_2)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.03)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    models = json.loads(MODEL_PATH.read_text())
    RESULTS_DIR.mkdir(exist_ok=True)
    files = {"peak": "model1_peak_tree.png", "imu": "model2_imu_tree.png"}
    for name, tree in models.items():
        draw(tree, name, RESULTS_DIR / files[name])


if __name__ == "__main__":
    main()
