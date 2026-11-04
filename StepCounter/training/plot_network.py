"""Draw the step net as neurons and layers, with the forward pass and backprop.

    .venv/bin/python training/plot_network.py

Writes training/results/steps/network_diagram.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "results" / "steps" / "network_diagram.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
LINE = "#c9c7c1"
BLUE = "#2a78d6"
RED = "#d1453b"
GREEN = "#2f9e5b"
ORANGE = "#e8a33d"

INPUTS = ["ax", "ay", "az", "gx", "gy", "gz", "|a|-1", "|g|"]
# (title, neurons in the layer, neurons drawn, note under it)
LAYERS = [
    ("Input", 8, 8, "8 numbers\nper sample"),
    ("Layer 1", 16, 7, "9 taps\n1 apart"),
    ("Layer 2", 16, 7, "9 taps\n2 apart"),
    ("Layer 3", 16, 7, "9 taps\n4 apart"),
    ("Layer 4", 16, 7, "9 taps\n8 apart"),
    ("Output", 1, 1, "1x1 layer"),
]


def neuron_ys(drawn, top=8.2, bottom=1.8):
    if drawn == 1:
        return [(top + bottom) / 2]
    step = (top - bottom) / (drawn - 1)
    return [top - i * step for i in range(drawn)]


def main():
    fig, ax = plt.subplots(figsize=(13, 10), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-1.2, 15.9)
    ax.set_ylim(-1.9, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")

    xs = [0.6 + i * 2.3 for i in range(len(LAYERS))]
    ys = [neuron_ys(d) for _, _, d, _ in LAYERS]

    # Connections (every neuron to every neuron in the next layer)
    for i in range(len(LAYERS) - 1):
        for y0 in ys[i]:
            for y1 in ys[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [y0, y1], color=LINE, linewidth=0.5, zorder=1)

    for i, ((title, n, drawn, note), x, yl) in enumerate(zip(LAYERS, xs, ys)):
        color = BLUE if 0 < i < len(LAYERS) - 1 else (INK_2 if i == 0 else GREEN)
        for j, y in enumerate(yl):
            if i > 0 and drawn < n and j == drawn // 2:
                ax.text(x, y, "⋮", ha="center", va="center", fontsize=16, color=INK_2)
                continue
            ax.add_patch(Circle((x, y), 0.28, facecolor=SURFACE, edgecolor=color, linewidth=1.8, zorder=3))
            if i == 0:
                ax.text(x - 0.42, y, INPUTS[j], ha="right", va="center", fontsize=9, color=INK_2)
        label = f"{title}\n({n} neuron{'s' if n > 1 else ''})"
        ax.text(x, 9.0, label, ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")
        ax.text(x, 1.0, note, ha="center", va="top", fontsize=9, color=INK_2)

    # Activation boxes between layers
    for i in range(1, len(LAYERS) - 1):
        mid = (xs[i] + xs[i + 1]) / 2
        ax.add_patch(FancyBboxPatch((mid - 0.42, 4.75), 0.84, 0.5, boxstyle="round,pad=0.05",
                                    facecolor=ORANGE, edgecolor="none", zorder=4))
        ax.text(mid, 5.0, "ReLU", ha="center", va="center", fontsize=9, color=INK, weight="bold", zorder=5)

    # Sigmoid and probability
    xo = xs[-1]
    ax.add_patch(FancyBboxPatch((xo + 0.6, 4.7), 1.2, 0.6, boxstyle="round,pad=0.05",
                                facecolor=GREEN, edgecolor="none", zorder=4))
    ax.text(xo + 1.2, 5.0, "sigmoid", ha="center", va="center", fontsize=9.5, color="white", weight="bold", zorder=5)
    ax.annotate("", xy=(xo + 0.58, 5.0), xytext=(xo + 0.3, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5))
    ax.text(xo + 2.05, 5.0, "P(heel strike)\n0 … 1", ha="left", va="center", fontsize=10, color=INK)

    # Loss box
    ax.add_patch(FancyBboxPatch((xo + 1.75, 2.0), 1.6, 1.25, boxstyle="round,pad=0.08",
                                facecolor="#fbe9e7", edgecolor=RED, linewidth=1.2))
    ax.text(xo + 2.55, 2.62, "Loss\ncompare with\nthe true strikes", ha="center", va="center",
            fontsize=9, color=INK)
    ax.annotate("", xy=(xo + 2.55, 3.35), xytext=(xo + 2.55, 4.45),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.2))

    # Forward pass and backprop arrows
    ax.add_patch(FancyArrowPatch((xs[0], 10.35), (xs[-1] + 1.2, 10.35), arrowstyle="-|>",
                                 mutation_scale=18, color=BLUE, linewidth=2.2))
    ax.text((xs[0] + xs[-1]) / 2, 10.55, "1. Forward pass: data flows in, a probability comes out",
            ha="center", va="bottom", fontsize=10.5, color=BLUE, weight="bold")
    ax.add_patch(FancyArrowPatch((xs[-1] + 2.55, -0.35), (xs[0], -0.35), arrowstyle="-|>",
                                 mutation_scale=18, color=RED, linewidth=2.2,
                                 connectionstyle="arc3,rad=0"))
    ax.plot([xs[-1] + 2.55, xs[-1] + 2.55], [1.92, -0.35], color=RED, linewidth=2.2)
    ax.text((xs[0] + xs[-1]) / 2, -0.6,
            "2. Backpropagation: the error flows back, giving every weight its share of the blame (gradient)\n"
            "3. Adam nudges each of the 8,145 weights to shrink the error. Repeat 2,400 times.",
            ha="center", va="top", fontsize=10, color=RED)

    ax.text(-1.1, 10.9, "The step net, at one moment in time", fontsize=15, color=INK,
            weight="bold", va="bottom")
    ax.text(15.8, 10.9, "Each line is a weight. In layers 1-4 each one is really 9 weights,\n"
                        "one per time tap, so a neuron sees 1.21 s of motion around now.",
            fontsize=8.5, color=INK_2, ha="right", va="bottom")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
