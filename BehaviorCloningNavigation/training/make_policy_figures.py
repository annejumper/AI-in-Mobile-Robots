"""Figures for the write-up, from the recorded walks.

    python3 training/make_policy_figures.py

  results/policy/filmed_walks.png   the two filmed runs: heading, what we did on
                                    each stride, and what the navigator said
  results/policy/on_time.png        instructions on time as data and DAgger were
                                    added, and when guiding the other person

Needs numpy and matplotlib. The bar-chart numbers are recomputed here with the
same leave-one-walk-out / leave-one-person-out tests as train_policy.py.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import train_policy as tp
import walklib

OUT = Path(__file__).resolve().parent / "results" / "policy"
FILMED = [("20261124_004034_Anne", "Anne"), ("20261124_004420_Ani", "Ani")]

# Reference palette (dataviz skill), light mode. "straight" is a recessive
# neutral so the events stand out; every event block also carries a text label.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
COLORS = {"straight": "#dddcd7", "left": "#2a78d6", "right": "#eb6834", "uturn_left": "#6250d6",
          "uturn_right": "#6250d6", "stairs_down": "#1baf7a", "stairs_up": "#eda100", "arrive": "#008300"}
SHORT = {"left": "L", "right": "R", "uturn_left": "U-turn", "uturn_right": "U-turn",
         "stairs_down": "stairs↓", "stairs_up": "step↑", "arrive": "arrive"}
NAMES = {"left": "left", "right": "right", "uturn_left": "U-turn left", "stairs_down": "stairs down",
         "stairs_up": "stairs / step up", "arrive": "arrive", "straight": "straight"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def runs(labels):
    """[(first stride, last stride, label)] runs of equal labels, 1-based."""
    out = []
    for i, lab in enumerate(labels, start=1):
        if out and out[-1][2] == lab:
            out[-1][1] = i
        else:
            out.append([i, i, lab])
    return out


def strip(ax, y, labels, label_events=True):
    """One row of blocks, a block per stride, with a 2px-ish gap between runs."""
    for a, b, lab in runs(labels):
        ax.add_patch(Rectangle((a - 0.5 + 0.08, y - 0.36), b - a + 1 - 0.16, 0.72,
                               facecolor=COLORS[lab], edgecolor="none"))
        fits = (b - a + 1) >= max(1, 0.42 * len(SHORT.get(lab, "")))
        if label_events and lab != "straight" and fits:
            ax.text((a + b) / 2, y, SHORT[lab], ha="center", va="center", fontsize=7.5,
                    color="white" if lab in ("left", "right", "uturn_left", "arrive") else INK,
                    fontweight="bold", clip_on=True)


def filmed_figure():
    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.5, 0.9, 0.55, 1.5, 0.9], hspace=0.12,
                          left=0.1, right=0.985, top=0.9, bottom=0.12)
    axes = [fig.add_subplot(gs[i]) for i in (0, 1, 3, 4)]
    for k, (wid, person) in enumerate(FILMED):
        d = walklib.WALKS_DIR / wid
        with open(d / "labels.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        with open(d / "strides.csv", newline="") as fh:
            board = list(csv.DictReader(fh))
        did = [r["label"] for r in rows]
        heading = [float(r["heading_deg"]) for r in rows]
        # The instruction showing while you walk stride k is what it said after stride k-1
        # (before the first stride: the first stride's own instruction).
        said = [board[0]["said"]] + [b["said"] for b in board[:-1]]
        n = len(did)

        ax = axes[2 * k]
        ax.plot(range(1, n + 1), heading, color=INK2, linewidth=2)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylabel("heading (°)\n+ = left", fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{person}'s filmed run: every instruction on time "
                     f"({'8 / 8' if person == 'Anne' else '7 / 7'})", loc="left", fontsize=11, color=INK,
                     fontweight="bold")

        ax = axes[2 * k + 1]
        strip(ax, 1, did)
        strip(ax, 0, said)
        ax.set_xlim(0.5, n + 0.5)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([1, 0], ["What we did", "Navigator said"], fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("stride (two steps, ≈ 1.3 m)" if k == 1 else "", fontsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    shown = ["left", "right", "uturn_left", "stairs_down", "stairs_up", "arrive", "straight"]
    handles = [Rectangle((0, 0), 1, 1, facecolor=COLORS[s]) for s in shown]
    fig.legend(handles, [NAMES[s] for s in shown], loc="lower center", ncol=len(shown), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.01, 0.965, "The navigator's instruction is aimed up to 2 strides ahead, so its blocks start a little "
             "before ours. After stopping at Nolop we moved for the camera (counted as arrive).",
             fontsize=8.5, color=INK2)
    fig.savefig(OUT / "filmed_walks.png", dpi=200)
    plt.close(fig)


def on_time(train, test):
    model = tp.build(train)
    ok = total = 0
    for w in test:
        _, events, _ = tp.score(w, tp.replay(model, w))
        ok += sum(e[2] == "in time" for e in events)
        total += len(events)
    return ok, total


def leave_one_out(walks):
    ok = total = 0
    for i, w in enumerate(walks):
        a, b = on_time(walks[:i] + walks[i + 1:], [w])
        ok, total = ok + a, total + b
    return ok, total


def on_time_figure():
    walks = tp.load_labeled(False)
    demos = [w for w in walks if w["kind"] == "demo"]
    anne_dagger = [w for w in walks if w["kind"] == "dagger" and w["person"] == "Anne"]
    stages = [("6 demos\n(both of us)", demos), ("+ Anne's\nDAgger walk", demos + anne_dagger),
              ("+ Ani's DAgger walk\n(final, 8 walks)", walks)]
    stage_vals = [leave_one_out(ws) for _, ws in stages]

    cross = []
    for test_p, train_p in (("Anne", "Ani"), ("Ani", "Anne")):
        test = [w for w in walks if w["person"] == test_p]
        cross.append((f"trained on {train_p}'s walks,\nguiding {test_p}",
                      on_time([w for w in demos if w["person"] == train_p], test),
                      on_time([w for w in walks if w["person"] == train_p], test)))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    blue, light_blue = "#2a78d6", "#9cc3f0"

    def bars(ax, xs, vals, color, width=0.6):
        for x, (ok, tot) in zip(xs, vals):
            pct = 100 * ok / tot
            ax.bar(x, pct, width=width, color=color, edgecolor=SURFACE, linewidth=2)
            ax.text(x, pct + 1.5, f"{pct:.0f}%\n({ok}/{tot})", ha="center", va="bottom", fontsize=9, color=INK)

    bars(a1, range(3), stage_vals, blue)
    a1.set_xticks(range(3), [s for s, _ in stages], fontsize=9)
    a1.set_title("Each walk guided by a model that never saw it", loc="left", fontsize=11, color=INK,
                 fontweight="bold")

    xs = [0, 1]
    bars(a2, [x - 0.2 for x in xs], [c[1] for c in cross], light_blue, width=0.38)
    bars(a2, [x + 0.2 for x in xs], [c[2] for c in cross], blue, width=0.38)
    a2.set_xticks(xs, [c[0] for c in cross], fontsize=9)
    a2.set_title("Guiding the partner it wasn't trained on", loc="left", fontsize=11, color=INK,
                 fontweight="bold")
    a2.legend([Rectangle((0, 0), 1, 1, facecolor=light_blue), Rectangle((0, 0), 1, 1, facecolor=blue)],
              ["demonstrations only", "demonstrations + DAgger walk"], frameon=False, fontsize=9,
              labelcolor=INK2, loc="upper right")

    for ax in (a1, a2):
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("instructions on time (%)", fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "on_time.png", dpi=200)
    plt.close(fig)
    return stage_vals, cross


def procedure_figure():
    """Flow of what we did: build -> practice -> demos -> hand labels -> train -> DAgger loop -> tests."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    accent, box_edge = "#2a78d6", "#c9c8c3"
    W, H = 2.15, 1.95

    def box(cx, cy, num, title, body, badge=None, edge=box_edge):
        ax.add_patch(FancyBboxPatch((cx - W / 2, cy - H / 2), W, H, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="white", edgecolor=edge, linewidth=1.6 if edge != box_edge else 1.1))
        y0 = cy + H / 2 - 0.2
        ax.text(cx - W / 2 + 0.14, y0, num, ha="left", va="top", fontsize=9, color=INK2, fontweight="bold")
        ax.text(cx - W / 2 + 0.42, y0, title, ha="left", va="top", fontsize=10, color=INK,
                fontweight="bold", linespacing=1.15)
        body_y = y0 - 0.3 * (title.count("\n") + 1) - 0.18
        ax.text(cx - W / 2 + 0.14, body_y, body, ha="left", va="top", fontsize=8.3, color=INK2, linespacing=1.3)
        if badge:      # under the box, so it never covers the text
            ax.text(cx, cy - H / 2 - 0.2, badge, ha="center", va="top", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.28", facecolor="#eef3fb", edgecolor="none"))

    def arrow(p, q, color=INK2, style="arc3", label=None, label_at=None, lw=1.4):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, linewidth=lw,
                                     connectionstyle=style, shrinkA=2, shrinkB=2))
        if label:
            ax.text(*label_at, label, ha="center", va="center", fontsize=8.5, color=color,
                    fontweight="bold" if color == accent else "normal")

    top, bottom = 5.0, 1.85
    xs = [1.2, 3.55, 5.9, 8.25, 10.6]
    box(xs[0], top, "1", "Step counter\n→ strides",
        "Our step net finds heel\nstrikes; ZUPT gives each\nstride's length & height;\ngyro gives heading")
    box(xs[1], top, "2", "Practice loop",
        "Tested recording on a\nshort loop; fixed heading\ndrift and data lost when\nthe battery was pulled", "not trained on")
    box(xs[2], top, "3", "Record\ndemonstrations",
        "Same start spot, same\ndirection; no labeling\nwhile walking", "3 Anne + 3 Ani")
    box(xs[3], top, "4", "Review & fix\nlabels by hand",
        "Auto-labels from heading\n& height; we fixed stair\nlandings, missed stairs,\nthe U-turn", None)
    box(xs[4], top, "5", "Train & deploy",
        "k-NN over every stride;\neach walk tested on a\nmodel that never saw it;\npushed to the board")
    box(xs[4], bottom, "6", "DAgger walk",
        "The navigator guides us;\nwhen it's wrong we tap\nthe right instruction\nand keep walking", "1 Anne + 1 Ani", edge=accent)
    box(xs[2] + 0.6, bottom, "7", "Filmed tests",
        "Guided by the final\n8-walk navigator;\nnever trained on", "Anne 8/8, Ani 7/7")
    box(xs[0] + 0.6, bottom, "8", "Evaluate",
        "On time: 87% → 94%\nwith DAgger; ~80%\nguiding the partner\nit wasn't trained on")

    for a, b in zip(xs[:-1], xs[1:]):
        arrow((a + W / 2, top), (b - W / 2, top))
    arrow((xs[4], top - H / 2), (xs[4], bottom + H / 2), color=accent)
    # The DAgger loop: corrected walks go back through review and training
    arrow((xs[4] - W / 2, bottom + 0.35), (xs[3], top - H / 2), color=accent, style="arc3,rad=-0.25",
          label="label, retrain,\nrepeat (×2)", label_at=(7.6, 3.05))
    arrow((xs[4] - W / 2, bottom - 0.35), (xs[2] + 0.6 + W / 2, bottom - 0.35),
          label="final model", label_at=((xs[4] - W / 2 + xs[2] + 0.6 + W / 2) / 2, bottom - 0.6))
    arrow((xs[2] + 0.6 - W / 2, bottom), (xs[0] + 0.6 + W / 2, bottom))
    # Rule changes found while testing fed back into labeling
    ax.text(6.0, 6.55, "Problems found in test walks changed the labeling rules (turn only where it starts; "
            "U-turn label) — then every walk was relabeled.", ha="center", fontsize=8.5, color=INK2,
            style="italic")
    ax.text(0.15, 0.12, "8 of the 10 allowed walks were trained on: 6 demonstrations + 2 DAgger walks.",
            fontsize=8.5, color=INK2)
    fig.savefig(OUT / "procedure.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    procedure_figure()
    filmed_figure()
    print("stages:", on_time_figure())
    print(f"Wrote {OUT / 'procedure.png'}, {OUT / 'filmed_walks.png'} and {OUT / 'on_time.png'}")
