"""
How wrong is the network, where, and does the lock policy protect against it?

    python lock/train.py --save models/finger_cnn_heldout.pt     (once: a copy that never sees the held-out recordings)
    python lock/evaluate.py

Evaluates the held-out model on the held-out recordings (whole recordings it never trained on,
the same split train.py uses) and saves to results/:

  eval_confusion.png     confusion matrix, as counts and as % of each true number
  eval_per_class.png     accuracy for each number, laptop pictures vs board pictures
  eval_recordings.png    accuracy for each held-out recording (shows the errors cluster)
  eval_confidence.png    how sure the network was when right vs wrong, with the 80% threshold
  eval_lock_policy.png   what a button press would do: enter the right digit, the wrong one, or be ignored
  eval_summary.txt       the numbers behind the plots

The model on the board (models/finger_cnn.pt) was trained on ALL recordings, so testing it on
them would look better than it really is; that's why a separate held-out copy is evaluated.
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from lock_policy import LockPolicy
from model import load_model, preprocess

# chart styling: one hue for magnitude, two fixed categorical hues, text in ink colors
INK, INK2, MUTED, GRID = "#1d1d1f", "#52514e", "#8a8986", "#e6e4df"
LAPTOP, BOARD = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
RIGHT, WRONG, IGNORED = "#2a78d6", "#eb6834", "#c9c7c0"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load_split():
    sys.argv = sys.argv[:1]
    from train import split            # the exact split used in training
    _, val = split(0.2, 0)
    return val


@torch.no_grad()
def run(model, items):
    rows = []
    for path, label in items:
        p = torch.softmax(model(preprocess(cv2.imread(str(path)))[None]), 1)[0].numpy()
        rows.append({"path": path, "true": label, "pred": int(p.argmax()), "conf": float(p.max()),
                     "source": "board" if path.name.startswith("b") else "laptop",
                     "session": path.name.split("_")[0]})
    return rows


def plot_confusion(rows, out):
    n = len(config.CLASSES)
    cm = np.zeros((n, n), int)
    for r in rows:
        cm[r["true"], r["pred"]] += 1
    pct = cm / cm.sum(1, keepdims=True) * 100
    acc = np.trace(cm) / cm.sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, data, fmt, title in ((axes[0], cm, "{:d}", "Counts"), (axes[1], pct, "{:.0f}%", "% of each true number")):
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(n):
            for j in range(n):
                v = data[i, j]
                if (cm[i, j] == 0):
                    continue
                ax.text(j, i, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="center", fontsize=10,
                        color="white" if pct[i, j] > 55 else INK)
        ax.set(xticks=range(n), yticks=range(n), xlabel="Predicted fingers", ylabel="True fingers", title=title)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Held-out confusion matrix: {acc:.1%} correct on {cm.sum():,} pictures from unseen recordings",
                 x=0.06, y=1.04, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.06, -0.02, "Almost every mistake is one finger off (just above or below the diagonal): "
             "3 vs 4 and 1 vs 2.", color=INK2)
    fig.savefig(out / "eval_confusion.png")
    plt.close(fig)
    return cm, acc


def plot_per_class(rows, out):
    n = len(config.CLASSES)
    acc = {}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.38
    for k, (src, col) in enumerate((("laptop", LAPTOP), ("board", BOARD))):
        vals, ns = [], []
        for c in range(n):
            rs = [r for r in rows if r["true"] == c and r["source"] == src]
            vals.append(np.mean([r["pred"] == c for r in rs]) * 100 if rs else np.nan)
            ns.append(len(rs))
        acc[src] = vals
        x = np.arange(n) + (k - 0.5) * w
        ax.bar(x, np.nan_to_num(vals), w - 0.04, color=col, label=f"{src.capitalize()} camera", zorder=3)
        for xi, v, m in zip(x, vals, ns):
            ax.text(xi, (0 if np.isnan(v) else v) + 1.5, "no held-out\nrecording" if np.isnan(v) else f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8 if np.isnan(v) else 9, color=MUTED if np.isnan(v) else INK2)
    ax.set(xticks=range(n), xticklabels=[f"{c} fingers" for c in config.CLASSES], ylim=(0, 112),
           ylabel="Correct (%)", title="Accuracy for each number, by camera")
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Accuracy for each number, by camera", pad=34)
    ax.legend(frameon=False, loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0))
    ax.tick_params(length=0)
    fig.text(0.02, -0.04, "Each bar is only 1-2 held-out recordings, so one recording with an unusual hand pose can pull a bar down.",
             color=INK2, fontsize=10)
    fig.savefig(out / "eval_per_class.png")
    plt.close(fig)
    return acc


def plot_recordings(rows, out):
    by = defaultdict(list)
    for r in rows:
        by[(r["true"], r["source"], r["session"])].append(r)
    items = sorted(by.items(), key=lambda kv: np.mean([r["pred"] == kv[0][0] for r in kv[1]]))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(items) + 1.2))
    labels = []
    for i, ((c, src, s), rs) in enumerate(items):
        a = np.mean([r["pred"] == c for r in rs]) * 100
        ax.barh(i, a, 0.7, color=LAPTOP if src == "laptop" else BOARD, zorder=3)
        wrong = Counter(r["pred"] for r in rs if r["pred"] != c).most_common(1)
        note = f"{a:.0f}%" + (f"   mostly read as {wrong[0][0]}" if wrong and a < 80 else "")
        ax.text(a + 1, i, note, va="center", fontsize=9, color=INK2)
        labels.append(f"{c} fingers · {src} · {len(rs)} pics")
    ax.set(yticks=range(len(items)), yticklabels=labels, xlim=(0, 130), xlabel="Correct (%)",
           title="Accuracy for each held-out recording")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(length=0)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LAPTOP, label="Laptop camera"), Patch(color=BOARD, label="Board camera")],
              frameon=False, loc="lower left", ncols=2, bbox_to_anchor=(0, 1.0))
    ax.set_title("Accuracy for each held-out recording", pad=30)
    fig.text(0.02, -0.02, "Errors cluster in a few recordings: 14 of the 20 are 89-100% correct.",
             color=INK2)
    fig.savefig(out / "eval_recordings.png")
    plt.close(fig)
    return [(k, np.mean([r["pred"] == k[0] for r in rs])) for k, rs in items]


def plot_confidence(rows, out):
    right = np.array([r["conf"] for r in rows if r["pred"] == r["true"]]) * 100
    wrong = np.array([r["conf"] for r in rows if r["pred"] != r["true"]]) * 100
    thr = LockPolicy.MIN_CONF * 100
    bins = np.linspace(15, 100, 35)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    for ax, data, col, name in ((axes[0], right, RIGHT, "Right answers"), (axes[1], wrong, WRONG, "Wrong answers")):
        ax.hist(data, bins, color=col, edgecolor="white", linewidth=1, zorder=3)
        ax.axvline(thr, color=INK, lw=1.5, ls="--", zorder=4)
        above = (data >= thr).mean() * 100
        ax.set_title(f"{name}: {len(data):,} pictures, {above:.0f}% of them at or above the 80% threshold",
                     fontsize=11.5)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_ylabel("Pictures")
        ax.tick_params(length=0)
    axes[1].set_xlabel("Network's confidence in its answer (%)")
    axes[0].text(thr - 1, axes[0].get_ylim()[1] * 0.92, "lock's 80% threshold ", ha="right", fontsize=9, color=INK2)
    fig.suptitle("How sure was the network when it was right vs wrong?", x=0.06, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "eval_confidence.png")
    plt.close(fig)
    return (right >= thr).mean(), (wrong >= thr).mean(), len(right), len(wrong)


def plot_policy(rows, out):
    """Replay each held-out recording frame by frame through the lock's rule: at every frame,
    would a button press enter the right digit, a wrong digit, or be ignored?"""
    pol = LockPolicy([1, 1, 1])
    by = defaultdict(list)
    for r in rows:
        if r["true"] != 0:
            by[(r["true"], r["session"])].append(r)
    res = {c: Counter() for c in config.CLASSES[1:]}
    for (c, _), rs in by.items():
        rs.sort(key=lambda r: r["path"].name)
        recent = []
        for r in rs:
            recent.append(r["pred"] if (r["conf"] >= pol.MIN_CONF and r["pred"] != 0) else None)
            recent = recent[-pol.RECENT_FRAMES:]
            top, n = Counter(recent).most_common(1)[0]
            cand = top if (top is not None and n >= pol.STEADY_FRAMES) else None
            res[c]["ignored" if cand is None else ("right" if cand == c else "wrong")] += 1
    cls = config.CLASSES[1:]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    left = np.zeros(len(cls))
    for key, col, name in (("right", RIGHT, "Enters the right digit"), ("ignored", IGNORED, "Press ignored (no steady answer)"),
                           ("wrong", WRONG, "Enters a wrong digit")):
        vals = np.array([res[c][key] / max(1, sum(res[c].values())) * 100 for c in cls])
        ax.barh(range(len(cls)), vals, 0.62, left=left, color=col, label=name, edgecolor="white", linewidth=2, zorder=3)
        for i, (l, v) in enumerate(zip(left, vals)):
            if v >= 7:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color="white" if key != "ignored" else INK)
        left += vals
    ax.set(yticks=range(len(cls)), yticklabels=[f"Showing {c}" for c in cls], xlim=(0, 100),
           xlabel="Share of moments in the held-out recordings (%)",
           title="What a button press would do, using the lock's rule (80% sure, 4 of last 6 frames agree)")
    ax.invert_yaxis()
    ax.tick_params(length=0)
    ax.legend(frameon=False, loc="upper left", ncols=3, bbox_to_anchor=(0, -0.16), fontsize=9.5)
    fig.savefig(out / "eval_lock_policy.png")
    plt.close(fig)
    tot = Counter()
    for c in cls:
        tot.update(res[c])
    return res, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=config.MODEL_PATH.parent / "finger_cnn_heldout.pt")
    args = ap.parse_args()
    if not args.model.exists():
        raise SystemExit(f"No {args.model}: run  python lock/train.py --save {args.model}  first")
    out = config.RESULTS_DIR
    out.mkdir(exist_ok=True)

    model = load_model(args.model, "cpu")
    rows = run(model, load_split())
    cm, acc = plot_confusion(rows, out)
    per_class = plot_per_class(rows, out)
    recs = plot_recordings(rows, out)
    r_above, w_above, n_right, n_wrong = plot_confidence(rows, out)
    res, tot = plot_policy(rows, out)

    off_by_one = sum(cm[i, j] for i in range(6) for j in range(6) if abs(i - j) == 1) / max(1, cm.sum() - np.trace(cm))
    lines = [
        f"held-out pictures: {len(rows)} from {len({(r['true'], r['session']) for r in rows})} recordings never used in training",
        f"accuracy: {acc:.1%}",
        f"mistakes that are one finger off: {off_by_one:.0%}",
        f"right answers at >= 80% confidence: {r_above:.0%} of {n_right};  wrong answers at >= 80%: {w_above:.0%} of {n_wrong}",
        "per number (laptop / board): " + ", ".join(
            f"{c}: {per_class['laptop'][c]:.0f}% / " + ("-" if np.isnan(per_class['board'][c]) else f"{per_class['board'][c]:.0f}%")
            for c in config.CLASSES),
        "button press outcome over all held-out moments showing 1-5: " + ", ".join(
            f"{k} {v / sum(tot.values()):.0%}" for k, v in tot.items()),
        "worst recordings: " + "; ".join(f"{c} fingers {src} {s}: {a:.0%}" for (c, src, s), a in recs[:3]),
    ]
    (out / "eval_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nplots in {out}")


if __name__ == "__main__":
    main()
