"""Train both gait models from the recordings in ../data and export them to the app.

    .venv/bin/python training/train.py

Steps:
  1. Load every data/session_*.csv and split it into labeled stretches. Pause
     and stretches shorter than MIN_SEGMENT_S are dropped, and TRIM_S is cut
     off both ends of each stretch (reaching for the phone, getting going).
  2. Cut each stretch into 2 s windows every HOP_S and compute features with
     gait-identifier/python/features.py (the same code the board runs).
     Windows labeled walk/run/bike where the leg is barely rotating are
     dropped: that's standing still before Pause was tapped, not the activity.
  3. Split each stretch by time: windows in the first TRAIN_FRACTION go to
     training, the rest to testing. Windows never straddle the cut, so no test
     sample was seen in training.
  4. Model 1: a depth-2 decision tree on the peak X acceleration only (up to 4
     thresholds-defined bins, one per activity).
     Model 2: a small decision tree on every feature, compared against KNN.
  5. Print accuracy and confusion matrices, save plots to training/results/,
     and write both trees to gait-identifier/python/model.json for the board.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parent.parent
APP_PY = ROOT / "gait-identifier" / "python"
sys.path.insert(0, str(APP_PY))
from features import (FEATURE_NAMES, PEAK_FEATURE, SAMPLE_HZ, WINDOW_N,  # noqa: E402
                      tree_predict, window_features)

DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_PATH = APP_PY / "model.json"

CLASSES = ["stand", "walk", "run", "bike"]
MIN_SEGMENT_S = 10
TRIM_S = 3
HOP_S = 0.5
# Moving windows average 100+ deg/s of leg rotation, standing about 3 deg/s.
STILL_GYRO_DPS = 20
TRAIN_FRACTION = 0.7
SEED = 0

# Chart colors (light surface), from the dataviz reference palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SEQ_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_segments():
    """[(file name, label, rows)] for every usable labeled stretch."""
    segments = []
    for path in sorted(DATA_DIR.glob("session_*.csv")):
        current, rows = None, []
        with open(path, newline="", errors="replace") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                if len(row) != 8 or "\x00" in "".join(row):
                    continue                      # torn line from a power cut
                label = row[1]
                if label != current:
                    if current is not None:
                        segments.append((path.name, current, rows))
                    current, rows = label, []
                rows.append([float(x) for x in row[2:]])
        if current is not None:
            segments.append((path.name, current, rows))

    trim = int(TRIM_S * SAMPLE_HZ)
    kept = []
    for name, label, rows in segments:
        if label not in CLASSES or len(rows) < MIN_SEGMENT_S * SAMPLE_HZ:
            continue
        kept.append((name, label, rows[trim:-trim]))
    return kept


def make_windows(segments):
    """Feature matrices and labels for the train and test splits."""
    hop = int(HOP_S * SAMPLE_HZ)
    split = {"train": ([], []), "test": ([], [])}
    still = 0
    for _, label, rows in segments:
        cut = int(len(rows) * TRAIN_FRACTION)
        for start in range(0, len(rows) - WINDOW_N + 1, hop):
            end = start + WINDOW_N
            if end <= cut:
                part = "train"
            elif start >= cut:
                part = "test"
            else:
                continue                          # straddles the cut
            f = window_features(rows[start:end])
            if label != "stand" and f["gyr_mag_mean"] < STILL_GYRO_DPS:
                still += 1
                continue
            split[part][0].append([f[n] for n in FEATURE_NAMES])
            split[part][1].append(label)
    print(f"\nDropped {still} still windows labeled as a moving activity")
    return {k: (np.array(x), np.array(y)) for k, (x, y) in split.items()}


def export_tree(clf, feature_names):
    """sklearn tree -> the dict features.tree_predict() walks."""
    t = clf.tree_
    return {
        "classes": [str(c) for c in clf.classes_],
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "features": [feature_names[i] if i >= 0 else None for i in t.feature],
        "thresholds": t.threshold.tolist(),
        "leaf_class": [int(np.argmax(v)) for v in t.value[:, 0, :]],
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)


def plot_confusions(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2), facecolor=SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    for ax, (title, y_true, y_pred) in zip(axes, results):
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        frac = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="#ffffff" if frac[i, j] > 0.55 else INK)
        style(ax)
        ax.set_xticks(range(len(CLASSES)), CLASSES)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted", color=INK_2)
        ax.set_ylabel("Actual", color=INK_2)
        ax.set_title(f"{title}\n{accuracy_score(y_true, y_pred):.0%} of test windows correct",
                     color=INK, fontsize=11, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_peak(X, y, tree, path):
    """Each test window's peak X acceleration, one row per activity, with the
    thresholds Model 1 learned."""
    col = FEATURE_NAMES.index(PEAK_FEATURE)
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style(ax)
    for i, c in enumerate(CLASSES):
        v = X[y == c, col]
        ax.scatter(v, i + rng.uniform(-0.25, 0.25, len(v)), s=14, color=SERIES_1,
                   alpha=0.55, linewidths=0)
    for thr in sorted({t for f, t in zip(tree["features"], tree["thresholds"]) if f}):
        ax.axvline(thr, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(thr, len(CLASSES) - 0.45, f" {thr:.2f} g", color=INK_2, fontsize=9, va="bottom")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_ylim(-0.6, len(CLASSES) - 0.1)
    ax.set_xlabel("Peak acceleration up the leg (X) in a 2 s window, g", color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Model 1 splits activities at these peak-X thresholds (test windows)",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    segments = load_segments()
    if not segments:
        sys.exit(f"No labeled stretches of {MIN_SEGMENT_S}+ s found in {DATA_DIR}")
    print("Labeled stretches used (after trimming):")
    for name, label, rows in segments:
        print(f"  {label:6} {len(rows) / SAMPLE_HZ:6.1f} s  {name}")

    data = make_windows(segments)
    X_tr, y_tr = data["train"]
    X_te, y_te = data["test"]
    print(f"\nWindows: {len(y_tr)} train, {len(y_te)} test")
    for c in CLASSES:
        print(f"  {c:6} {np.sum(y_tr == c):4} train  {np.sum(y_te == c):4} test")
    missing = [c for c in CLASSES if c not in y_tr]
    if missing:
        sys.exit(f"No training data for: {', '.join(missing)}")

    # Model 1: peak X acceleration only
    peak_col = [FEATURE_NAMES.index(PEAK_FEATURE)]
    m1 = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(X_tr[:, peak_col], y_tr)
    p1 = m1.predict(X_te[:, peak_col])

    # Model 2: decision tree on every feature, plus KNN for comparison
    m2 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=SEED).fit(X_tr, y_tr)
    p2 = m2.predict(X_te)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_tr, y_tr)
    pk = knn.predict(X_te)

    results = [
        ("Model 1: peak X accel only", y_te, p1),
        ("Model 2: decision tree, full IMU", y_te, p2),
        ("Comparison: KNN (k=5), full IMU", y_te, pk),
    ]
    print()
    for title, yt, yp in results:
        print(f"{title:36} {accuracy_score(yt, yp):6.1%}")
        cm = confusion_matrix(yt, yp, labels=CLASSES)
        print("    actual \\ predicted  " + " ".join(f"{c:>6}" for c in CLASSES))
        for c, row in zip(CLASSES, cm):
            print(f"    {c:20} " + " ".join(f"{n:6}" for n in row))

    print("\nModel 1 rules:")
    print(export_text(m1, feature_names=[PEAK_FEATURE], decimals=2))
    print("Model 2 rules:")
    print(export_text(m2, feature_names=FEATURE_NAMES, decimals=2))
    top = sorted(zip(m2.feature_importances_, FEATURE_NAMES), reverse=True)[:5]
    print("Model 2's most useful features: " + ", ".join(f"{n} ({w:.0%})" for w, n in top if w > 0))

    tree1 = export_tree(m1, [PEAK_FEATURE])
    tree2 = export_tree(m2, FEATURE_NAMES)

    # The board walks these trees with features.tree_predict(); check it agrees
    # with sklearn on every test window before exporting.
    for x, e1, e2 in zip(X_te, p1, p2):
        feats = dict(zip(FEATURE_NAMES, x))
        assert tree_predict(tree1, feats) == e1, "Model 1 export doesn't match sklearn"
        assert tree_predict(tree2, feats) == e2, "Model 2 export doesn't match sklearn"

    MODEL_PATH.write_text(json.dumps({"peak": tree1, "imu": tree2}, indent=1))
    print(f"\nWrote {MODEL_PATH.relative_to(ROOT)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_confusions(results, RESULTS_DIR / "confusion_matrices.png")
    plot_peak(X_te, y_te, tree1, RESULTS_DIR / "model1_peak_thresholds.png")
    print(f"Wrote plots to {RESULTS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
