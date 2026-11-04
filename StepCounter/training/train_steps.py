"""Train the step-counting neural net in PyTorch, test it, and export it to the board.

    .venv/bin/python training/train_steps.py

Data (all from data/, copied off the board):
  - counted trials (data/trials.csv + the samples tagged with that trial):
    walks of a step count you counted yourself, some over a measured
    distance. Their heel strikes come from the gyro rule in labels.py, kept
    only if it agrees with your count. These are also the test set.
  - walk / run recordings that aren't trials: labeled by the same rule
    without a count to check against (extra training examples).
  - stand / bike recordings: "no steps here" examples.

The net (StepCNN) reads 4 s of the 8 IMU input channels and outputs a
heel-strike probability for every sample. Training target: a narrow bump
(Gaussian, TARGET_SIGMA samples wide) at each labeled strike, 0 elsewhere.
Loss: binary cross-entropy. Optimizer: Adam.

Testing:
  - with counted trials: K-fold cross-validation by trial. Each trial is
    counted by a net that never saw it, and compared with your count.
    The rule-based peak detector (tuned on the same training folds) is the
    baseline. Distance models are fitted and tested the same way.
  - with no trials yet: each recording is split by time (first 70% train,
    last 30% test) and the net is compared against the rule's labels
    (a sanity check only: that's not real ground truth).

Outputs:
  step-counter/python/stepnet.json   weights + fitted distance models (board)
  training/results/steps/            plots, per-trial table, training history
                                     (the animation is made from the history)
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
APP_PY = ROOT / "step-counter" / "python"
sys.path.insert(0, str(APP_PY))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stepnet  # noqa: E402
from labels import (DATA_DIR, SWING_DPS, SWING_DPS_RANGE, heel_strikes,  # noqa: E402
                    LEGACY_PERSON, label_trial, load_session, load_trials)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "steps"
MODEL_PATH = APP_PY / "stepnet.json"

CHANNELS = 16
KERNEL = 9
DILATIONS = [1, 2, 4, 8]       # each layer looks twice as far: 1.2 s of context in all
WINDOW = 400                   # 4 s training crops
TARGET_SIGMA = 3.0             # samples
EPOCHS = 40
BATCHES_PER_EPOCH = 60
BATCH = 32
LR = 3e-3
TRAIN_FRACTION = 0.7
MIN_SEGMENT_S = 8
K_FOLDS = 5
EARLY_SNAPSHOTS = {1, 2, 3, 4, 6, 8, 11, 15, 20, 26, 33, 41, 50}   # batches into epoch 1
SEED = 0

# Chart colors (light surface), from the dataviz reference palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5cd9"]


class StepCNN(nn.Module):
    """Fully convolutional: one output per input sample, so any length works."""

    def __init__(self):
        super().__init__()
        layers, ch = [], stepnet.N_INPUTS
        for d in DILATIONS:
            layers.append(nn.Conv1d(ch, CHANNELS, KERNEL, dilation=d, padding=d * (KERNEL - 1) // 2))
            ch = CHANNELS
        self.convs = nn.ModuleList(layers)
        self.out = nn.Conv1d(CHANNELS, 1, 1)

    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        return self.out(x)[:, 0]                 # logits, (batch, time)

    def export(self):
        convs = list(self.convs) + [self.out]
        return [{"weight": c.weight.detach().numpy().round(6).tolist(),
                 "bias": c.bias.detach().numpy().round(6).tolist(),
                 "dilation": c.dilation[0]} for c in convs]


CONTEXT = sum(d * (KERNEL - 1) // 2 for d in DILATIONS)


# ---------- data ----------

class Segment:
    """A stretch of samples with its heel strikes (empty for stand/bike)."""

    def __init__(self, x, strikes, kind, source, trial=None, person=LEGACY_PERSON):
        self.x, self.strikes, self.kind, self.source, self.trial = x, np.asarray(strikes, int), kind, source, trial
        self.person = person
        self.target = np.zeros(len(x), dtype=np.float32)
        t = np.arange(len(x))
        for s in self.strikes:
            lo, hi = max(0, s - 12), min(len(x), s + 13)
            self.target[lo:hi] = np.maximum(self.target[lo:hi],
                                            np.exp(-0.5 * ((t[lo:hi] - s) / TARGET_SIGMA) ** 2))


def runs(mask):
    """(start, end) of every run of True in a boolean array."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def load_all():
    """(trial segments, other segments)."""
    trial_meta = {(t["session"], t["trial"]): t for t in load_trials()}
    trials, others = [], []
    for path in sorted(DATA_DIR.glob("session_*.csv")):
        labels, trial_ids, people, x = load_session(path)
        for tid in sorted(set(trial_ids) - {0}):
            meta = trial_meta.get((path.name, tid))
            if meta is None:
                continue                         # stopped without entering a count
            seg_x = x[trial_ids == tid]
            strikes = label_trial(seg_x, meta["counted_steps"])
            seg = Segment(seg_x, strikes if strikes is not None else [], meta["kind"], path.name, meta,
                          meta["person"])
            seg.verified = strikes is not None
            trials.append(seg)
        for person in sorted(set(people)):
            for label in ("walk", "run", "stand", "bike"):
                for a, b in runs((labels == label) & (trial_ids == 0) & (people == person)):
                    if b - a < MIN_SEGMENT_S * stepnet.SAMPLE_HZ:
                        continue
                    a, b = a + 100, b - 100       # reaching for the phone
                    seg_x = x[a:b]
                    strikes = heel_strikes(seg_x) if label in ("walk", "run") else []
                    others.append(Segment(seg_x, strikes, label, path.name, person=person))
    return trials, others


def time_split(segments):
    train, test = [], []
    for s in segments:
        cut = int(len(s.x) * TRAIN_FRACTION)
        train.append(Segment(s.x[:cut], s.strikes[s.strikes < cut], s.kind, s.source, person=s.person))
        test.append(Segment(s.x[cut:], s.strikes[s.strikes >= cut] - cut, s.kind, s.source, person=s.person))
    return train, test


# ---------- training ----------

def augment(x, rng):
    """Random strap rotation about the leg (X), gain and noise."""
    x = x.copy()
    a = np.radians(rng.uniform(-20, 20))
    c, s = np.cos(a), np.sin(a)
    for i, j in ((1, 2), (4, 5)):
        x[:, i], x[:, j] = c * x[:, i] - s * x[:, j], s * x[:, i] + c * x[:, j]
    x *= rng.uniform(0.9, 1.1)
    x[:, :3] += rng.normal(0, 0.01, (len(x), 3))
    x[:, 3:] += rng.normal(0, 1.0, (len(x), 3))
    return x


def make_batch(segments, weights, rng):
    xs, ys = [], []
    for i in rng.choice(len(segments), BATCH, p=weights):
        seg = segments[i]
        if len(seg.x) <= WINDOW:
            start, x, y = 0, seg.x, seg.target
            pad = WINDOW - len(x)
            x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
            y = np.pad(y, (0, pad))
        else:
            start = rng.integers(0, len(seg.x) - WINDOW)
            x, y = seg.x[start:start + WINDOW], seg.target[start:start + WINDOW]
        xs.append(stepnet.net_inputs(augment(x, rng)))
        ys.append(y)
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))


def loss_fn(logits, target):
    # Edges don't see their full context (the board never uses them either).
    logits, target = logits[:, CONTEXT:-CONTEXT], target[:, CONTEXT:-CONTEXT]
    weight = 1.0 + 9.0 * target              # strikes are rare: make them count
    return nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight)


def train(segments, val_segments=None, history=None, seed=SEED, log=True):
    """Train a fresh StepCNN. If history is a dict, per-epoch snapshots go in it."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = StepCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    # Sample segments by length, but give stand/bike a fair share of batches
    lengths = np.array([len(s.x) for s in segments], float)
    boost = np.array([2.0 if s.kind in ("stand", "bike") else 1.0 for s in segments])
    weights = lengths * boost / (lengths * boost).sum()
    val = make_batch(val_segments, np.full(len(val_segments), 1 / len(val_segments)),
                     np.random.default_rng(123)) if val_segments else None
    for epoch in range(EPOCHS + 1):
        if epoch:                            # epoch 0 = the untrained net
            model.train()
            total, grads = 0.0, np.zeros(len(DILATIONS) + 1)
            for batch in range(1, BATCHES_PER_EPOCH + 1):
                x, y = make_batch(segments, weights, rng)
                loss = loss_fn(model(x), y)
                opt.zero_grad()
                loss.backward()              # backpropagation: d(loss)/d(every weight)
                g = [c.weight.grad.norm().item() for c in list(model.convs) + [model.out]]
                grads += g
                opt.step()                   # nudge every weight downhill
                total += loss.item()
                # Most of the learning happens in the first epoch, so the
                # animation gets snapshots after individual batches there.
                if history is not None and epoch == 1 and batch in EARLY_SNAPSHOTS:
                    snapshot(model, history, batch / BATCHES_PER_EPOCH, loss.item(), None, np.array(g))
            sched.step()
            train_loss = total / BATCHES_PER_EPOCH
        else:
            with torch.no_grad():
                x, y = make_batch(segments, weights, rng)
                train_loss, grads = loss_fn(model(x), y).item(), np.zeros(len(DILATIONS) + 1)
        val_loss = None
        if val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(val[0]), val[1]).item()
        if history is not None:
            snapshot(model, history, epoch, train_loss, val_loss, grads / BATCHES_PER_EPOCH)
        if log and (epoch % 10 == 0 or epoch == EPOCHS):
            print(f"  epoch {epoch:3d}  train loss {train_loss:.4f}"
                  + (f"  val loss {val_loss:.4f}" if val_loss is not None else ""))
    model.eval()
    return model


def to_stepnet(model, extra=None):
    return stepnet.StepNet({"layers": model.export(), **(extra or {})})


def count_steps(net, x):
    return 2 * len(stepnet.find_strikes(net.probabilities(x), net.threshold))


def tune_rule(train_trials):
    """Swing threshold for the baseline rule that best matches training counts."""
    if not train_trials:
        return SWING_DPS
    errs = [np.mean([abs(2 * len(heel_strikes(t.x, th)) - t.trial["counted_steps"])
                     for t in train_trials]) for th in SWING_DPS_RANGE]
    return int(SWING_DPS_RANGE[int(np.argmin(errs))])


# ---------- distance ----------

def stride_terms(x, strikes):
    """Per strike: (kind, a coefficient, b coefficient, ZUPT m).

    kind is "half" (first stride after standing), "walk" or "run". Its
    step-length model's length is A·a + B·b; ZUPT is only measured on walking
    strides.
    """
    out = []
    for i, s in enumerate(strikes):
        prev = strikes[i - 1] if i else None
        if prev is not None and (s - prev) * stepnet.DT <= stepnet.MAX_STRIDE_S \
                and s + stepnet.stride_samples_needed(prev, s) <= len(x):
            cadence = 2.0 / ((s - prev) * stepnet.DT)
            if stepnet.is_running(x, prev, s):
                out.append(("run", 2.0, 2.0 * cadence, None))
            else:
                out.append(("walk", 2.0, 2.0 * cadence, stepnet.stride_length(x, prev, s)[1]))
        else:
            cadence = 2.0 / stepnet.TYPICAL_STRIDE_S
            out.append(("half", 1.0, cadence, None))     # half a typical stride
    return out


def trial_distance_terms(net, seg):
    """(Aw, Bw, Ar, Br, Z, Ah, Bh): step-length coefficients of the walking
    (incl. half) and running strides, the walking strides' raw ZUPT metres,
    and the half strides' coefficients (ZUPT can't measure those)."""
    strikes = stepnet.find_strikes(net.probabilities(seg.x), net.threshold)
    terms = stride_terms(seg.x, strikes)

    def total(kinds, col):
        return sum(t[col] for t in terms if t[0] in kinds)
    return (total(("walk", "half"), 1), total(("walk", "half"), 2),
            total(("run",), 1), total(("run",), 2),
            total(("walk",), 3), total(("half",), 1), total(("half",), 2))


def fit_distance(rows):
    """rows: [(*trial_distance_terms, true m)] -> walking model {a, b},
    running model {a, b}, zupt_scale.

    The walking model and ZUPT's scale are fitted on trials with no running
    strides. The running model keeps the walking slope (every run was at
    about the same pace, so a slope of its own can't be fitted) and fits its
    own intercept to what's left of each run's distance.
    """
    rows = [r for r in rows if r[0] + r[2] > 0]   # the net found no strides: nothing to fit
    default = dict(stepnet.DEFAULT_STEP_MODEL)
    walks = [r for r in rows if r[2] == 0]
    runs = [r for r in rows if r[2] > 0]
    if not walks:
        return default, default, 1.0
    M = np.array([[r[0], r[1]] for r in walks])
    d = np.array([r[7] for r in walks])
    cad = np.array([r[1] / max(r[0], 1e-9) for r in walks])
    a = b = None
    if len(walks) >= 3 and np.ptp(cad) > 0.3:
        a, b = np.linalg.lstsq(M, d, rcond=None)[0]
    if b is None or b < 0:
        # Not enough different speeds to fit a slope (or the fit says faster
        # steps are shorter, which is noise): keep the typical slope.
        b = default["b"]
        a = float(np.sum((d - M[:, 1] * b) * M[:, 0]) / np.sum(M[:, 0] ** 2))
    model = {"a": round(float(a), 4), "b": round(float(b), 4)}

    run_model = dict(model)
    if runs:
        Ar = np.array([r[2] for r in runs])
        rest = np.array([r[7] - r[0] * a - r[1] * b - r[3] * b for r in runs])
        run_model = {"a": round(float(np.sum(rest * Ar) / np.sum(Ar ** 2)), 4), "b": model["b"]}

    z = np.array([r[4] for r in walks])
    rest = d - np.array([r[5] * a + r[6] * b for r in walks])
    scale = float(np.sum(rest * z) / np.sum(z * z)) if np.sum(z * z) > 0 else 1.0
    return model, run_model, round(scale, 4)


def predict_distance(row, model, run_model, scale):
    Aw, Bw, Ar, Br, Z, Ah, Bh = row[:7]
    run_m = Ar * run_model["a"] + Br * run_model["b"]
    cadence_m = Aw * model["a"] + Bw * model["b"] + run_m
    zupt_m = Z * scale + Ah * model["a"] + Bh * model["b"] + run_m
    return cadence_m, zupt_m


# ---------- history for the animation ----------

DEMO = {}


def snapshot(model, history, epoch, train_loss, val_loss, grads):
    if "x" not in DEMO:
        return
    with torch.no_grad():
        logits = model(torch.tensor(stepnet.net_inputs(DEMO["x"])[None]))[0]
    history.setdefault("epoch", []).append(epoch)
    history.setdefault("train_loss", []).append(train_loss)
    history.setdefault("val_loss", []).append(np.nan if val_loss is None else val_loss)
    history.setdefault("grad_norms", []).append(grads)
    history.setdefault("demo_prob", []).append(torch.sigmoid(logits).numpy())
    history.setdefault("conv1_w", []).append(model.convs[0].weight.detach().numpy().copy())


# ---------- plots ----------

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def plot_example(net, seg, path, title):
    n = min(len(seg.x), 1000)
    x, prob = seg.x[:n], net.probabilities(seg.x)[:n]
    found = stepnet.find_strikes(prob, net.threshold)
    t = np.arange(n) / stepnet.SAMPLE_HZ
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    axs[0].plot(t, np.linalg.norm(x[:, :3], axis=1), color=SERIES[0], lw=1.2)
    axs[0].set_ylabel("|accel| (g)", color=INK_2)
    axs[1].plot(t, x[:, 5], color=SERIES[1], lw=1.2)
    axs[1].set_ylabel("gyro Z (°/s)", color=INK_2)
    axs[2].plot(t, seg.target[:n], color=INK_2, lw=1, ls="--", label="target (labeled strikes)")
    axs[2].plot(t, prob, color=SERIES[2], lw=1.6, label="net output")
    axs[2].plot(np.array(found) / stepnet.SAMPLE_HZ, prob[found], "o", color=INK, ms=5,
                label=f"counted strikes ({len(found)} → {2 * len(found)} steps)")
    axs[2].set_ylabel("P(heel strike)", color=INK_2)
    axs[2].set_xlabel("time (s)", color=INK_2)
    axs[2].legend(loc="upper right", frameon=False, fontsize=9)
    for a in axs:
        style(a)
        for s in seg.strikes[seg.strikes < n]:
            a.axvline(s / stepnet.SAMPLE_HZ, color=GRID, lw=1, zorder=0)
    axs[0].set_title(title, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.plot(history["epoch"], history["train_loss"], color=SERIES[0], lw=2, label="training")
    if not np.all(np.isnan(history["val_loss"])):
        ax.plot(history["epoch"], history["val_loss"], color=SERIES[1], lw=2, label="held out")
    ax.set_xlabel("epoch", color=INK_2)
    ax.set_ylabel("loss (weighted BCE)", color=INK_2)
    ax.set_title("Training loss", color=INK, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_trials(rows, path):
    """Counted vs network vs baseline for every held-out trial."""
    kinds = sorted({r["kind"] for r in rows})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    top = max(max(r["counted"], r["nn"], r["rule"]) for r in rows) * 1.08
    ax = axs[0]
    ax.plot([0, top], [0, top], color=INK_2, lw=1, ls="--")
    for i, k in enumerate(kinds):
        rs = [r for r in rows if r["kind"] == k]
        ax.plot([r["counted"] for r in rs], [r["nn"] for r in rs], "o", color=SERIES[i % len(SERIES)], label=k, ms=7)
    ax.set_xlabel("steps counted by hand", color=INK_2)
    ax.set_ylabel("steps counted by the net", color=INK_2)
    ax.set_title("Each held-out trial", color=INK, loc="left")
    ax.legend(frameon=False)
    style(ax)
    ax = axs[1]
    groups = kinds + ["all"]
    for j, (key, name) in enumerate((("nn_err_pct", "neural net"), ("rule_err_pct", "gyro rule"))):
        means, sds = [], []
        for g in groups:
            e = [r[key] for r in rows if g == "all" or r["kind"] == g]
            means.append(np.mean(e))
            sds.append(np.std(e))
        xs = np.arange(len(groups)) + (j - 0.5) * 0.38
        ax.bar(xs, means, 0.36, yerr=sds, capsize=3, color=SERIES[j], label=name, error_kw={"ecolor": INK_2})
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel("step count error (%), mean ± SD", color=INK_2)
    ax.set_title("Error by trial type", color=INK, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def summarize(name, errs, counts):
    errs, counts = np.asarray(errs, float), np.asarray(counts, float)
    pct = 100 * errs / counts
    rng = np.random.default_rng(0)
    boot = [np.mean(np.abs(rng.choice(pct, len(pct)))) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:11s} error {errs.mean():+.1f} ± {errs.std():.1f} steps   "
          f"{pct.mean():+.1f} ± {pct.std():.1f} %   mean |error| {np.abs(pct).mean():.1f} % "
          f"(95% CI {lo:.1f}–{hi:.1f} %)")


# ---------- cross-validation ----------

def cross_validate(trials, others, folds, name, held_out=None):
    """Count every trial with a net trained without its fold. Returns one row per trial.

    held_out[i], if given, is a person whose other recordings are also left
    out of fold i's training data.
    """
    rows = []
    for fold, test_idx in enumerate(folds):
        train_t = [t for i, t in enumerate(trials) if i not in test_idx and t.verified]
        train_o = [s for s in others if held_out is None or s.person != held_out[fold]]
        print(f"{name}, fold {fold + 1}/{len(folds)}: training on {len(train_t)} trials + other data"
              + (f" (without {held_out[fold]})" if held_out else ""))
        net = to_stepnet(train(train_t + train_o, log=False, seed=SEED + fold))
        rule_th = tune_rule(train_t)
        fit_rows = [(*trial_distance_terms(net, t), t.trial["distance_m"])
                    for i, t in enumerate(trials) if t.trial["distance_m"] and i not in test_idx]
        step_model, run_model, scale = fit_distance(fit_rows)
        for i in sorted(test_idx):
            t = trials[i]
            n = t.trial["counted_steps"]
            nn_steps = count_steps(net, t.x)
            rule_steps = 2 * len(heel_strikes(t.x, rule_th))
            row = {"person": t.person, "session": t.source, "trial": t.trial["trial"], "kind": t.trial["kind"],
                   "counted": n, "nn": nn_steps, "rule": rule_steps,
                   "nn_err": nn_steps - n, "rule_err": rule_steps - n,
                   "nn_err_pct": 100 * (nn_steps - n) / n, "rule_err_pct": 100 * (rule_steps - n) / n,
                   "distance_m": t.trial["distance_m"] or ""}
            if t.trial["distance_m"]:
                terms = trial_distance_terms(net, t)
                c_m, z_m = predict_distance(terms, step_model, run_model, scale)
                row.update(cadence_m=round(c_m, 2), zupt_m=round(z_m, 2))
            rows.append(row)
    return rows


def report(rows, heading, stem):
    """Print the step and distance errors in rows, and save them as <stem>.csv/.png."""
    with open(RESULTS_DIR / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["cadence_m", "zupt_m"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{heading}:")
    groups = [("kind", k) for k in sorted({r["kind"] for r in rows})]
    people = sorted({r["person"] for r in rows})
    if len(people) > 1:
        groups += [("person", p) for p in people]
    for key, value in groups + [(None, "all")]:
        rs = [r for r in rows if key is None or r[key] == value]
        print(f" {value} ({len(rs)} trials)")
        summarize("neural net", [r["nn_err"] for r in rs], [r["counted"] for r in rs])
        summarize("gyro rule", [r["rule_err"] for r in rs], [r["counted"] for r in rs])
    plot_trials(rows, RESULTS_DIR / f"{stem}.png")
    dist_rows = [r for r in rows if r["distance_m"] != ""]
    if dist_rows:
        print(" distance:")
        kinds = sorted({r["kind"] for r in dist_rows})
        for kind in kinds + ["all"]:
            rs = [r for r in dist_rows if kind in ("all", r["kind"])]
            print(f"  {kind} ({len(rs)} trials)")
            for key, name in (("cadence_m", "steps × step length"), ("zupt_m", "ZUPT integration")):
                e = np.array([100 * (r[key] - r["distance_m"]) / r["distance_m"] for r in rs])
                print(f"   {name:20s} error {e.mean():+.1f} ± {e.std():.1f} %  mean |error| {np.abs(e).mean():.1f} %")


# ---------- main ----------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials, others = load_all()
    good = [t for t in trials if t.verified]
    print(f"Counted trials: {len(trials)} ({len(good)} where the gyro rule matched the count)")
    for t in trials:
        if not t.verified:
            print(f"  ! trial {t.source} #{t.trial['trial']} ({t.trial['kind']}, "
                  f"{t.trial['counted_steps']} steps): rule couldn't match the count; test only")
    for kind in ("walk", "run", "stand", "bike"):
        segs = [s for s in others if s.kind == kind]
        print(f"Other {kind:5s}: {sum(len(s.x) for s in segs) / 100:6.0f} s, "
              f"{sum(len(s.strikes) for s in segs)} rule-labeled strikes")

    demo_pool = good or [s for s in others if s.kind == "walk"]
    demo = max(demo_pool, key=lambda s: len(s.strikes))
    # 6 s of steady walking from the middle of it, for the plots and animation
    mid = demo.strikes[len(demo.strikes) // 2] if len(demo.strikes) else len(demo.x) // 2
    demo_at = int(np.clip(mid - 300, 0, max(0, len(demo.x) - 600)))
    DEMO["x"] = demo.x[demo_at:demo_at + 600]
    history = {}
    extra = {"threshold": stepnet.STRIKE_THRESHOLD, "trained_on": []}

    if trials:
        # ----- K-fold cross-validation by trial -----
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(trials))
        k = min(K_FOLDS, len(trials))
        rows = cross_validate(trials, others, [set(order[f::k]) for f in range(k)], "by trial")
        report(rows, "Step count on held-out trials (cross-validated by trial)", "trials_eval")
        people = sorted({t.person for t in trials})
        if len(people) >= 2:
            # Harder test: every person is counted by a net that never saw them
            folds = [{i for i, t in enumerate(trials) if t.person == p} for p in people]
            rows = cross_validate(trials, others, folds, "leave one person out", held_out=people)
            report(rows, "Step count on people the net never saw (leave one person out)", "people_eval")
        final_train = good + others
    else:
        print("\nNo counted trials yet: time-split sanity check against the rule's labels.")
        train_segs, test_segs = time_split(others)
        model = train(train_segs, val_segments=test_segs, history=history)
        net = to_stepnet(model)
        print("\nHeld-out last 30% of each recording (rule labels, not hand counts):")
        for kind in ("walk", "run", "stand", "bike"):
            segs = [s for s in test_segs if s.kind == kind]
            if not segs:
                continue
            rule = sum(2 * len(s.strikes) for s in segs)
            got = sum(count_steps(net, s.x) for s in segs)
            print(f"  {kind:5s} rule {rule:4d} steps   net {got:4d} steps")
        plot_example(net, max((s for s in test_segs if s.kind == "walk"), key=lambda s: len(s.x)),
                     RESULTS_DIR / "example_walk.png", "Held-out walking")
        final_train = train_segs

    if trials:
        # ----- final model on everything, with the history for the animation -----
        print("\nFinal model:")
        model = train(final_train, history=history)
        net = to_stepnet(model)
    dist = [(*trial_distance_terms(net, t), t.trial["distance_m"]) for t in trials if t.trial["distance_m"]]
    step_model, run_model, scale = fit_distance(dist)
    print(f"  walking step length = {step_model['a']:.3f} + {step_model['b']:.3f} × cadence (steps/s) m;"
          f" running {run_model['a']:.3f} + {run_model['b']:.3f} × cadence;"
          f" ZUPT scale {scale:.3f}" + ("" if dist else " (defaults: no measured walks yet)"))
    extra.update(step_length=step_model, run_step_length=run_model, zupt_scale=scale,
                 trained_on=sorted({s.source for s in final_train}))
    export = {"layers": model.export(), **extra}

    # The board's numpy forward pass must match PyTorch.
    check = stepnet.StepNet(export)
    x = stepnet.net_inputs(DEMO["x"])
    with torch.no_grad():
        ref = model(torch.tensor(x[None]))[0].numpy()
    diff = np.abs(check.logits(x) - ref).max()
    print(f"  numpy vs PyTorch max difference: {diff:.2e}")
    assert diff < 1e-3

    MODEL_PATH.write_text(json.dumps(export, separators=(",", ":")))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Wrote {os.path.relpath(MODEL_PATH, ROOT)} ({n_params} weights, "
          f"{MODEL_PATH.stat().st_size // 1024} KB)")

    plot_loss(history, RESULTS_DIR / "loss.png")
    if trials:
        plot_example(net, demo, RESULTS_DIR / "example_trial.png", f"Trial: {demo.trial['kind']}")
    np.savez_compressed(RESULTS_DIR / "history.npz", demo_x=DEMO["x"],
                        demo_target=demo.target[demo_at:demo_at + 600], context=CONTEXT,
                        **{k: np.array(v) for k, v in history.items()})
    print(f"Plots and history in {os.path.relpath(RESULTS_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
