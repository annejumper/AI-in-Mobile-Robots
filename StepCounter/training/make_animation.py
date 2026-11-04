"""Render the "how it works" video: IMU -> neural net -> steps, and how PyTorch trains it.

    .venv/bin/python training/make_animation.py          # -> training/results/steps/how_it_works.mp4
    .venv/bin/python training/make_animation.py --preview # one PNG per scene, in results/steps/preview/

Every number and curve in it is real: the network's weights come from
step-counter/python/stepnet.json, and the training scenes replay
training/results/steps/history.npz, which train_steps.py saves (the net's
output on one walking clip, the loss, gradient sizes and first-layer
weights after every epoch). Run train_steps.py first.
"""

import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "step-counter" / "python"))
import stepnet  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "steps"
OUT = RESULTS / "how_it_works.mp4"
FPS = 24

BG = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
GOLD = "#eda100"
TINT = "#eef4fc"
CHANNEL_NAMES = ["accel X", "accel Y", "accel Z", "gyro X", "gyro Y", "gyro Z", "|accel|", "|gyro|"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13, "text.color": INK,
                     "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2,
                     "animation.ffmpeg_path": imageio_ffmpeg.get_ffmpeg_exe()})

H = np.load(RESULTS / "history.npz")
NET = stepnet.StepNet.load(ROOT / "step-counter" / "python" / "stepnet.json")
X = H["demo_x"]                                    # (600, 6) walking clip
INPUTS = stepnet.net_inputs(X)                     # (8, 600)
N = INPUTS.shape[1]
T = np.arange(N) / stepnet.SAMPLE_HZ
EPOCHS = H["epoch"]
N_WEIGHTS = sum(w.size + b.size for w, b, _ in NET.layers)


def activations():
    """Every layer's output for the demo clip, with the board's numpy maths."""
    out, h = [INPUTS], INPUTS
    for i, (w, b, d) in enumerate(NET.layers):
        k = w.shape[2]
        pad = d * (k - 1) // 2
        hp = np.pad(h, ((0, 0), (pad, pad)))
        taps = np.stack([hp[:, j * d: j * d + N] for j in range(k)], axis=1)
        h = np.einsum("oik,ikn->on", w, taps) + b[:, None]
        if i < len(NET.layers) - 1:
            h = np.maximum(h, 0)
        out.append(h)
    return out


ACTS = activations()
PROB = 1 / (1 + np.exp(-ACTS[-1][0]))
STRIKES = stepnet.find_strikes(PROB, NET.threshold)
TARGET = H["demo_target"]


# ---------- drawing helpers ----------

def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def seg(t, a, b):
    """0..1 progress of t through [a, b]."""
    return ease((t - a) / (b - a))


def canvas(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return ax


def title(ax, text, sub=None, alpha=1.0):
    ax.text(0.6, 8.35, text, fontsize=24, weight="bold", alpha=alpha, va="center")
    if sub:
        ax.text(0.6, 7.8, sub, fontsize=14, color=INK_2, alpha=alpha, va="center")


def caption(ax, text, alpha=1.0):
    ax.text(8, 0.45, text, fontsize=15, color=INK_2, ha="center", va="center", alpha=alpha)


def box(ax, x, y, w, h, text, color=BLUE, lit=0.0, fs=13, sub=None):
    face = (1 - lit) * np.array(matplotlib.colors.to_rgb(BG)) + lit * np.array(matplotlib.colors.to_rgb(TINT))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=face, ec=color, lw=1.6 + 2 * lit))
    ax.text(x + w / 2, y + h / 2 + (0.16 if sub else 0), text, ha="center", va="center", fontsize=fs, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center", fontsize=fs - 3, color=INK_2)


def arrow(ax, x0, y0, x1, y1, color=INK_2, lw=1.6, alpha=1.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, alpha=alpha))


def plot_ax(fig, rect):
    a = fig.add_axes(rect)
    a.set_facecolor(BG)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    a.grid(color=GRID, lw=0.8)
    a.set_axisbelow(True)
    return a


# ---------- scenes: each is (seconds, draw(fig, t)) ----------

def s_title(fig, t):
    ax = canvas(fig)
    a = seg(t, 0, 1)
    ax.text(8, 5.3, "How the step counter works", fontsize=38, weight="bold", ha="center", alpha=a)
    ax.text(8, 4.4, "An ankle IMU, a small neural net trained in PyTorch, and backpropagation",
            fontsize=17, color=INK_2, ha="center", alpha=seg(t, 0.5, 1.5))
    ax.text(8, 3.6, "Arduino UNO Q  ·  MPU-6050 (3-axis accelerometer + 3-axis gyroscope)  ·  100 samples/s",
            fontsize=14, color=MUTED, ha="center", alpha=seg(t, 1, 2))


def s_system(fig, t):
    ax = canvas(fig)
    title(ax, "The whole system", "Train once on the laptop, then the board counts by itself")
    ax.text(0.6, 6.9, "LAPTOP  (once)", fontsize=12, color=MUTED, weight="bold")
    ax.text(0.6, 4.2, "ANKLE + PHONE  (live, every 0.5 s)", fontsize=12, color=MUTED, weight="bold")
    top = [(0.6, "Recorded walks", "CSV, hand-counted"), (4.4, "PyTorch", "train the CNN"),
           (8.2, "stepnet.json", f"{N_WEIGHTS:,} weights")]
    bottom = [(0.6, "MPU-6050", "6 axes, 100 Hz"), (3.7, "STM32 sketch", "reads I²C"),
              (6.8, "Linux · numpy", "same net, forward only"), (9.9, "Strikes → steps", "+ distance"),
              (13.0, "Phone page", "live count")]
    for i, (x, name, sub) in enumerate(top):
        lit = seg(t, 1 + i * 0.8, 1.6 + i * 0.8) * (1 - seg(t, 4.5, 5))
        box(ax, x, 5.6, 3.2, 1.1, name, color=BLUE, lit=lit, sub=sub)
        if i:
            arrow(ax, x - 0.55, 6.15, x - 0.05, 6.15)
    for i, (x, name, sub) in enumerate(bottom):
        lit = seg(t, 5 + i * 0.8, 5.5 + i * 0.8)
        box(ax, x, 2.9, 2.7, 1.1, name, color=GREEN, lit=lit, sub=sub, fs=12)
        if i:
            arrow(ax, x - 0.38, 3.45, x - 0.05, 3.45)
    arrow(ax, 9.8, 5.55, 8.4, 4.05, color=BLUE, lw=2, alpha=seg(t, 3.5, 4.5))
    ax.text(9.55, 4.75, "weights copied to the board", fontsize=11, color=BLUE, alpha=seg(t, 3.5, 4.5))
    # A packet of samples travelling along the live chain
    u = ((t - 5) % 4) / 4 if t > 5 else 0
    if t > 5:
        xs = 0.6 + 1.35 + u * (13.0 + 1.35 - 0.6 - 1.35)
        ax.plot(xs, 3.45 - 0.75, "o", color=ORANGE, ms=10)
    caption(ax, ["The net is trained on the laptop, where PyTorch can compute gradients.",
                 "Only its weights go to the board, which runs the same maths with numpy."][int(t > 5)])


def s_inputs(fig, t):
    ax = canvas(fig)
    title(ax, "What the network sees", "8 channels: the 6 sensor axes plus the size of the acceleration and of the rotation")
    reveal = int(N * seg(t, 0.3, 5))
    for c in range(8):
        a = plot_ax(fig, [0.1, 0.70 - c * 0.075, 0.82, 0.065])
        a.plot(T[:reveal], INPUTS[c, :reveal], color=BLUE if c < 3 else ORANGE if c < 6 else GREEN, lw=1.4)
        a.set_xlim(0, T[-1])
        lim = max(0.3, np.abs(INPUTS[c]).max() * 1.1)
        a.set_ylim(-lim, lim)
        a.set_yticks([])
        a.set_xticks([] if c < 7 else range(7))
        a.grid(False)
        a.text(-0.02, 0.5, CHANNEL_NAMES[c], transform=a.transAxes, ha="right", va="center", fontsize=11)
        for s in STRIKES:
            if s < reveal:
                a.axvline(T[s], color=GRID, lw=1.2, zorder=0)
    caption(ax, "Each input is divided by a fixed scale (2 g, 300 °/s) so all channels are about −1…1. "
                "Grey lines: heel strikes.", alpha=seg(t, 4, 5))


def s_conv(fig, t):
    ax = canvas(fig)
    title(ax, "Layer 1: a convolution", "A 9-sample filter slides along time. At every position: multiply, add up, ReLU")
    w, b, d = NET.layers[0]
    f = int(np.argmax(np.abs(w).sum(axis=(1, 2))))         # the strongest filter
    c = int(np.argmax(np.abs(w[f]).sum(axis=1)))           # and the input it weighs most
    post = ACTS[1][f]
    raw = np.einsum("ik,ikn->n", w[f], np.stack(
        [np.pad(INPUTS, ((0, 0), (4, 4)))[:, j: j + N] for j in range(9)], axis=1)) + b[f]
    mid = STRIKES[len(STRIKES) // 2] if STRIKES else N // 2
    lo = int(np.clip(mid - 150, 5, N - 305))
    span = slice(lo, lo + 300)
    pos = int(span.start + (span.stop - span.start - 1) * seg(t, 1, 11))
    a1 = plot_ax(fig, [0.07, 0.46, 0.56, 0.27])
    a1.plot(T[span], INPUTS[c, span], color=ORANGE, lw=1.6)
    lim = np.abs(INPUTS[c, span]).max() * 1.15
    a1.add_patch(Rectangle((T[pos - 4], -lim), 9 / 100, 2 * lim, color=GOLD, alpha=0.4))
    a1.set_ylim(-lim, lim)
    a1.set_xlim(T[span.start], T[span.stop - 1])
    a1.set_ylabel(f"input: {CHANNEL_NAMES[c]}")
    a1.set_xticklabels([])
    a2 = plot_ax(fig, [0.07, 0.14, 0.56, 0.27])
    a2.plot(T[span.start:pos + 1], raw[span.start:pos + 1], color=MUTED, lw=1.2, ls="--", label="weighted sum")
    a2.plot(T[span.start:pos + 1], post[span.start:pos + 1], color=BLUE, lw=2.2, label="after ReLU")
    a2.plot(T[pos], post[pos], "o", color=BLUE, ms=8)
    a2.axhline(0, color=INK_2, lw=1)
    a2.set_xlim(T[span.start], T[span.stop - 1])
    lim = np.abs(raw[span]).max() * 1.1
    a2.set_ylim(-lim, lim)
    a2.set_ylabel(f"output: filter {f + 1}")
    a2.set_xlabel("time (s)")
    a2.legend(loc="upper left", frameon=False, fontsize=11)
    a3 = plot_ax(fig, [0.71, 0.46, 0.25, 0.25])
    a3.bar(range(9), w[f, c], color=GOLD)
    a3.axhline(0, color=INK_2, lw=1)
    a3.set_title(f"its 9 learned weights for {CHANNEL_NAMES[c]}", fontsize=12)
    a3.set_xticks(range(9))
    ax.text(11.3, 3.35, "h[t] = ReLU( b + Σ w[c,k] · x[c, t+k−4] )", fontsize=14)
    ax.text(11.3, 2.95, "sum over 8 channels c and 9 taps k", fontsize=11, color=INK_2)
    ax.text(11.3, 2.45, "72 weights + 1 bias per filter.\nLayer 1 has 16 filters,\n"
                        "so it outputs 16 new channels.", fontsize=12, color=INK_2, va="top",
            alpha=seg(t, 4, 5))
    caption(ax, "ReLU sets negative sums to 0, so a filter only 'fires' where its pattern appears.",
            alpha=seg(t, 7, 8))


def s_layers(fig, t):
    ax = canvas(fig)
    title(ax, "Four layers, each looking further apart",
          "Dilation 1, 2, 4, 8: same 9-tap filters, spread out. Each output sees 1.2 s of motion")
    names = ["input (8)", "layer 1 (16), dilation 1", "layer 2 (16), dilation 2",
             "layer 3 (16), dilation 4", "layer 4 (16), dilation 8", "P(heel strike)"]
    # How many samples of each row one output sample depends on: each layer
    # adds 8 × its dilation (input 121, layer 1 113, ... layer 4 and output 1)
    span = [121, 113, 97, 65, 1, 1]
    centre = STRIKES[2] if len(STRIKES) > 2 else N // 2
    show = seg(t, 8, 9)
    for i, act in enumerate(ACTS):
        a = fig.add_axes([0.2, 0.74 - i * 0.12, 0.74, 0.1])
        alpha = seg(t, i * 1.2, i * 1.2 + 1)
        if i == 5:                                 # show the output as a probability
            act, cmap, vmin, vmax = PROB[None], "Greens", 0, 1
        elif i == 0:
            vmax = np.percentile(np.abs(act), 99) + 1e-6
            cmap, vmin = "RdBu_r", -vmax
        else:
            cmap, vmin, vmax = "Blues", 0, np.percentile(act, 99) + 1e-6
        a.imshow(act, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                 extent=[0, T[-1], 0, act.shape[0]], interpolation="nearest")
        a.set_yticks([])
        a.set_xticks([] if i < 5 else range(7))
        a.text(-0.01, 0.5, names[i], transform=a.transAxes,
               ha="right", va="center", fontsize=11, alpha=max(alpha, 0.2))
        if show:
            a.add_patch(Rectangle((T[centre] - span[i] / 200, 0), max(span[i], 3) / 100, act.shape[0],
                                  fill=False, ec=GOLD, lw=2.5, alpha=show))
    caption(ax, "Gold: all the samples one output depends on (121 = 1.2 s)",
            alpha=seg(t, 8.5, 9.5))


def s_output(fig, t):
    ax = canvas(fig)
    title(ax, "From probabilities to steps",
          "Output = P(heel strike now) for every sample. Each bump above 0.5 is one strike")
    sweep = int(N * seg(t, 0.5, 8))
    a = plot_ax(fig, [0.07, 0.18, 0.62, 0.52])
    a.plot(T[:sweep], PROB[:sweep], color=GREEN, lw=2.2)
    a.axhline(NET.threshold, color=INK_2, ls="--", lw=1)
    a.text(T[-1], NET.threshold + 0.03, "threshold 0.5", ha="right", fontsize=11, color=INK_2)
    got = [s for s in STRIKES if s < sweep]
    a.plot(T[got], PROB[got], "o", color=INK, ms=9)
    a.set_xlim(0, T[-1])
    a.set_ylim(-0.03, 1.08)
    a.set_xlabel("time (s)")
    a.set_ylabel("P(heel strike)")
    ax.text(12.9, 5.4, f"{len(got)}", fontsize=70, weight="bold", ha="center", va="center")
    ax.text(12.9, 4.4, "heel strikes (ankle foot)", fontsize=13, ha="center", color=INK_2)
    ax.text(13.0, 3.2, f"= {2 * len(got)} steps", fontsize=28, weight="bold", ha="center", color=GREEN)
    ax.text(12.9, 2.4, "one strike per stride, and\nthe other foot steps in between", fontsize=12,
            ha="center", color=INK_2, va="top")
    caption(ax, "On the board this runs every 0.5 s on the last 4 s, keeping only samples with 0.6 s "
                "of data on both sides.", alpha=seg(t, 6, 7))


def s_loss(fig, t):
    ax = canvas(fig)
    title(ax, "Training: how wrong is it?", "The loss compares the output with the target bumps at each labeled strike")
    e = 0
    a = plot_ax(fig, [0.07, 0.33, 0.86, 0.4])
    p0 = H["demo_prob"][e]
    a.plot(T, TARGET, color=INK_2, lw=1.5, ls="--", label="target y (labeled strikes)")
    a.plot(T, p0, color=GREEN, lw=2, label="untrained net's output p")
    a.fill_between(T, TARGET, p0, color=ORANGE, alpha=0.25 * seg(t, 2, 3), label="error")
    a.set_xlim(0, T[-1])
    a.set_ylim(-0.03, 1.3)
    a.set_yticks([0, 0.5, 1])
    a.legend(loc="upper left", frameon=False, ncol=3, fontsize=11)
    a.set_xlabel("time (s)")
    ax.text(8, 1.65, "L  =  − mean over samples of   w · [ y log p + (1 − y) log(1 − p) ]",
            fontsize=17, ha="center", alpha=seg(t, 3, 4))
    ax.text(8, 1.05, f"binary cross-entropy, weight w = 1 + 9y so the rare strikes count.   "
                     f"Untrained loss: {H['train_loss'][0]:.3f}", fontsize=13, ha="center",
            color=INK_2, alpha=seg(t, 4, 5))


def s_backprop(fig, t):
    ax = canvas(fig)
    title(ax, "Backpropagation", "Which way should each of the 8,145 weights move to make the loss smaller?")
    names = ["input", "conv 1", "conv 2", "conv 3", "conv 4", "1×1 out", "loss"]
    xs = np.linspace(1.0, 13.6, len(names))
    y0 = 4.9
    grads = H["grad_norms"][1]
    for i, (x, name) in enumerate(zip(xs, names)):
        fwd = seg(t, i * 0.5, i * 0.5 + 0.4)
        back = seg(t, 4.5 + (6 - i) * 0.7, 4.5 + (6 - i) * 0.7 + 0.5) if i else 0
        color = ORANGE if back > 0 else BLUE
        box(ax, x, y0, 1.5, 1.0, name, color=color if i < 6 else GOLD, lit=max(fwd * (1 - seg(t, 3.6, 4)), back), fs=12)
        if i:
            arrow(ax, xs[i - 1] + 1.55, y0 + 0.65, x - 0.05, y0 + 0.65, color=BLUE, alpha=0.4 + 0.6 * fwd)
            arrow(ax, x - 0.05, y0 + 0.3, xs[i - 1] + 1.55, y0 + 0.3, color=ORANGE, alpha=back)
        if 1 <= i <= 5:
            g = grads[i - 1] / grads.max()
            ax.add_patch(Rectangle((x + 0.25, y0 - 0.25 - 1.6 * g), 1.0, 1.6 * g, color=ORANGE,
                                   alpha=0.85 * back))
            ax.text(x + 0.75, y0 - 2.1, f"{grads[i - 1]:.2f}", ha="center", fontsize=11,
                    color=INK_2, alpha=back)
    ax.text(0.6, 2.6, "gradient size\nper layer\n(first batch)", fontsize=11, color=ORANGE, alpha=seg(t, 6, 7))
    ax.text(xs[3], 6.75, "forward pass: input → prediction → loss", color=BLUE, fontsize=14,
            alpha=seg(t, 0, 0.5) * (1 - seg(t, 4, 4.5)), ha="center")
    ax.text(xs[3], 6.75, "backward pass: the chain rule, from the loss back to every weight", color=ORANGE,
            fontsize=14, alpha=seg(t, 4.5, 5), ha="center")
    ax.text(8, 1.75, "∂L/∂w₁ = ∂L/∂p · ∂p/∂h₄ · ∂h₄/∂h₃ · ∂h₃/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂w₁",
            fontsize=16, ha="center", alpha=seg(t, 8, 9))
    ax.text(8, 1.05, "In PyTorch:  loss.backward()  fills in every weight's gradient,  "
                     "then  optimizer.step()  moves each weight a little downhill (Adam).",
            fontsize=13, ha="center", color=INK_2, alpha=seg(t, 10, 11))
    ax.text(8, 0.45, "w  ←  w − η · (smoothed gradient)        η = learning rate = 0.003",
            fontsize=13, ha="center", color=INK_2, alpha=seg(t, 12, 13))


def s_learning(fig, t):
    ax = canvas(fig)
    u = seg(t, 0.5, 14)
    e = int(round(u * (len(EPOCHS) - 1)))
    batches = int(round(EPOCHS[e] * 60))
    title(ax, f"Learning: {batches:,} updates  (epoch {EPOCHS[e]:.2g} of {EPOCHS[-1]:.0f})",
          "Each update: 32 random 4 s clips → forward → loss → backward → step")
    a = plot_ax(fig, [0.06, 0.12, 0.52, 0.6])
    a.plot(T, TARGET, color=INK_2, lw=1.5, ls="--", label="target")
    a.plot(T, H["demo_prob"][e], color=GREEN, lw=2.2, label="net output")
    a.set_ylim(-0.03, 1.08)
    a.set_xlim(0, T[-1])
    a.set_xlabel("time (s)")
    a.set_title("output on a walking clip", fontsize=13, loc="left")
    a.legend(loc="upper right", frameon=False, fontsize=11)
    b = plot_ax(fig, [0.66, 0.47, 0.3, 0.25])
    b.plot(EPOCHS[:e + 1], H["train_loss"][:e + 1], color=BLUE, lw=2)
    b.set_xscale("symlog", linthresh=1)            # stretch out the first epoch
    b.set_xlim(0, EPOCHS[-1])
    b.set_xticks([0, 0.5, 1, 5, 10, 40], ["0", "½", "1", "5", "10", "40"])
    b.set_ylim(0, H["train_loss"].max() * 1.05)
    b.set_title("loss", fontsize=13, loc="left")
    b.set_xlabel("epoch")
    c = fig.add_axes([0.66, 0.1, 0.3, 0.26])
    w = H["conv1_w"][e]                                   # (16, 8, 9)
    vmax = np.abs(H["conv1_w"][-1]).max()
    c.imshow(w.reshape(16, -1), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    c.set_title("layer 1 weights (16 filters × 72)", fontsize=12, loc="left")
    c.set_xticks([])
    c.set_yticks([])


def s_deploy(fig, t):
    ax = canvas(fig)
    title(ax, "On the board", "Same net, same output, no PyTorch needed")
    lines = [
        (f"1.  Export: {N_WEIGHTS:,} weights  →  stepnet.json", BLUE),
        ("2.  Every 0.5 s the board runs the 5 convolutions on the last 4 s with numpy", GREEN),
        ("     (checked: matches PyTorch to within 0.0001)", MUTED),
        ("3.  Bumps above 0.5 → heel strikes → steps = 2 × strikes", GREEN),
        ("4.  Distance, per stride: integrate the IMU between two foot plants (ZUPT)", ORANGE),
        ("     and, as a check, steps × a step length that grows with cadence", MUTED),
        ("5.  The phone shows the count, the distance and the net's output live", BLUE),
    ]
    for i, (text, color) in enumerate(lines):
        ax.text(1.2, 6.6 - i * 0.78, text, fontsize=17, color=INK if color != MUTED else MUTED,
                alpha=seg(t, i * 0.9, i * 0.9 + 0.6))
        if color != MUTED:
            ax.plot(0.85, 6.68 - i * 0.78, "s", color=color, ms=10, alpha=seg(t, i * 0.9, i * 0.9 + 0.6))


def s_end(fig, t):
    ax = canvas(fig)
    a = seg(t, 0, 1)
    ax.text(8, 5.0, "IMU  →  convolutions  →  P(heel strike)  →  steps", fontsize=26, weight="bold",
            ha="center", alpha=a)
    ax.text(8, 4.1, "trained by:  forward pass  →  loss  →  backpropagation  →  weight update  (×2,400)",
            fontsize=17, ha="center", color=INK_2, alpha=seg(t, 0.8, 1.8))


SCENES = [(5, s_title), (11, s_system), (7, s_inputs), (13, s_conv), (11, s_layers), (9, s_output),
          (7, s_loss), (15, s_backprop), (16, s_learning), (8, s_deploy), (4, s_end)]


def main():
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    if "--preview" in sys.argv:
        out = RESULTS / "preview"
        out.mkdir(exist_ok=True)
        for i, (dur, scene) in enumerate(SCENES):
            fig.clf()
            scene(fig, dur * 0.95)
            fig.savefig(out / f"scene{i + 1:02d}.png", facecolor=BG)
        print(f"Wrote {len(SCENES)} preview PNGs to {out}")
        return
    total = sum(d for d, _ in SCENES)
    writer = FFMpegWriter(fps=FPS, bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(OUT), dpi=100):
        for i, (dur, scene) in enumerate(SCENES):
            for k in range(int(dur * FPS)):
                fig.clf()
                scene(fig, k / FPS)
                writer.grab_frame(facecolor=BG)
            print(f"  scene {i + 1}/{len(SCENES)} done")
    print(f"Wrote {OUT} ({total} s)")


if __name__ == "__main__":
    main()
