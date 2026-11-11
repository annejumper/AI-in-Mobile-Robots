"""
Fit the "V" data from Anna's Code with gradient descent, written out by hand (numpy only).
Notation follows the hand-written notes:

        x ──[ m  b ]── z ──[ ReLU ]── a

        z = m x + b
        a = ReLU(z) = 0 if z < 0,  z if z > 0
        L = (1/N) Σ (y - a)^2

        dL/dm = dL/da · da/dz · dz/dm
        dL/db = dL/da · da/dz · dz/db

        m_new = m_old - α dL/dm
        b_new = b_old - α dL/db

Part 1  One neuron with a ReLU (the notes exactly). Prints the first 5 iterations of m and b
        so you can check them against the Excel sheet. One ramp can't make a V.

Part 2  Two of those neurons side by side, added together by an output neuron:
            a1 = ReLU(m1 x + b1),  a2 = ReLU(m2 x + b2),  ŷ = w1 a1 + w2 a2 + c
        One ramp goes up, the other goes down: together they make the V.

Usage:
    python fit_v.py                               # uses annacode-v.tsv next to this file
    python fit_v.py my_data.tsv                   # a TSV saved from Anna's Code
    python fit_v.py --m0 1 --b0 0 --alpha 0.0001  # other starting values (defaults match the Excel sheet)
    python fit_v.py --no-relu                     # Part 1 without the ReLU (plain line z = mx + b)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def load_tsv(path):
    """Anna's Code TSV: header, then Time, x, y columns (the Time column is optional)."""
    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    return data[:, -2], data[:, -1]


def relu(z):
    """a = 0 if z < 0,  z if z > 0"""
    return np.maximum(0, z)


def relu_slope(z):
    """da/dz = 0 if z < 0,  1 if z > 0   (worked out separately for every point)"""
    return (z > 0).astype(float)


# ── Part 1: one neuron, x → [m b] → z → ReLU → a ────────────────────────────

def fit_one_neuron(x, y, m, b, alpha, epochs, use_relu=True, show=5):
    N = len(x)
    losses = []
    act = "ReLU" if use_relu else "no activation"
    print(f"\nPart 1: one neuron, {act}   (α = {alpha}, start m = {m}, b = {b})")
    print(f"{'iter':>4} {'m_initial':>10} {'b_initial':>10} {'Loss':>10} {'dL/dm':>10} {'dL/db':>10} "
          f"{'m_new':>10} {'b_new':>10}")

    for it in range(epochs):
        # forward
        z = m * x + b
        a = relu(z) if use_relu else z
        L = (1 / N) * np.sum((y - a) ** 2)

        # chain rule, one piece at a time (each is an array: one value per point)
        dL_da = -(2 / N) * (y - a)
        da_dz = relu_slope(z) if use_relu else np.ones_like(z)
        dz_dm = x
        dz_db = 1

        # a point with z < 0 has da/dz = 0, so it adds nothing to the sums below
        dL_dm = np.sum(dL_da * da_dz * dz_dm)
        dL_db = np.sum(dL_da * da_dz * dz_db)

        losses.append(L)

        # gradient descent update
        m_new = m - alpha * dL_dm
        b_new = b - alpha * dL_db

        if it < show:   # same columns as the spreadsheet's iteration blocks
            print(f"{it + 1:>4} {m:>10.4f} {b:>10.4f} {L:>10.4f} {dL_dm:>10.4f} {dL_db:>10.4f} "
                  f"{m_new:>10.4f} {b_new:>10.4f}")
        m, b = m_new, b_new

    print(f"final m = {m:.4f}, b = {b:.4f}, L = {losses[-1]:.2f}")
    predict = (lambda xq: relu(m * xq + b)) if use_relu else (lambda xq: m * xq + b)
    return predict, losses, m, b


# ── Part 2: two ReLU neurons + an output neuron ─────────────────────────────

def fit_two_neurons(x, y, alpha, epochs, seed):
    """
    Same chain rule, one more link because a1 and a2 now feed ŷ:

        dL/dm1 = dL/dŷ · dŷ/da1 · da1/dz1 · dz1/dm1 = Σ [-(2/N)(y - ŷ)] · w1 · ReLU'(z1) · x
        dL/db1 = dL/dŷ · dŷ/da1 · da1/dz1 · dz1/db1 = Σ [-(2/N)(y - ŷ)] · w1 · ReLU'(z1) · 1
        dL/dw1 = dL/dŷ · dŷ/dw1                    = Σ [-(2/N)(y - ŷ)] · a1
        dL/dc  = dL/dŷ · dŷ/dc                     = Σ [-(2/N)(y - ŷ)] · 1
        (and the same for neuron 2)

    x and y are rescaled to mean 0, std 1 first so that one α works for every weight;
    the weights are converted back to the original units at the end.
    """
    rng = np.random.default_rng(seed)
    x_mean, x_std, y_mean, y_std = x.mean(), x.std(), y.mean(), y.std()
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    N = len(xs)

    # The random starting weights matter: with some seeds both ReLUs start ramping the same way
    # (or switch off for every point), gradient descent gets stuck, and the fit stays a bent line.
    m1, m2 = rng.normal(0, 1, 2)
    b1, b2 = rng.normal(0, 0.5, 2)
    w1, w2 = rng.normal(0, 1, 2)
    c = 0.0

    losses = []
    for _ in range(epochs):
        # forward
        z1, z2 = m1 * xs + b1, m2 * xs + b2
        a1, a2 = relu(z1), relu(z2)
        y_hat = w1 * a1 + w2 * a2 + c
        losses.append((1 / N) * np.sum((ys - y_hat) ** 2) * y_std ** 2)   # L in original units

        # backward
        dL_dyhat = -(2 / N) * (ys - y_hat)

        dL_dw1 = np.sum(dL_dyhat * a1)
        dL_dw2 = np.sum(dL_dyhat * a2)
        dL_dc = np.sum(dL_dyhat)

        dL_dm1 = np.sum(dL_dyhat * w1 * relu_slope(z1) * xs)
        dL_db1 = np.sum(dL_dyhat * w1 * relu_slope(z1) * 1)
        dL_dm2 = np.sum(dL_dyhat * w2 * relu_slope(z2) * xs)
        dL_db2 = np.sum(dL_dyhat * w2 * relu_slope(z2) * 1)

        # gradient descent update
        m1 -= alpha * dL_dm1; b1 -= alpha * dL_db1
        m2 -= alpha * dL_dm2; b2 -= alpha * dL_db2
        w1 -= alpha * dL_dw1; w2 -= alpha * dL_dw2
        c -= alpha * dL_dc

    # Convert back to original units (so they can be typed into Anna's Code)
    M1, B1 = m1 / x_std, b1 - m1 * x_mean / x_std
    M2, B2 = m2 / x_std, b2 - m2 * x_mean / x_std
    W1, W2, C = w1 * y_std, w2 * y_std, c * y_std + y_mean

    print(f"\nPart 2: two ReLU neurons + output neuron   (α = {alpha}, {epochs} epochs, seed {seed})")
    print(f"  neuron 1: a1 = ReLU({M1:+.3f} x {B1:+.3f})")
    print(f"  neuron 2: a2 = ReLU({M2:+.3f} x {B2:+.3f})")
    print(f"  output:   ŷ = {W1:.3f}·a1 {W2:+.3f}·a2 {C:+.3f}")
    print(f"  final L = {losses[-1]:.2f}")

    def predict(xq):
        return W1 * relu(M1 * xq + B1) + W2 * relu(M2 * xq + B2) + C

    return predict, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="?", default=HERE / "annacode-v.tsv")
    # Defaults match the Excel sheet: m_initial = 1.3, b_initial = -60, α = 0.0001
    ap.add_argument("--m0", type=float, default=1.3, help="starting m")
    ap.add_argument("--b0", type=float, default=-60.0, help="starting b")
    ap.add_argument("--alpha", type=float, default=1e-4, help="learning rate α for Part 1")
    ap.add_argument("--epochs", type=int, default=2000, help="epochs for Part 1")
    ap.add_argument("--no-relu", action="store_true", help="Part 1 as a plain line, no ReLU")
    ap.add_argument("--alpha2", type=float, default=0.05, help="learning rate α for Part 2")
    ap.add_argument("--epochs2", type=int, default=2000, help="epochs for Part 2")
    # Seed 3 gives one rising and one falling ramp, which is the V. Try --seed 1 to see it get stuck.
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    x, y = load_tsv(args.tsv)
    print(f"Loaded {len(x)} points from {args.tsv}")

    predict1, loss1, m, b = fit_one_neuron(x, y, args.m0, args.b0, args.alpha, args.epochs,
                                           use_relu=not args.no_relu)
    predict2, loss2 = fit_two_neurons(x, y, args.alpha2, args.epochs2, args.seed)

    label1 = f"one neuron: a = {'ReLU(' if not args.no_relu else ''}{m:.2f}x {b:+.2f}{')' if not args.no_relu else ''}"
    xq = np.linspace(x.min(), x.max(), 300)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=14, color="gray", alpha=0.7, label="V data")
    ax.plot(xq, predict1(xq), color="tab:orange", lw=2, label=label1)
    ax.plot(xq, predict2(xq), color="tab:blue", lw=2, label="two ReLU neurons")
    ax.set(xlabel="x", ylabel="y", title="V data and fits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "v_fit.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(loss1, color="tab:orange", label="one neuron")
    ax.plot(loss2, color="tab:blue", label="two ReLU neurons")
    ax.set(xlabel="epoch", ylabel="L (mean squared error)", yscale="log", title="Loss vs epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "v_loss.png", dpi=150)

    print(f"\nSaved {HERE / 'v_fit.png'} and {HERE / 'v_loss.png'}")
    plt.show()


if __name__ == "__main__":
    main()
