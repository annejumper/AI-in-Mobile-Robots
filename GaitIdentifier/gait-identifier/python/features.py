# Features computed from one window of IMU samples, plus a tiny decision-tree
# evaluator. Plain Python (no numpy) so the exact same code runs on the board
# (main.py) and in training (training/train.py).
#
# A window is a list of [ax, ay, az, gx, gy, gz] rows in g and deg/s, sampled
# at SAMPLE_HZ. WINDOW_S is long enough to hold at least two strides even when
# walking slowly.

import math

SAMPLE_HZ = 100
WINDOW_S = 2.0
WINDOW_N = int(SAMPLE_HZ * WINDOW_S)
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]

# Model 1 uses only this feature: the peak acceleration up the leg (X), which
# is where heel strikes show up.
PEAK_FEATURE = "ax_max"


def _mean_std(v):
    m = sum(v) / len(v)
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _cadence_hz(v):
    """Strongest repeat rate of v between 0.5 and 4 Hz, by autocorrelation.

    That range covers slow walking up to sprinting and pedaling. Returns 0 if
    there's no clear rhythm (e.g. standing still).
    """
    m = sum(v) / len(v)
    d = [x - m for x in v]
    energy = sum(x * x for x in d)
    if energy < 1e-6:
        return 0.0
    best_lag, best_r = 0, 0.3          # need at least 0.3 correlation to count
    for lag in range(SAMPLE_HZ // 4, SAMPLE_HZ * 2 + 1):
        r = sum(d[i] * d[i + lag] for i in range(len(d) - lag)) / energy
        if r > best_r:
            best_lag, best_r = lag, r
    return SAMPLE_HZ / best_lag if best_lag else 0.0


def window_features(window):
    """Feature dict for one window (see AXES for the row layout)."""
    f = {}
    cols = list(zip(*window))
    for name, col in zip(AXES, cols):
        mean, std = _mean_std(col)
        f[f"{name}_mean"] = mean
        f[f"{name}_std"] = std
        f[f"{name}_max"] = max(col)
        f[f"{name}_min"] = min(col)

    acc = [math.sqrt(ax * ax + ay * ay + az * az) for ax, ay, az, *_ in window]
    gyr = [math.sqrt(gx * gx + gy * gy + gz * gz) for *_, gx, gy, gz in window]
    f["acc_mag_mean"], f["acc_mag_std"] = _mean_std(acc)
    f["acc_mag_max"] = max(acc)
    f["gyr_mag_mean"], f["gyr_mag_std"] = _mean_std(gyr)
    f["gyr_mag_max"] = max(gyr)
    # Rotation about Z (left-right axis) is the leg swinging forward and back,
    # which repeats once per stride or pedal stroke.
    f["cadence_hz"] = _cadence_hz(cols[5])
    return f


FEATURE_NAMES = list(window_features([[0.0] * 6] * WINDOW_N).keys())


def tree_predict(tree, features):
    """Walk a tree exported by training/train.py and return its class name."""
    node = 0
    while tree["left"][node] != -1:
        name = tree["features"][node]
        node = tree["left"][node] if features[name] <= tree["thresholds"][node] else tree["right"][node]
    return tree["classes"][tree["leaf_class"][node]]
