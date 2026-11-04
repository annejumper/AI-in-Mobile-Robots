# The step counter's neural net, run with numpy, plus how its output is turned
# into steps and distance. The same code runs on the board (main.py) and in
# training (training/train_steps.py), so the numbers in the writeup are the
# numbers the board would give.
#
# The net is trained in PyTorch on the laptop (training/train_steps.py) and
# saved to stepnet.json. Here it is just its weights: a stack of 1D
# convolutions over the 6 IMU channels that outputs, for every sample, the
# probability that the sensor foot's heel is striking the ground right then.
#
# A sample is [ax, ay, az, gx, gy, gz] in g and deg/s. Axes as mounted on the
# ankle: X up the leg, Y forward, Z to the left (gz = leg swinging).

import json
import math

import numpy as np

SAMPLE_HZ = 100
DT = 1.0 / SAMPLE_HZ
G = 9.81
ACC_SCALE_G = 2.0           # inputs are divided by these so they're roughly -1..1
GYR_SCALE_DPS = 300.0
N_INPUTS = 8                # 6 IMU axes + |accel| - 1 g + |rotation|

# Turning probabilities into strikes (see StrikeTracker)
STRIKE_THRESHOLD = 0.5
MIN_STRIKE_GAP = 35         # samples. One foot can't strike twice in 0.35 s
# A gap longer than this between strikes means you stopped: the stride isn't
# measured, and its length is guessed from the step-length model instead.
MAX_STRIDE_S = 2.5
# A stride whose acceleration peaks above this is running. In the recorded
# trials running strides peak at 7.6-15 g and walking/stairs at 2-7 g.
RUN_PEAK_G = 9.0


def net_inputs(x):
    """(N, 6) IMU samples -> (N_INPUTS, N) scaled network inputs."""
    x = np.asarray(x, dtype=np.float32)
    acc, gyr = x[:, :3], x[:, 3:6]
    return np.concatenate([
        (acc / ACC_SCALE_G).T,
        (gyr / GYR_SCALE_DPS).T,
        ((np.linalg.norm(acc, axis=1) - 1.0) / ACC_SCALE_G)[None],
        (np.linalg.norm(gyr, axis=1) / GYR_SCALE_DPS)[None],
    ]).astype(np.float32)


class StepNet:
    """Forward pass of the exported network (same maths as torch.nn.Conv1d)."""

    def __init__(self, model):
        self.layers = [(np.array(l["weight"], dtype=np.float32),
                        np.array(l["bias"], dtype=np.float32), l["dilation"])
                       for l in model["layers"]]
        self.threshold = model.get("threshold", STRIKE_THRESHOLD)
        self.step_model = model.get("step_length", DEFAULT_STEP_MODEL)
        self.run_model = model.get("run_step_length", self.step_model)
        self.zupt_scale = model.get("zupt_scale", 1.0)
        # Samples at each end of a window that don't see their full context
        self.context = sum(d * (w.shape[2] - 1) // 2 for w, _, d in self.layers)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def logits(self, inputs):
        """(N_INPUTS, N) -> (N,) heel-strike logit for every sample."""
        h = inputs
        for i, (w, b, d) in enumerate(self.layers):
            out_ch, in_ch, k = w.shape
            pad = d * (k - 1) // 2
            n = h.shape[1]
            hp = np.pad(h, ((0, 0), (pad, pad)))
            # Each output is a weighted sum of k inputs spaced d samples apart
            taps = np.stack([hp[:, j * d: j * d + n] for j in range(k)], axis=1)  # (in, k, n)
            h = np.einsum("oik,ikn->on", w, taps) + b[:, None]
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)          # ReLU
        return h[0]

    def probabilities(self, x):
        return 1.0 / (1.0 + np.exp(-self.logits(net_inputs(x))))


class StrikeTracker:
    """Turns a stream of per-sample probabilities into heel strikes.

    A strike is a stretch where the probability stays above the threshold;
    it's placed at the stretch's highest point. A second stretch starting
    within MIN_STRIKE_GAP of the last strike is ignored (still the same one).
    Feed samples in order with add(); it returns a strike's sample number
    once the stretch has ended, else None.
    """

    def __init__(self, threshold=STRIKE_THRESHOLD):
        self.threshold = threshold
        self.best = None            # (p, seq) of the stretch in progress
        self.last = None            # seq of the last strike

    def add(self, seq, p):
        if p >= self.threshold:
            if self.best is None:
                if self.last is not None and seq - self.last < MIN_STRIKE_GAP:
                    return None
                self.best = (p, seq)
            elif p > self.best[0]:
                self.best = (p, seq)
            return None
        if self.best is None:
            return None
        strike = self.best[1]
        self.best, self.last = None, strike
        return strike


def find_strikes(prob, threshold=STRIKE_THRESHOLD):
    """Strike sample indices in a whole recording (what the board does live)."""
    tracker = StrikeTracker(threshold)
    strikes = [s for s in (tracker.add(i, p) for i, p in enumerate(prob)) if s is not None]
    last = tracker.add(len(prob), 0.0)          # close a stretch still open at the end
    return strikes + ([last] if last is not None else [])


# ---------- distance ----------
#
# Two ways to measure each stride (heel strike to the next heel strike of the
# same foot = two steps):
#   step-length model: stride = 2 * (a + b * cadence), cadence in steps/s.
#       People take longer steps when they walk faster. a and b are fitted to
#       walks of known length (defaults are typical adult values).
#   ZUPT ("zero-velocity update"): the ankle is nearly still for a moment
#       while the foot is planted. Starting from that moment, integrate the
#       gyro to track which way the sensor is pointing, rotate the
#       acceleration into the room's frame, remove gravity and integrate
#       twice to get position. The velocity should be zero again at the next
#       foot plant; whatever isn't is drift, which is removed linearly. The
#       horizontal distance travelled is the stride length.
#
# Running strides get their own step-length model (running steps are longer
# at the same cadence), and ZUPT isn't used on them: the foot is only planted
# for ~0.2 s, so the ankle is never still, and 100 Hz catches only a few
# samples of each 10-15 g impact. Their ZUPT value is the running model's.

DEFAULT_STEP_MODEL = {"a": 0.30, "b": 0.22}


def stride_from_cadence(stride_s, step_model=DEFAULT_STEP_MODEL):
    cadence = 2.0 / stride_s
    return 2.0 * (step_model["a"] + step_model["b"] * cadence)


def stance_index(x, start, end):
    """Stillest sample in the first 60% of the stride after heel strike `start`."""
    lo = start + 5
    hi = max(lo + 1, start + int(0.6 * (end - start)))
    gyr = np.linalg.norm(np.asarray(x[lo:hi])[:, 3:6], axis=1)
    return lo + int(np.argmin(np.convolve(gyr, np.ones(5) / 5, mode="same")))


def _rotation_to_up(acc):
    """Rotation matrix taking the measured gravity direction to world +Z."""
    a = acc / np.linalg.norm(acc)
    up = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, up)
    s, c = np.linalg.norm(v), float(np.dot(a, up))
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def zupt_stride(x, s0, s1):
    """Horizontal distance (m) the ankle moved between two foot plants s0 < s1."""
    x = np.asarray(x, dtype=float)
    R = _rotation_to_up(x[max(0, s0 - 3): s0 + 4, :3].mean(axis=0))
    vel = np.zeros(3)
    vels = [vel]
    for k in range(s0, s1):
        w = np.radians(x[k, 3:6]) * DT                 # rotation this sample (rad)
        angle = np.linalg.norm(w)
        if angle > 1e-9:
            axis = w / angle
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = R @ (np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K)
        acc = (R @ x[k + 1, :3] - np.array([0.0, 0.0, 1.0])) * G
        vel = vel + acc * DT
        vels.append(vel)
    vels = np.array(vels)
    # The ankle is still at both plants, so any end velocity is drift.
    ramp = np.linspace(0.0, 1.0, len(vels))[:, None]
    vels = vels - ramp * vels[-1]
    pos = vels[1:].sum(axis=0) * DT
    return float(np.hypot(pos[0], pos[1]))


TYPICAL_STRIDE_S = 1.1


def stride_samples_needed(prev, strike):
    """How many samples after `strike` stride_length() reads."""
    return int(0.6 * (strike - prev)) + 3


def is_running(x, prev, strike):
    """True if the stride from strike `prev` to `strike` hit a running impact."""
    return float(np.abs(np.asarray(x[prev:strike])[:, :3]).max()) >= RUN_PEAK_G


def stride_length(x, prev, strike, step_model=DEFAULT_STEP_MODEL, zupt_scale=1.0,
                  run_model=None, zupt=True):
    """(cadence model m, ZUPT m) for the stride ending at heel strike `strike`.

    `prev` is the previous strike of the same foot, or None if there wasn't
    one within MAX_STRIDE_S (the first stride after standing). That first
    swing only carries the foot from beside the other one to a step ahead,
    so it counts as half a typical stride. Unless zupt is False (no next
    foot plant: the cadence model is used for both), x must hold
    stride_samples_needed() samples past `strike`.
    """
    if prev is None or (strike - prev) * DT > MAX_STRIDE_S:
        half = stride_from_cadence(TYPICAL_STRIDE_S, step_model) / 2
        return half, half
    stride_s = (strike - prev) * DT
    if run_model is not None and is_running(x, prev, strike):
        m = stride_from_cadence(stride_s, run_model)
        return m, m
    m = stride_from_cadence(stride_s, step_model)
    if not zupt:
        return m, m
    s0 = stance_index(x, prev, strike)
    s1 = stance_index(x, strike, strike + (strike - prev))
    return m, zupt_stride(x, s0, s1) * zupt_scale
