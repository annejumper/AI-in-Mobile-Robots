# The navigator: a behavior-cloned policy that says what to do next.
#
# It works on strides, not raw IMU samples: after every stride of the
# sensor foot it looks at the walk so far (from the step counter, the
# heading tracker and ZUPT) and predicts what the demonstrator did on the
# next stride or two: straight, left, right, uturn_left, uturn_right,
# stairs_up, stairs_down or arrive. Looking ahead means the instruction shows up before the turn.
#
# State (features()), all computed from the strides so far:
#   dist_total      metres walked since Start
#   n_left, n_right turns made so far (a run of strides that together turn
#                   45 degrees or more one way counts as one turn)
#   n_stairs        flights of stairs so far
#   since_event     metres since the last turn or flight of stairs
#   heading         degrees turned since Start (+ = left), /90
#   dh1, dh2        turn on the last stride and the last two strides
#   dz1             height change on the last stride
# The counts are what tell it where it is on the route ("I've done two
# rights and a left, and walked 6 m since"), which is why it doesn't need
# GPS. Turns are counted from the heading, so a turn you make is counted
# whether or not the navigator told you to make it.
#
# Model: k nearest neighbours. Every stride of every training walk is a
# stored example (state -> what the demonstrator did next). To decide, it
# finds the K stored states closest to now and takes a distance-weighted
# vote of what was done next. With fewer than 10 walks that's about as much
# model as the data can support, and it runs in numpy on the board.
#
# The same code runs on the board (main.py) and in training
# (training/train_policy.py).

import json
import math

import numpy as np

LABELS = ["straight", "left", "right", "stairs_up", "stairs_down", "arrive", "uturn_left", "uturn_right"]
TURNS = ("left", "right", "uturn_left", "uturn_right")
LOOKAHEAD = 2               # strides ahead a turn or stairs is announced
TURN_STRIDE_DEG = 20        # a stride turning this much may be part of a turn
TURN_TOTAL_DEG = 45         # ...a run of them adding up to this is a turn
STAIR_DZ_M = 0.20           # a stride rising or dropping this much is on stairs
FEATURES = ["dist_total", "n_left", "n_right", "n_stairs", "since_event", "heading", "dh1", "dh2", "dz1"]
# How much a difference of 1 in each feature counts, after dividing by these.
# A whole extra turn (1) matters as much as 4 m more walked, or a 90 degree
# heading difference.
SCALES = {"dist_total": 4.0, "n_left": 1.0, "n_right": 1.0, "n_stairs": 1.0, "since_event": 2.0,
          "heading": 1.0, "dh1": 45.0, "dh2": 60.0, "dz1": 0.25}
K = 5


def features(strides):
    """State after the given strides (dicts with dist_m, dz_m, heading_deg, dheading_deg)."""
    dist = since = 0.0
    n_left = n_right = n_stairs = 0
    run_sum, run_counted, on_stairs = 0.0, False, False
    for s in strides:
        dh, dz = float(s["dheading_deg"]), float(s["dz_m"])
        dist += float(s["dist_m"])
        since += float(s["dist_m"])
        # turns: count a run of same-direction turning strides once, as soon as it reaches 45 deg
        if abs(dh) >= TURN_STRIDE_DEG and (run_sum == 0 or math.copysign(1, dh) == math.copysign(1, run_sum)):
            run_sum += dh
        else:
            run_sum, run_counted = (dh if abs(dh) >= TURN_STRIDE_DEG else 0.0), False
        if not run_counted and abs(run_sum) >= TURN_TOTAL_DEG:
            run_counted = True
            since = 0.0
            if run_sum > 0:
                n_left += 1
            else:
                n_right += 1
        # stairs: count a flight on its first stride
        stairs = abs(dz) >= STAIR_DZ_M
        if stairs and not on_stairs:
            n_stairs += 1
            since = 0.0
        on_stairs = stairs
    last = strides[-1] if strides else None
    dh1 = float(last["dheading_deg"]) if last else 0.0
    dh2 = dh1 + (float(strides[-2]["dheading_deg"]) if len(strides) >= 2 else 0.0)
    return {"dist_total": dist, "n_left": n_left, "n_right": n_right, "n_stairs": n_stairs,
            "since_event": since, "heading": (float(last["heading_deg"]) if last else 0.0) / 90.0,
            "dh1": dh1, "dh2": dh2, "dz1": float(last["dz_m"]) if last else 0.0}


def vector(f):
    return np.array([f[k] / SCALES[k] for k in FEATURES])


def instructions(labels):
    """Per stride, what a guide would say for it: a turn only on the stride it starts.

    A turn that takes two strides is labeled left/right on both, but once
    you've started turning the instruction should go back to straight.
    (Trained on the raw labels, the first navigator kept saying "right"
    after a right turn.) Stairs stay "stairs" for the whole flight.
    """
    return ["straight" if lab in TURNS and i and labels[i - 1] == lab else lab
            for i, lab in enumerate(labels)]


def target(labels, t):
    """What to say after stride t (0-based; -1 = before the first stride), from a labeled walk.

    The next stride's instruction, unless that's straight and a turn/stairs/
    arrive comes within LOOKAHEAD strides: then that.
    """
    said = instructions(labels)
    n = len(said)
    nxt = said[t + 1] if t + 1 < n else "arrive"
    if nxt != "straight":
        return nxt
    for k in range(t + 2, min(n, t + 1 + LOOKAHEAD)):
        if said[k] != "straight":
            return said[k]
    return "straight"


class Policy:
    def __init__(self, model):
        self.x = np.array(model["x"], dtype=float)
        self.y = np.array(model["y"], dtype=int)
        self.k = model.get("k", K)
        self.labels = model.get("labels", LABELS)
        self.info = model.get("info", {})

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def predict(self, strides):
        """(label, confidence 0-1, votes {label: share}) for the state after these strides."""
        v = vector(features(strides))
        d = np.sqrt(((self.x - v) ** 2).sum(axis=1))
        idx = np.argsort(d)[: self.k]
        w = 1.0 / (d[idx] + 0.1)
        votes = np.zeros(len(self.labels))
        np.add.at(votes, self.y[idx], w)
        votes /= votes.sum()
        best = int(np.argmax(votes))
        return self.labels[best], float(votes[best]), \
            {self.labels[i]: round(float(votes[i]), 2) for i in np.nonzero(votes)[0]}
