"""
A tiny k-nearest-neighbors classifier for stroke type (flat / topspin /
slice) from IMU swing features. Trained with train_strokes.py; used by the
game automatically if strokes_model.json exists.
"""

import json
from collections import Counter

import numpy as np


SPIN_FEATURES = 3   # pitch, yaw and other-axis gyro peaks; accel and duration mostly add noise


class StrokeKNN:
    def __init__(self, X, y, k=5):
        self.X = np.asarray(X, dtype=float)
        self.y = list(y)
        self.k = k
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-9   # standardize so no feature dominates
        self.Z = (self.X - self.mean) / self.std
        self._spin = {}

    def predict(self, features, exclude=None):
        z = (np.asarray(features, dtype=float) - self.mean) / self.std
        d = np.linalg.norm(self.Z - z, axis=1)
        if exclude is not None:
            d[exclude] = np.inf
        nearest = np.argsort(d)[:self.k]
        return Counter(self.y[i] for i in nearest).most_common(1)[0][0]

    def spin_model(self, stroke):
        """Flat / topspin / slice model built from only this stroke's swings.
        The camera already tells forehand from backhand (far more reliably than
        the IMU), and once the side is known the spin is much easier to tell apart."""
        if stroke not in self._spin:
            idx = [i for i, y in enumerate(self.y) if y.startswith(stroke + "_")]
            self._spin[stroke] = StrokeKNN(self.X[idx, :SPIN_FEATURES],
                                           [self.y[i].split("_")[1] for i in idx], self.k) if idx else None
        return self._spin[stroke]

    def predict_spin(self, features, stroke):
        m = self.spin_model(stroke)
        return m.predict(list(features)[:SPIN_FEATURES]) if m else None

    def leave_one_out_accuracy(self):
        hits = sum(self.predict(self.X[i], exclude=i) == self.y[i] for i in range(len(self.y)))
        return hits / len(self.y)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"X": self.X.tolist(), "y": self.y, "k": self.k}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["X"], d["y"], d["k"])
