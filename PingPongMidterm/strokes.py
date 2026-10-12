"""
A tiny k-nearest-neighbors classifier for stroke type (flat / topspin /
slice) from IMU swing features. Trained with train_strokes.py; used by the
game automatically if strokes_model.json exists.
"""

import json
from collections import Counter

import numpy as np


class StrokeKNN:
    def __init__(self, X, y, k=5):
        self.X = np.asarray(X, dtype=float)
        self.y = list(y)
        self.k = k
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-9   # standardize so no feature dominates
        self.Z = (self.X - self.mean) / self.std

    def predict(self, features, exclude=None):
        z = (np.asarray(features, dtype=float) - self.mean) / self.std
        d = np.linalg.norm(self.Z - z, axis=1)
        if exclude is not None:
            d[exclude] = np.inf
        nearest = np.argsort(d)[:self.k]
        return Counter(self.y[i] for i in nearest).most_common(1)[0][0]

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
