"""Train the navigator (behavior cloning) on your labeled walks and export it to the board.

    python3 training/train_policy.py              # walks marked "Train on this walk"
    python3 training/train_policy.py --all        # every labeled walk (for trying things out)
    python3 training/train_policy.py --deploy     # ...and put it on the board (USB)

Each labeled walk (walks/<id>/labels.csv, made by label_walks.py) becomes
one example per stride: the state after that stride (policy.features())
and what the demonstrator did next (policy.target()). The model is
k-nearest-neighbours over those examples (see campus-nav/python/policy.py).

Testing: with two or more walks, each walk is replayed through a model
trained on the *other* walks only (leave-one-walk-out), so the numbers say
how it would do on a walk it has never seen. Reported:
  - stride accuracy: how often it said the right thing after each stride;
  - events: for every turn, flight of stairs and the arrival, whether the
    model said it in time (on or before the stride it happened), late
    (up to 2 strides after), or missed it;
  - false alarms: strides where it said turn/stairs but you went straight
    and nothing was coming up within LOOKAHEAD strides.

Writes campus-nav/python/policy.json (board) and training/results/policy/report.txt.
"""

import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import walklib

sys.path.insert(0, str(walklib.APP_PY))
import policy  # noqa: E402

OUT = walklib.APP_PY / "policy.json"
RESULTS = Path(__file__).resolve().parent / "results" / "policy"
BOARD_APP = "/home/arduino/ArduinoApps/campus-nav"


def load_labeled(include_all):
    walks = []
    for wid in walklib.list_walks():
        d = walklib.WALKS_DIR / wid
        if not (d / "labels.csv").exists() or not (d / "labels.json").exists():
            continue
        info = json.loads((d / "labels.json").read_text())
        if not include_all and not info.get("use"):
            continue
        with open(d / "labels.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        strides = [{k: float(r[k]) for k in ("dist_m", "dz_m", "heading_deg", "dheading_deg")} for r in rows]
        walks.append({"id": wid, "route": info.get("route", ""), "kind": info.get("kind", ""),
                      "person": info.get("person", ""), "strides": strides,
                      "labels": [r["label"] for r in rows]})
    return walks


def examples(walk):
    """(X, y) for one walk: the state before the first stride and after each one."""
    xs, ys = [], []
    for t in range(-1, len(walk["strides"])):
        xs.append(policy.vector(policy.features(walk["strides"][: t + 1])))
        ys.append(policy.LABELS.index(policy.target(walk["labels"], t)))
    return xs, ys


def build(walks):
    xs, ys = [], []
    for w in walks:
        x, y = examples(w)
        xs += x
        ys += y
    return {"x": [[round(float(v), 4) for v in row] for row in xs], "y": ys, "k": policy.K,
            "labels": policy.LABELS, "features": policy.FEATURES, "scales": policy.SCALES,
            "lookahead": policy.LOOKAHEAD,
            "info": {"trained": f"{datetime.now():%Y-%m-%d %H:%M}", "walks": [w["id"] for w in walks],
                     "people": sorted({w["person"] for w in walks}), "examples": len(ys)}}


def replay(model, walk):
    """What the model says after each stride of a walk: list of predicted labels (index t+1)."""
    p = policy.Policy(model)
    return [p.predict(walk["strides"][: t + 1])[0] for t in range(-1, len(walk["strides"]))]


def score(walk, said):
    """Stride accuracy, event timing and false alarms for one replayed walk."""
    labels = walk["labels"]
    want = [policy.target(labels, t) for t in range(-1, len(labels))]
    acc = np.mean([a == b for a, b in zip(said, want)])
    events = []                     # (label, first stride, verdict)
    i = 0
    while i < len(labels):
        lab = labels[i]
        j = i
        while j + 1 < len(labels) and labels[j + 1] == lab:
            j += 1
        if lab != "straight":
            # said[t + 1] is what it said after stride t; stride i (0-based) is announced in time
            # if it was said after any of strides i-LOOKAHEAD .. i-1, i.e. said[i-LOOKAHEAD+1 .. i]
            early = said[max(0, i - policy.LOOKAHEAD + 1): i + 1]
            late = said[i + 1: i + 3]
            verdict = "in time" if lab in early else "late" if lab in late else "missed"
            events.append((lab, i + 1, verdict))
        i = j + 1
    false_alarms = sum(1 for s, w in zip(said, want) if s != "straight" and w == "straight")
    return acc, events, false_alarms


def main():
    include_all = "--all" in sys.argv
    walks = load_labeled(include_all)
    lines = []
    out = lambda s="": (print(s), lines.append(s))  # noqa: E731
    out(f"Labeled walks used: {len(walks)}" + ("  (--all: including ones not marked for training)" if include_all else ""))
    for w in walks:
        c = Counter(w["labels"])
        out(f"  {w['id']}  {w['kind']:8} {w['person']:8} {w['route'][:28]:28} {len(w['labels']):3d} strides  "
            + "  ".join(f"{k}={v}" for k, v in c.items() if k != "straight"))
    if not walks:
        out("Nothing to train on: label some walks with label_walks.py and tick 'Train on this walk'.")
        return

    if len(walks) >= 2:
        out("\nLeave-one-walk-out test (each walk replayed by a model that never saw it):")
        accs, verdicts, fas = [], Counter(), 0
        for i, w in enumerate(walks):
            model = build(walks[:i] + walks[i + 1:])
            said = replay(model, w)
            acc, events, fa = score(w, said)
            accs.append(acc)
            fas += fa
            verdicts.update(v for _, _, v in events)
            out(f"  {w['id']}: stride accuracy {acc:.0%}, false alarms {fa}")
            for lab, at, v in events:
                out(f"      {lab:11} at stride {at:3d}: {v}")
        n_ev = sum(verdicts.values())
        out(f"\n  Overall: stride accuracy {np.mean(accs):.0%}; events in time {verdicts['in time']}/{n_ev}, "
            f"late {verdicts['late']}, missed {verdicts['missed']}; false alarms {fas}")
    else:
        out("\nOnly one walk, so there's nothing to test on. (Replaying it through a model trained on "
            "itself only checks the plumbing.)")
        acc, events, fa = score(walks[0], replay(build(walks), walks[0]))
        out(f"  self-replay stride accuracy {acc:.0%}, events in time "
            f"{sum(v == 'in time' for *_, v in events)}/{len(events)}, false alarms {fa}")

    model = build(walks)
    OUT.write_text(json.dumps(model))
    out(f"\nWrote {OUT.relative_to(walklib.ROOT)} ({model['info']['examples']} examples from {len(walks)} walks)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.txt").write_text("\n".join(lines) + "\n")

    if "--deploy" in sys.argv:
        r = subprocess.run(["adb", "push", str(OUT), f"{BOARD_APP}/python/policy.json"],
                           capture_output=True, text=True)
        print("Pushed to the board" if r.returncode == 0 else f"Push failed: {r.stderr.strip()}")
        print("It's picked up at the next walk you start (no restart needed).")


if __name__ == "__main__":
    main()
