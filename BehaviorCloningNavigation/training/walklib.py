"""Loading recorded walks, recomputing their strides, and first-guess labels.

A walk is a folder in walks/ copied off the board (data/walks/ there):
    meta.json     summary written by the board
    imu.csv       every raw sample (t_s, accel in g, gyro in deg/s, heading)
    strides.csv   one row per stride as the board saw it live
    marks.csv     buttons tapped during the walk (optional)
and, once you've reviewed it in label_walks.py:
    labels.json   your corrected segments + overrides of route/kind/use
    labels.csv    one row per stride: the features and the label to train on

Strides are recomputed from imu.csv with the code in campus-nav/python (the
same code the board runs), so an improvement there (like the heading fix of
2026-11-23) applies to old walks too. If imu.csv ends early (the board lost
power before it was written out), the board's own stride rows fill in the
rest.

Labels say what you did on each stride:
    straight, left, right, uturn_left, uturn_right, stairs_up, stairs_down
and the last stride of a walk is "arrive". Turns are found from the heading
(a run of strides that together turn 45 degrees or more; 135 or more is a
U-turn), stairs from the
height change of each stride (ZUPT).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APP_PY = ROOT / "campus-nav" / "python"
sys.path.insert(0, str(APP_PY))
import stepnet  # noqa: E402
from heading import HeadingTracker  # noqa: E402

WALKS_DIR = ROOT / "walks"
SAMPLE_HZ = 100
LABELS = ["straight", "left", "right", "uturn_left", "uturn_right", "stairs_up", "stairs_down", "arrive"]
KINDS = ["demo", "dagger", "test", "practice"]

STAIR_DZ_M = 0.20           # same as the board
TURN_STRIDE_DEG = 20        # a stride turning at least this much may be part of a turn
TURN_TOTAL_DEG = 45         # ...and the run of them must add up to at least this
UTURN_DEG = 135             # a turn this big is turning around (uturn_left/right)


def _read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", errors="ignore") as fh:
        return list(csv.DictReader(line.replace("\0", "") for line in fh))


def _float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def list_walks():
    return sorted(p.name for p in WALKS_DIR.iterdir() if p.is_dir() and (p / "imu.csv").exists()) \
        if WALKS_DIR.exists() else []


def load_imu(d):
    rows = []
    for r in _read_csv(d / "imu.csv"):
        try:
            rows.append([float(r[k]) for k in ("t_s", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")])
        except (KeyError, TypeError, ValueError):
            pass                            # a line cut short by a power cut
    return np.array(rows) if rows else np.zeros((0, 7))


def recompute_strides(d):
    """One dict per stride, recomputed from the raw samples (see module docstring)."""
    net = stepnet.StepNet.load(APP_PY / "stepnet.json")
    imu = load_imu(d)
    t, x = imu[:, 0], imu[:, 1:7]
    out = []
    if len(x) > 2 * net.context:
        ht = HeadingTracker()
        heading = np.array([ht.add(s) for s in x])
        heading -= heading[0]
        strikes = stepnet.find_strikes(net.probabilities(x), net.threshold)
        prev = None
        for s in strikes:
            pv = prev if prev is not None and (s - prev) / SAMPLE_HZ <= stepnet.MAX_STRIDE_S else None
            need = stepnet.FIRST_NEED if pv is None else stepnet.stride_samples_needed(pv, s)
            m = stepnet.stride_motion(x, pv, s, net.step_model, net.zupt_scale, net.run_model,
                                      zupt=s + need <= len(x))
            out.append({"t_s": round(float(t[s]), 2),
                        "stride_s": round((s - pv) / SAMPLE_HZ, 2) if pv is not None else None,
                        "dist_m": round(m["dist_m"], 3), "dz_m": round(m["dz_m"], 3),
                        "heading_deg": round(float(heading[s]), 1), "source": "imu"})
            prev = s
    # The board's rows for whatever imu.csv lost at the end
    end_t = float(t[-1]) if len(t) else -1.0
    h = out[-1]["heading_deg"] if out else 0.0
    for r in _read_csv(d / "strides.csv"):
        ts = _float(r.get("t_s"), -1)
        if ts <= end_t + 0.3:
            continue
        h += _float(r.get("dheading_deg"))
        out.append({"t_s": ts, "stride_s": _float(r.get("stride_s"), None),
                    "dist_m": _float(r.get("dist_m")), "dz_m": _float(r.get("dz_m")),
                    "heading_deg": round(h, 1), "source": "board"})
    for i, r in enumerate(out):
        r["stride"] = i + 1
        r["dheading_deg"] = round(r["heading_deg"] - out[i - 1]["heading_deg"], 1) if i else 0.0
    return out


def auto_labels(strides):
    """First guess at a label per stride (see module docstring)."""
    n = len(strides)
    labels = ["straight"] * n
    for i, s in enumerate(strides):
        if s["dz_m"] >= STAIR_DZ_M:
            labels[i] = "stairs_up"
        elif s["dz_m"] <= -STAIR_DZ_M:
            labels[i] = "stairs_down"
    # Turns: runs of strides that each turn at least TURN_STRIDE_DEG the same way
    i = 0
    while i < n:
        dh = strides[i]["dheading_deg"]
        if abs(dh) < TURN_STRIDE_DEG:
            i += 1
            continue
        j = i
        while j + 1 < n and abs(strides[j + 1]["dheading_deg"]) >= TURN_STRIDE_DEG \
                and np.sign(strides[j + 1]["dheading_deg"]) == np.sign(dh):
            j += 1
        total = sum(strides[k]["dheading_deg"] for k in range(i, j + 1))
        if abs(total) >= TURN_TOTAL_DEG:
            for k in range(i, j + 1):
                side = "left" if total > 0 else "right"
                labels[k] = f"uturn_{side}" if abs(total) >= UTURN_DEG else side
        i = j + 1
    if n:
        labels[-1] = "arrive"
    return labels


def segments_from_labels(labels):
    """[{start, end, label}] runs of equal labels (stride numbers, 1-based, inclusive)."""
    segs = []
    for i, lab in enumerate(labels):
        if segs and segs[-1]["label"] == lab:
            segs[-1]["end"] = i + 1
        else:
            segs.append({"start": i + 1, "end": i + 1, "label": lab})
    return segs


def labels_from_segments(segs, n):
    labels = ["straight"] * n
    for s in segs:
        for k in range(max(1, s["start"]), min(n, s["end"]) + 1):
            labels[k - 1] = s["label"]
    return labels


def load_walk(walk_id):
    """Everything the labeling page needs about one walk."""
    d = WALKS_DIR / walk_id
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {"id": walk_id}
    strides = recompute_strides(d)
    saved = None
    if (d / "labels.json").exists():
        try:
            saved = json.loads((d / "labels.json").read_text())
        except ValueError:
            saved = None
    auto = segments_from_labels(auto_labels(strides))
    marks = [{"t_s": _float(m.get("t_s")), "kind": m.get("kind", "")} for m in _read_csv(d / "marks.csv")]
    return {
        "id": walk_id, "meta": meta, "strides": strides, "marks": marks,
        "auto_segments": auto,
        "segments": saved["segments"] if saved and saved.get("n_strides") == len(strides) else auto,
        "reviewed": bool(saved and saved.get("n_strides") == len(strides)),
        "stale": bool(saved and saved.get("n_strides") != len(strides)),
        "route": (saved or {}).get("route", meta.get("route", "")),
        "kind": (saved or {}).get("kind", meta.get("kind", "demo")),
        "person": (saved or {}).get("person", meta.get("person", "")),
        "use": (saved or {}).get("use", meta.get("kind", "demo") in ("demo", "dagger")),
        "labels": LABELS, "kinds": KINDS,
    }


def save_labels(walk_id, body):
    """Write labels.json (what you decided) and labels.csv (per stride, for training)."""
    d = WALKS_DIR / walk_id
    strides = recompute_strides(d)
    n = len(strides)
    segs = [{"start": int(s["start"]), "end": int(s["end"]), "label": s["label"]}
            for s in body["segments"] if s["label"] in LABELS]
    labels = labels_from_segments(segs, n)
    keep = {k: body.get(k) for k in ("route", "kind", "person", "use")}
    (d / "labels.json").write_text(json.dumps(dict(keep, n_strides=n, segments=segs), indent=1))
    with open(d / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stride", "t_s", "stride_s", "dist_m", "dz_m", "heading_deg", "dheading_deg", "label"])
        for s, lab in zip(strides, labels):
            w.writerow([s["stride"], s["t_s"], "" if s["stride_s"] is None else s["stride_s"],
                        s["dist_m"], s["dz_m"], s["heading_deg"], s["dheading_deg"], lab])
    return {"ok": True, "n_strides": n}
