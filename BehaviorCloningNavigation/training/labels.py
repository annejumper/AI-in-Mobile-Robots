"""Loading recordings, and finding heel strikes to train the network on.

Heel strikes come from the classic shank-gyro rule: while the leg swings
forward, gz (rotation about the left-right axis) dips hard, to about -400 deg/s
walking. That's mid-swing. The heel strike is the first gz peak after the dip,
when the swing is stopped by the foot hitting the ground.

That rule is the "teacher". It's only trusted on a counted trial when
2 x strikes matches the step count you walked (see label_trial), and the
network is then trained to reproduce those labels from raw IMU data, so it
also learns what is *not* a step (biking, shuffling, standing).
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

SAMPLE_HZ = 100
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Swing dips smaller than this aren't a stride (deg/s). A range is tried when
# matching a hand count; this is the default for unchecked data.
SWING_DPS = 150
SWING_DPS_RANGE = np.arange(60, 401, 10)
MIN_STRIDE_S = 0.4
STRIKE_SEARCH_S = 0.4      # heel strike comes within this long after mid-swing


def lowpass(x, cutoff_hz, axis=0):
    b, a = butter(2, cutoff_hz / (SAMPLE_HZ / 2))
    return filtfilt(b, a, x, axis=axis)


IMU_COLUMNS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
# Recordings from before the wearer's name was saved were all Anne's.
LEGACY_PERSON = "Anne"


def load_session(path):
    """(labels, trial ids, people, (N, 6) samples) for one data/session_*.csv.

    Columns are found by name, since they were added over time: older
    recordings have no trial column (ids are all 0) and no person column
    (LEGACY_PERSON).
    """
    labels, trials, people, rows = [], [], [], []
    with open(path, newline="", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:                     # empty file: powered off right after start
            return (np.array([]), np.array([], dtype=int), np.array([]),
                    np.zeros((0, 6), dtype=np.float32))
        col = {name: i for i, name in enumerate(header)}
        imu = [col[c] for c in IMU_COLUMNS]
        for row in reader:
            if len(row) != len(header) or "\x00" in "".join(row):
                continue                       # torn line from a power cut
            try:
                values = [float(row[i]) for i in imu]
            except ValueError:
                continue
            labels.append(row[col["label"]])
            trials.append(int(row[col["trial"]]) if "trial" in col else 0)
            people.append(row[col["person"]] if "person" in col else LEGACY_PERSON)
            rows.append(values)
    return (np.array(labels), np.array(trials), np.array(people),
            np.array(rows, dtype=np.float32).reshape(-1, 6))


def load_trials():
    """Rows of data/trials.csv as dicts (empty list if there are none yet)."""
    path = DATA_DIR / "trials.csv"
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["trial"] = int(r["trial"])
        r["person"] = r.get("person") or LEGACY_PERSON
        r["counted_steps"] = int(r["counted_steps"]) if r["counted_steps"] else None
        r["distance_m"] = float(r["distance_m"]) if r["distance_m"] else None
    return [r for r in rows if r["counted_steps"]]


def heel_strikes(x, swing_dps=SWING_DPS):
    """Sample indices of the sensor foot's heel strikes in (N, 6) samples."""
    if len(x) < 20:
        return np.array([], dtype=int)
    gz = lowpass(x[:, 5], 6.0)
    swings, _ = find_peaks(-gz, height=swing_dps, prominence=swing_dps,
                           distance=int(MIN_STRIDE_S * SAMPLE_HZ))
    strikes = []
    for s in swings:
        end = min(len(gz) - 1, s + int(STRIKE_SEARCH_S * SAMPLE_HZ))
        peaks, _ = find_peaks(gz[s:end + 1])
        if len(peaks):
            strikes.append(s + peaks[0])
    return np.array(strikes, dtype=int)


def label_trial(x, counted_steps):
    """Heel strikes for a counted trial, or None if the rule can't match the count.

    Tries every swing threshold in SWING_DPS_RANGE and keeps those where
    2 x strikes is within 1 of the count (the sensor only sees one foot, so
    the count's parity can't be known). Of those, the middle of the longest
    run of agreeing thresholds is used: a count that only matches at one
    knife-edge threshold isn't trusted.
    """
    ok = [t for t in SWING_DPS_RANGE if abs(2 * len(heel_strikes(x, t)) - counted_steps) <= 1]
    if not ok:
        return None
    runs, run = [], [ok[0]]
    for t in ok[1:]:
        if t - run[-1] <= 10:
            run.append(t)
        else:
            runs.append(run)
            run = [t]
    runs.append(run)
    best = max(runs, key=len)
    if len(best) < 3:
        return None
    return heel_strikes(x, best[len(best) // 2])
