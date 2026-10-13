"""Train your own stroke classifier from IMU swings: forehand and backhand,
each flat / topspin / slice (6 classes).

    .venv/bin/python train_strokes.py [--swings 10] [--swing-axis y] [--spin-axis z]
    .venv/bin/python train_strokes.py --labels backhand_slice     # redo / add one class

For each stroke type you do a set of swings with the hub. Each swing becomes
a feature vector (pitch rate, yaw twist, the third gyro axis, accel,
duration), and a k-nearest-neighbors model is saved to strokes_model.json.
tennis.py loads it automatically and uses it for the spin TYPE, with a
separate flat/topspin/slice model per stroke (the camera decides forehand vs
backhand, and the yaw twist still sets the spin AMOUNT).
"""

import argparse
import json
import math
import os
import time

import legoeducation as le  # noqa: F401  (the hub library under lelib)
from lelib import doubleMotor

import racket as rk
from strokes import StrokeKNN

DOUBLE_MOTOR_CARD_SERIAL = 6065
LABELS = [f"{stroke}_{spin}" for stroke in ("forehand", "backhand")
          for spin in ("flat", "topspin", "slice")]
DATA = "strokes_data.json"
MODEL = "strokes_model.json"


GO_WINDOW_S = 2.0   # after the high beep you have this long to swing
SETTLE_S = 0.5      # stop listening this long after your last swing ends
RESET_S = 1.5       # after the low beep: reset to ready position (motion ignored)


def listen(dm, det, seconds, on_swing=None, stop=None):
    """Feed the detector for up to `seconds`; call on_swing(ev) for each swing.
    stop() can end it early."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end and not (stop and stop()):
        imu = dm.imu_device
        if imu is not listen.last:
            listen.last = imu
            g = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
            if not any(math.isnan(v) for v in g):
                ev = det.feed(time.monotonic(), g,
                              (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ))
                if ev and on_swing:
                    on_swing(ev)
        time.sleep(0.002)


listen.last = None


def collect(dm, det, n):
    """Guided rhythm: HIGH beep = swing now, LOW beep = reset. Only the
    strongest motion after each high beep is kept, so windups and resets
    are never saved as strokes."""
    swings = []
    while len(swings) < n:
        listen(dm, det, 0.4)                     # let any leftover motion finish
        dm.beep(frequency=1046, blocking=False)  # GO
        best = []

        def keep(ev):
            if not best or ev.power > best[0].power:
                best[:] = [ev]

        def done():
            return best and not det.in_swing and time.monotonic() - best[0].t_end > SETTLE_S

        listen(dm, det, GO_WINDOW_S, keep, done)
        if not best:
            print("  no swing heard - again")
            continue
        ev = best[0]
        swings.append(ev.features())
        print(f"  swing {len(swings)}/{n}  power {ev.power * 100:.0f}%  spin {ev.spin:+.2f}")
        dm.beep(frequency=330, blocking=False)   # reset
        listen(dm, det, RESET_S)
    return swings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swings", type=int, default=10, help="swings per stroke type")
    ap.add_argument("--labels", default=",".join(LABELS),
                    help="comma-separated classes to record (default: all 6)")
    ap.add_argument("--swing-axis", choices=["x", "y", "z"], default=rk.SWING_AXIS)
    ap.add_argument("--spin-axis", choices=["x", "y", "z"], default=rk.SPIN_AXIS)
    args = ap.parse_args()

    dm = doubleMotor()
    print(f"Connecting to Double Motor, card {DOUBLE_MOTOR_CARD_SERIAL} ...")
    dm.connect(card_serial=DOUBLE_MOTOR_CARD_SERIAL)
    det = rk.SwingDetector(args.swing_axis, args.spin_axis)

    data = {"X": [], "y": []}
    if os.path.exists(DATA) and input(f"Add to existing {DATA}? [Y/n] ").strip().lower() != "n":
        with open(DATA) as f:
            data = json.load(f)

    labels = [lb.strip() for lb in args.labels.split(",") if lb.strip()]
    for label in labels:
        if label not in LABELS:
            raise SystemExit(f"unknown label {label!r}; choose from {', '.join(LABELS)}")
    for label in labels:
        stroke, spin = label.split("_")
        input(f"\nReady to record {args.swings} {stroke.upper()} {spin.upper()} swings"
              "\n  Press Enter, then: HIGH beep = swing, LOW beep = walk back to ready.")
        X = collect(dm, det, args.swings)
        data["X"] += X
        data["y"] += [label] * len(X)

    with open(DATA, "w") as f:
        json.dump(data, f)
    knn = StrokeKNN(data["X"], data["y"], k=5)
    knn.save(MODEL)
    print(f"\nSaved {MODEL} from {len(data['y'])} swings. "
          f"Leave-one-out accuracy: {knn.leave_one_out_accuracy() * 100:.0f}%\n")
    print_confusion(knn)
    for stroke in ("forehand", "backhand"):
        m = knn.spin_model(stroke)
        if m:
            print(f"{stroke} spin, with the side from the camera: {m.leave_one_out_accuracy() * 100:.0f}%")
    print("\ntennis.py will use it automatically. Delete the file to go back to twist-only spin.")


def print_confusion(knn):
    """Leave-one-out confusion table: rows = what you did, columns = what the model guessed."""
    classes = [c for c in LABELS if c in knn.y]
    short = {c: c.replace("forehand_", "FH ").replace("backhand_", "BH ") for c in classes}
    print("did \\ guessed " + "".join(f"{short[c]:>12s}" for c in classes) + "   accuracy")
    for actual in classes:
        idx = [i for i, y in enumerate(knn.y) if y == actual]
        guesses = [knn.predict(knn.X[i], exclude=i) for i in idx]
        row = "".join(f"{guesses.count(c):12d}" for c in classes)
        print(f"{short[actual]:>14s}{row}   {guesses.count(actual) / len(idx) * 100:5.0f}%")


if __name__ == "__main__":
    main()
