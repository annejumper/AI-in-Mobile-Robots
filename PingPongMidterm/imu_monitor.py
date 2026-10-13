"""Watch the Double Motor's IMU live and tune the swing thresholds.

    .venv/bin/python imu_monitor.py [--swing-axis y] [--spin-axis z]

Hold the hub the way you will in the game. Each swing prints its power
(pitch rate) and spin (yaw twist: + topspin, - slice, 0 flat).
  * Swing with NO twist: busiest axis should be the swing axis, spin ~0.
  * Twist only: busiest axis should be the spin axis.
  * If your topspin twist prints as SLICE, set SPIN_SIGN = -1 in racket.py.
  * Shafts: the status line shows both motor shaft angles. Turn one by hand;
    a full turn prints "SHAFT TURN -> to the net!" (SHAFT_TURN_DEG in racket.py).
    If the angles stay at "--", the hub isn't reporting motor positions.
"""

import argparse
import math
import time

import legoeducation as le  # noqa: F401  (the hub library under lelib)
from lelib import doubleMotor

import racket as rk

DOUBLE_MOTOR_CARD_SERIAL = 6065


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swing-axis", choices=["x", "y", "z"], default=rk.SWING_AXIS)
    ap.add_argument("--spin-axis", choices=["x", "y", "z"], default=rk.SPIN_AXIS)
    args = ap.parse_args()

    dm = doubleMotor()
    print(f"Connecting to Double Motor, card {DOUBLE_MOTOR_CARD_SERIAL} ...")
    dm.connect(card_serial=DOUBLE_MOTOR_CARD_SERIAL)
    dm.reset_heading()
    det = rk.SwingDetector(args.swing_axis, args.spin_axis)
    shafts = rk.ShaftTurnDetector()
    print(f"Swing thresholds: start {rk.SWING_START}, full {rk.SWING_FULL}, "
          f"spin dead zone {rk.SPIN_DEAD}, spin full {rk.SPIN_FULL}.  Ctrl-C to stop.\n")

    last, peaks, shown = None, [0.0, 0.0, 0.0], 0.0
    try:
        while True:
            imu = dm.imu_device
            if imu is not last:
                last = imu
                g = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
                a = (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
                if any(math.isnan(v) for v in g):
                    continue
                peaks = [max(p * 0.995, abs(v)) for p, v in zip(peaks, g)]
                now = time.monotonic()
                pos = rk.shaft_positions(dm)
                if shafts.feed(now, pos):
                    print(f"\nSHAFT TURN -> to the net!  (a {rk.SHAFT_TURN_DEG} degree turn)")
                ev = det.feed(now, g, a)
                if ev:
                    kind = "TOPSPIN" if ev.spin > 0.15 else "SLICE" if ev.spin < -0.15 else "FLAT"
                    print(f"\nSWING  power {ev.power * 100:4.0f}%  (peak pitch rate {ev.peak_rate:+6.0f})   "
                          f"{kind:7s} spin {ev.spin:+.2f}  (peak yaw twist {ev.peak_twist:+6.0f})   "
                          f"{(ev.t_end - ev.t_start) * 1000:.0f} ms")
                if time.monotonic() - shown > 0.1:
                    shown = time.monotonic()
                    busiest = "xyz"[peaks.index(max(peaks))]
                    sh = "  ".join("--" if p is None else f"{p:+6.0f}" for p in pos) or "--"
                    print(f"\rgyro x{g[0]:+6.0f} y{g[1]:+6.0f} z{g[2]:+6.0f}  "
                          f"yaw {imu.yaw:+5.0f} pitch {imu.pitch:+5.0f} roll {imu.roll:+5.0f}  "
                          f"busiest axis: {busiest}   shafts {sh}   ", end="", flush=True)
            time.sleep(0.002)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
