# Linux side of the GaitIdentifier wearable (step 2: labeled recording).
#
# Pulls batches of raw MPU-6050 samples from the sketch over the Bridge,
# converts them to g and deg/s, and saves every sample with its activity label
# (picked with the button on D2) to a CSV in data/, one file per power-on.
# These files are the training data. It also prints the latest reading a few
# times a second.
#
# Axes as mounted on the ankle: X up the leg, Y forward, Z to the left.
# Standing still, X should read about +1 g (or -1 g if the board is upside down).

import csv
import time
from datetime import datetime
from pathlib import Path

from arduino.app_utils import App, Bridge

SAMPLE_HZ = 100
ACCEL_LSB_PER_G = 2048.0    # +-16 g range
GYRO_LSB_PER_DPS = 16.4     # +-2000 deg/s range
# Subtracted from each accel reading. From the axis check on 2026-10-26:
# Z read +1.18 g pointing up and -0.85 g pointing down, so it sits 0.17 g high.
# X and Y read within 0.03 g of 1 g, so they're left alone.
ACCEL_OFFSET_G = (0.0, 0.0, 0.17)
LABELS = ["pause", "stand", "walk", "run", "bike"]
POLL_PERIOD_S = 0.05        # ~5 samples per batch
PRINT_PERIOD_S = 0.25
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATA_DIR.mkdir(exist_ok=True)
csv_path = DATA_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["t_s", "label", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"])
print(f"Saving samples to {csv_path}")

sensor_ok = None
next_seq = None
dropped = 0
latest = None
label = None
_last_print = 0.0


def parse(batch):
    """'<ok>|<seq>|lbl,ax,ay,az,gx,gy,gz;...' -> (ok, seq, rows of [label, g..., deg/s...])."""
    ok, seq, body = batch.split("|", 2)
    rows = []
    for item in body.split(";"):
        if item:
            v = [int(x) for x in item.split(",")]
            rows.append([LABELS[v[0]]]
                        + [c / ACCEL_LSB_PER_G - off for c, off in zip(v[1:4], ACCEL_OFFSET_G)]
                        + [c / GYRO_LSB_PER_DPS for c in v[4:]])
    return ok == "1", int(seq), rows


def loop():
    global sensor_ok, next_seq, dropped, latest, label, _last_print
    try:
        ok, seq, rows = parse(str(Bridge.call("get_samples")))
    except Exception as exc:
        print(f"Bridge call get_samples failed: {exc}")
        time.sleep(0.5)
        return

    if ok != sensor_ok:
        sensor_ok = ok
        if ok:
            whoami = int(Bridge.call("get_whoami"))
            print(f"MPU-6050 connected (WHO_AM_I = 0x{whoami & 0xFF:02X})")
        else:
            print("MPU-6050 not answering: check wiring (VCC 5V, GND, SDA D20, SCL D21)")

    if rows:
        if next_seq is not None and seq > next_seq:
            dropped += seq - next_seq
        for i, r in enumerate(rows):
            writer.writerow([f"{(seq + i) / SAMPLE_HZ:.2f}", r[0]] + [f"{x:.3f}" for x in r[1:]])
        # Flush every batch so a battery that runs flat loses at most ~50 ms.
        csv_file.flush()
        next_seq = seq + len(rows)
        latest = rows[-1]
        if latest[0] != label:
            label = latest[0]
            print(f"Activity: {label}")

    now = time.monotonic()
    if ok and latest and now - _last_print >= PRINT_PERIOD_S:
        _last_print = now
        _, ax, ay, az, gx, gy, gz = latest
        print(f"{label:5}  accel g  X{ax:+6.2f} Y{ay:+6.2f} Z{az:+6.2f}  |  "
              f"gyro deg/s  X{gx:+7.1f} Y{gy:+7.1f} Z{gz:+7.1f}  |  dropped {dropped}")
    time.sleep(POLL_PERIOD_S)


# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs).
try:
    App.run(user_loop=loop)
finally:
    csv_file.close()
