# Linux side of the GaitIdentifier wearable.
#
# Pulls batches of raw MPU-6050 samples from the sketch over the Bridge,
# converts them to g and deg/s, and:
#   - classifies the last 2 s every 0.5 s with both models in model.json
#     (made by training/train.py): "peak" uses only the peak X acceleration,
#     "imu" uses every IMU feature;
#   - saves every sample with its activity label to a CSV in data/, one file
#     per power-on (the training data).
#
# The web page in assets/ (http://<board IP>:7000, open it on a phone on the
# same network) shows the live answer, picks which model drives the LED
# matrix, and switches to Record mode to label new training data.
#
# Axes as mounted on the ankle: X up the leg, Y forward, Z to the left.
# Standing still, X should read about +1 g (or -1 g if the board is upside down).

import csv
import json
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

from features import WINDOW_N, tree_predict, window_features

SAMPLE_HZ = 100
ACCEL_LSB_PER_G = 2048.0    # +-16 g range
GYRO_LSB_PER_DPS = 16.4     # +-2000 deg/s range
# Subtracted from each accel reading. From the axis check on 2026-10-26:
# Z read +1.18 g pointing up and -0.85 g pointing down, so it sits 0.17 g high.
# X and Y read within 0.03 g of 1 g, so they're left alone.
ACCEL_OFFSET_G = (0.0, 0.0, 0.17)
LABELS = ["pause", "stand", "walk", "run", "bike"]
MODELS = {"peak": "Peak acceleration", "imu": "Full IMU"}
CLASSIFY_EVERY = 50         # samples between predictions (0.5 s)
PLOT_S = 10                 # history kept for the page's live plots
PLOT_EVERY = 2              # send every 2nd sample (50 Hz) to the page
SMOOTH_N = 3                # majority vote over the last 3 predictions
POLL_PERIOD_S = 0.05        # ~5 samples per batch
PRINT_PERIOD_S = 1.0
APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
MODEL_PATH = Path(__file__).resolve().parent / "model.json"

models = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else {}
print(f"Models loaded: {', '.join(models) or 'none (run training/train.py)'}")

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
seconds_recorded = {name: 0.0 for name in LABELS}
window = deque(maxlen=WINDOW_N)     # last 2 s of [ax..gz] for the models
since_classify = 0
recent = {name: deque(maxlen=SMOOTH_N) for name in models}
prediction = {name: None for name in models}
# For the live plots: [seq, ax..gz] and [seq, {model: prediction}]
plot_samples = deque(maxlen=PLOT_S * SAMPLE_HZ // PLOT_EVERY)
plot_predictions = deque(maxlen=PLOT_S * SAMPLE_HZ // CLASSIFY_EVERY + 1)
mode = "live"                       # "live": matrix shows the prediction
active_model = "imu" if "imu" in models else next(iter(models), None)
shown_matrix = None                 # last value sent with set_matrix
_last_print = 0.0

# The web page's requests run on the web server's thread, so they only leave
# requests here. loop() owns the Bridge and passes them on.
requested_label = None


def api_set_label(n: int):
    global requested_label
    if not 0 <= n < len(LABELS):
        return {"error": f"label must be 0-{len(LABELS) - 1}"}
    requested_label = n
    return {"requested": LABELS[n]}


def api_set_mode(name: str):
    global mode
    if name not in ("live", "record"):
        return {"error": "mode must be live or record"}
    mode = name
    return {"mode": mode}


def api_set_model(name: str):
    global active_model
    if name not in models:
        return {"error": f"model must be one of {', '.join(models)}"}
    active_model = name
    return {"model": active_model}


def api_status():
    return {
        "labels": LABELS,
        "label": label,
        "sensor_ok": bool(sensor_ok),
        "seconds": {k: round(v) for k, v in seconds_recorded.items()},
        "file": csv_path.name,
        "dropped": dropped,
        "mode": mode,
        "models": {k: MODELS.get(k, k) for k in models},
        "model": active_model,
        "prediction": prediction,
        "warming_up": len(window) < WINDOW_N,
        "peak_cutoffs": sorted({t for f, t in zip(models["peak"]["features"],
                                                  models["peak"]["thresholds"]) if f})
                        if "peak" in models else [],
    }


def api_stream(since: int = -1):
    """Plot data newer than sample number `since` (everything kept if -1)."""
    # list() copies a deque in one step, so the loop thread can't change it mid-copy.
    samples = [s for s in list(plot_samples) if s[0] > since]
    preds = [p for p in list(plot_predictions) if p[0] > since]
    return {"hz": SAMPLE_HZ, "samples": samples, "predictions": preds}


ui = WebUI()
ui.expose_api("POST", "/api/label/{n}", api_set_label)
ui.expose_api("POST", "/api/mode/{name}", api_set_mode)
ui.expose_api("POST", "/api/model/{name}", api_set_model)
ui.expose_api("GET", "/api/status", api_status)
ui.expose_api("GET", "/api/stream", api_stream)


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


def classify(seq):
    """Run every model on the window ending at sample `seq` and update the smoothed answers."""
    f = window_features(list(window))
    for name, tree in models.items():
        recent[name].append(tree_predict(tree, f))
        votes = Counter(recent[name]).most_common()
        # Ties go to the newest prediction so a real change shows up quickly.
        top = [p for p, n in votes if n == votes[0][1]]
        prediction[name] = recent[name][-1] if recent[name][-1] in top else top[0]
    plot_predictions.append([seq, dict(prediction)])


def update_matrix():
    """Live mode: matrix shows the active model's answer. Record mode: the label."""
    global shown_matrix
    want = -1
    if mode == "live" and active_model and prediction.get(active_model):
        want = LABELS.index(prediction[active_model])
    if want != shown_matrix:
        Bridge.call("set_matrix", want)
        shown_matrix = want


def loop():
    global sensor_ok, next_seq, dropped, latest, label, requested_label
    global since_classify, _last_print
    if requested_label is not None:
        n, requested_label = requested_label, None
        try:
            Bridge.call("set_label", n)
        except Exception as exc:
            print(f"Bridge call set_label failed: {exc}")

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
            window.clear()
            for name in models:
                recent[name].clear()
                prediction[name] = None

    if rows:
        if next_seq is not None and seq > next_seq:
            dropped += seq - next_seq
        for i, r in enumerate(rows):
            writer.writerow([f"{(seq + i) / SAMPLE_HZ:.2f}", r[0]] + [f"{x:.3f}" for x in r[1:]])
            seconds_recorded[r[0]] += 1 / SAMPLE_HZ
            window.append(r[1:])
            if (seq + i) % PLOT_EVERY == 0:
                plot_samples.append([seq + i] + [round(x, 2) for x in r[1:]])
        # Flush every batch so a battery that runs flat loses at most ~50 ms.
        csv_file.flush()
        next_seq = seq + len(rows)
        latest = rows[-1]
        if latest[0] != label:
            label = latest[0]
            print(f"Recording label: {label}")

        since_classify += len(rows)
        if models and len(window) == WINDOW_N and since_classify >= CLASSIFY_EVERY:
            since_classify = 0
            classify(next_seq - 1)

    try:
        update_matrix()
    except Exception as exc:
        print(f"Bridge call set_matrix failed: {exc}")

    now = time.monotonic()
    if ok and latest and now - _last_print >= PRINT_PERIOD_S:
        _last_print = now
        ax, ay, az = latest[1:4]
        guesses = "  ".join(f"{k}={v}" for k, v in prediction.items())
        print(f"{mode:6} {guesses}  |  rec {label:5}  |  "
              f"accel g X{ax:+5.2f} Y{ay:+5.2f} Z{az:+5.2f}  |  dropped {dropped}")
    time.sleep(POLL_PERIOD_S)


# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs).
try:
    App.run(user_loop=loop)
finally:
    csv_file.close()
