# Linux side of the StepCounter wearable (builds on the GaitIdentifier).
#
# Pulls batches of raw MPU-6050 samples from the sketch over the Bridge,
# converts them to g and deg/s, and:
#   - counts steps with the neural net in stepnet.json (made by
#     training/train_steps.py): every 0.5 s it runs on the last 4 s and turns
#     its heel-strike probabilities into strikes. One strike per stride of the
#     sensor foot, so steps = 2 x strikes;
#   - measures every stride two ways (steps x cadence-based step length, and
#     ZUPT integration of the IMU) and adds it to the distance walked;
#   - classifies the activity every 0.5 s with the two decision trees in
#     model.json (made by training/train_activity.py);
#   - runs counted trials: you press Start on the phone, walk while counting
#     your steps, press Stop and type in your count (and the distance, if
#     you walked a measured one). Each trial goes in data/trials.csv next to
#     what the net counted: the test data for the uncertainty analysis;
#   - saves every sample with its activity label, trial number and who is
#     wearing it to a CSV in data/, one file per power-on (the training data).
#     Several people can record: the wearer's name is picked on the phone
#     and remembered in data/people.json across power cycles.
#
# The web page in assets/ (http://<board IP>:7000, open it on a phone on the
# same network) shows all of it.
#
# Axes as mounted on the ankle: X up the leg, Y forward, Z to the left.
# Standing still, X should read about +1 g (or -1 g if the board is upside down).

import csv
import json
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import stepnet
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
TRIAL_KINDS = ["slow", "normal", "fast", "stairs", "run"]
CLASSIFY_EVERY = 50         # samples between predictions (0.5 s)
NET_WINDOW = 400            # samples the step net sees each time (4 s)
HISTORY_N = 800             # raw samples kept for measuring strides (8 s)
STOP_SETTLE = 150           # samples after Stop before a trial's count is final
PLOT_S = 10                 # history kept for the page's live plots
PLOT_EVERY = 2              # send every 2nd sample (50 Hz) to the page
SMOOTH_N = 3                # majority vote over the last 3 predictions
POLL_PERIOD_S = 0.05        # ~5 samples per batch
PRINT_PERIOD_S = 1.0
APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
TRIALS_PATH = DATA_DIR / "trials.csv"
PEOPLE_PATH = DATA_DIR / "people.json"
DEFAULT_PERSON = "Anne"     # the wearer until someone else is picked
MAX_NAME = 30
TRIAL_FIELDS = ["session", "person", "trial", "kind", "start_t", "end_t", "counted_steps", "distance_m",
                "net_steps", "cadence_m", "zupt_m", "saved_at"]
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.json"
STEPNET_PATH = HERE / "stepnet.json"

models = json.loads(MODEL_PATH.read_text()) if MODEL_PATH.exists() else {}
print(f"Activity models loaded: {', '.join(models) or 'none (run training/train_activity.py)'}")
net = stepnet.StepNet.load(STEPNET_PATH) if STEPNET_PATH.exists() else None
print("Step net loaded" if net else "No step net (run training/train_steps.py)")

DATA_DIR.mkdir(exist_ok=True)
csv_path = DATA_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["t_s", "label", "trial", "person", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"])
print(f"Saving samples to {csv_path}")

saved_trials = []
if TRIALS_PATH.exists():
    with open(TRIALS_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        saved_trials = list(reader)
    if reader.fieldnames != TRIAL_FIELDS:
        # Written before a column was added: rewrite it with the new columns
        with open(TRIALS_PATH, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(saved_trials)

people = {"current": DEFAULT_PERSON, "known": [DEFAULT_PERSON]}
if PEOPLE_PATH.exists():
    try:
        people.update(json.loads(PEOPLE_PATH.read_text()))
    except ValueError:
        print(f"Couldn't read {PEOPLE_PATH}, starting with {DEFAULT_PERSON}")
print(f"Wearer: {people['current']}")

sensor_ok = None
next_seq = None
dropped = 0
latest = None
label = None
seconds_recorded = {name: 0.0 for name in LABELS}
window = deque(maxlen=WINDOW_N)     # last 2 s of [ax..gz] for the activity models
since_classify = 0
recent = {name: deque(maxlen=SMOOTH_N) for name in models}
prediction = {name: None for name in models}
# For the live plots: [seq, ax..gz], [seq, {model: prediction}], [seq, P(strike)]
plot_samples = deque(maxlen=PLOT_S * SAMPLE_HZ // PLOT_EVERY)
plot_predictions = deque(maxlen=PLOT_S * SAMPLE_HZ // CLASSIFY_EVERY + 1)
plot_prob = deque(maxlen=PLOT_S * SAMPLE_HZ // PLOT_EVERY)
plot_strikes = deque(maxlen=64)
mode = "live"                       # "live": matrix shows the prediction
active_model = "imu" if "imu" in models else next(iter(models), None)
shown_matrix = None                 # last value sent with set_matrix
_last_print = 0.0

# Step counting state
history = deque(maxlen=HISTORY_N)   # [ax..gz] of the newest samples, for the net and ZUPT
history_end = None                  # seq of the sample after the newest in history
tracker = stepnet.StrikeTracker(net.threshold if net else stepnet.STRIKE_THRESHOLD)
fed_until = None                    # net output is final up to (not incl.) this seq
last_strike = None
pending_strides = deque()           # (prev strike, strike) waiting for enough samples
count = {"strikes": 0, "cadence_m": 0.0, "zupt_m": 0.0}
recent_strikes = deque(maxlen=5)
# Each strike is two steps, but the page shows them one at a time: the sensor
# foot's step when the strike is found, the other foot's half a stride later.
other_foot_due = None               # seq at which the latest strike's second step shows

# Trial state: idle -> walking (Start) -> settling (Stop) -> done (count final) -> idle (Save)
trial = {"state": "idle", "id": 0, "kind": None, "start_seq": None, "stop_seq": None, "result": None}
trial_number = 0

# The web page's requests run on the web server's thread, so they only leave
# requests here. loop() owns the Bridge and the counters and acts on them.
requested_label = None
requests = deque()


def api_set_label(n: int):
    global requested_label
    if not 0 <= n < len(LABELS):
        return {"error": f"label must be 0-{len(LABELS) - 1}"}
    requested_label = n
    return {"requested": LABELS[n]}


def api_set_mode(name: str):
    global mode
    if name not in ("live", "trial", "record"):
        return {"error": "mode must be live, trial or record"}
    mode = name
    return {"mode": mode}


def api_set_model(name: str):
    global active_model
    if name not in models:
        return {"error": f"model must be one of {', '.join(models)}"}
    active_model = name
    return {"model": active_model}


def api_reset():
    requests.append(("reset",))
    return {"ok": True}


def api_trial_start(kind: str):
    if kind not in TRIAL_KINDS:
        return {"error": f"kind must be one of {', '.join(TRIAL_KINDS)}"}
    if trial["state"] != "idle":
        return {"error": "a trial is already running"}
    requests.append(("start", kind))
    return {"ok": True}


def api_trial_stop():
    requests.append(("stop",))
    return {"ok": True}


def api_trial_save(steps: int, distance: float = 0.0):
    if trial["state"] != "done":
        return {"error": "no finished trial to save"}
    if steps <= 0:
        return {"error": "steps must be positive"}
    requests.append(("save", steps, distance if distance > 0 else None))
    return {"ok": True}


def api_set_person(name: str):
    name = " ".join(name.replace(",", " ").split())[:MAX_NAME]
    if not name:
        return {"error": "type a name"}
    if trial["state"] != "idle":
        return {"error": "finish the trial first"}
    requests.append(("person", name))
    return {"ok": True, "person": name}


def api_trial_discard():
    requests.append(("discard",))
    return {"ok": True}


def counter_view():
    cadence = 0.0
    if len(recent_strikes) >= 2 and next_seq is not None and next_seq - recent_strikes[-1] < 3 * SAMPLE_HZ:
        stride = (recent_strikes[-1] - recent_strikes[0]) / (len(recent_strikes) - 1) / SAMPLE_HZ
        cadence = 120.0 / stride            # steps per minute
    steps = 2 * count["strikes"]
    if steps and other_foot_due is not None and next_seq is not None and next_seq < other_foot_due:
        steps -= 1                          # the other foot's step hasn't been shown yet
    return {"steps": steps, "strikes": count["strikes"],
            "cadence_m": round(count["cadence_m"], 1), "zupt_m": round(count["zupt_m"], 1),
            "cadence_spm": round(cadence)}


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
        "has_net": net is not None,
        "counter": counter_view(),
        "trial_kinds": TRIAL_KINDS,
        "trial": {k: trial[k] for k in ("state", "id", "kind", "result")},
        "trials": saved_trials[-30:],
        "person": people["current"],
        "people": people["known"],
    }


def api_stream(since: int = -1):
    """Plot data newer than sample number `since` (everything kept if -1)."""
    # list() copies a deque in one step, so the loop thread can't change it mid-copy.
    samples = [s for s in list(plot_samples) if s[0] > since]
    preds = [p for p in list(plot_predictions) if p[0] > since]
    return {"hz": SAMPLE_HZ, "samples": samples, "predictions": preds,
            "prob": list(plot_prob), "strikes": list(plot_strikes)}


ui = WebUI()
ui.expose_api("POST", "/api/label/{n}", api_set_label)
ui.expose_api("POST", "/api/mode/{name}", api_set_mode)
ui.expose_api("POST", "/api/model/{name}", api_set_model)
ui.expose_api("POST", "/api/reset", api_reset)
ui.expose_api("POST", "/api/trial/start/{kind}", api_trial_start)
ui.expose_api("POST", "/api/trial/stop", api_trial_stop)
ui.expose_api("POST", "/api/trial/save/{steps}", api_trial_save)
ui.expose_api("POST", "/api/trial/discard", api_trial_discard)
ui.expose_api("POST", "/api/person/{name}", api_set_person)
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


# ---------- step counting ----------

def reset_count():
    global last_strike, other_foot_due
    count.update(strikes=0, cadence_m=0.0, zupt_m=0.0)
    pending_strides.clear()
    recent_strikes.clear()
    last_strike = None
    other_foot_due = None


def restart_stream():
    """Samples were lost or the sensor dropped: the net starts over from here."""
    global history_end, fed_until, tracker, last_strike
    history.clear()
    history_end = None
    fed_until = None
    tracker = stepnet.StrikeTracker(net.threshold if net else stepnet.STRIKE_THRESHOLD)
    pending_strides.clear()
    last_strike = None


def counts(strike):
    """Outside a trial every strike counts; in one, only those before Stop."""
    if trial["state"] in ("idle", "walking"):
        return True
    return trial["state"] == "settling" and strike < trial["stop_seq"]


def run_net():
    """Run the net on the newest NET_WINDOW samples and feed its final outputs to the tracker."""
    global fed_until, last_strike, other_foot_due
    n = min(len(history), NET_WINDOW)
    if n < 2 * net.context + CLASSIFY_EVERY:
        return
    start = history_end - n
    x = np.array(list(history)[-n:], dtype=np.float32)
    prob = net.probabilities(x)
    # Only samples with the net's full context on both sides are final.
    first = start + net.context if fed_until is None else max(fed_until, start + net.context)
    last = history_end - net.context
    for seq in range(first, last):
        p = float(prob[seq - start])
        if seq % PLOT_EVERY == 0:
            plot_prob.append([seq, round(p, 3)])
        strike = tracker.add(seq, p)
        if strike is None or not counts(strike):
            continue
        plot_strikes.append(strike)
        recent_strikes.append(strike)
        count["strikes"] += 1
        prev = last_strike if last_strike is not None and \
            (strike - last_strike) / SAMPLE_HZ <= stepnet.MAX_STRIDE_S else None
        pending_strides.append((prev, strike))
        last_strike = strike
        stride = strike - prev if prev is not None else stepnet.TYPICAL_STRIDE_S * SAMPLE_HZ
        other_foot_due = history_end + int(stride / 2)
    fed_until = last


def measure_strides():
    """Add the length of every stride whose next foot plant has now been seen."""
    while pending_strides:
        prev, strike = pending_strides[0]
        if prev is not None and history_end < strike + stepnet.stride_samples_needed(prev, strike):
            return
        pending_strides.popleft()
        start = history_end - len(history)
        if prev is not None and prev - 5 < start:
            prev = None                         # fell out of history: count it as a half stride
        if prev is None:
            c, z = stepnet.stride_length(None, None, strike, net.step_model)
        else:
            x = np.array(list(history), dtype=float)
            c, z = stepnet.stride_length(x, prev - start, strike - start, net.step_model, net.zupt_scale,
                                         net.run_model)
        count["cadence_m"] += c
        count["zupt_m"] += z


# ---------- trials ----------

def handle_requests():
    global trial_number, requested_label
    while requests:
        req = requests.popleft()
        if req[0] == "reset" and trial["state"] == "idle":
            reset_count()
        elif req[0] == "start" and trial["state"] == "idle":
            trial_number += 1
            reset_count()
            trial.update(state="walking", id=trial_number, kind=req[1],
                         start_seq=next_seq or 0, stop_seq=None, result=None)
            requested_label = LABELS.index("run" if req[1] == "run" else "walk")
            print(f"Trial {trial_number} ({req[1]}) started")
        elif req[0] == "stop" and trial["state"] == "walking":
            trial.update(state="settling", stop_seq=next_seq or 0)
            requested_label = LABELS.index("pause")
        elif req[0] == "save" and trial["state"] == "done":
            row = dict(trial["result"], counted_steps=req[1],
                       distance_m="" if req[2] is None else f"{req[2]:.2f}",
                       saved_at=f"{datetime.now():%Y-%m-%d %H:%M:%S}")
            new_file = not TRIALS_PATH.exists()
            with open(TRIALS_PATH, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=TRIAL_FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow(row)
            saved_trials.append({k: str(v) for k, v in row.items()})
            print(f"Trial {row['trial']} saved: counted {req[1]}, net {row['net_steps']}")
            trial.update(state="idle", id=0, kind=None, result=None)
            reset_count()
        elif req[0] == "person" and trial["state"] == "idle":
            people["current"] = req[1]
            if req[1] not in people["known"]:
                people["known"].append(req[1])
            PEOPLE_PATH.write_text(json.dumps(people))
            print(f"Wearer: {req[1]}")
        elif req[0] == "discard" and trial["state"] in ("walking", "settling", "done"):
            if trial["state"] != "done":
                requested_label = LABELS.index("pause")
            trial.update(state="idle", id=0, kind=None, result=None)
            reset_count()


def finish_trial():
    """After Stop, wait until the net has seen past the stop and every stride is measured."""
    if trial["state"] != "settling" or next_seq is None:
        return
    if next_seq < trial["stop_seq"] + STOP_SETTLE or (fed_until or 0) < trial["stop_seq"]:
        return
    # A stride still waiting for a next foot plant that never came (you
    # stopped walking) can't be integrated: use the cadence model for both.
    start = history_end - len(history)
    x = np.array(list(history), dtype=float)
    while pending_strides:
        prev, strike = pending_strides.popleft()
        if prev is None or prev < start:
            c, z = stepnet.stride_length(None, None, strike, net.step_model)
        else:
            c, z = stepnet.stride_length(x, prev - start, strike - start, net.step_model,
                                         run_model=net.run_model, zupt=False)
        count["cadence_m"] += c
        count["zupt_m"] += z
    trial["state"] = "done"
    trial["result"] = {
        "session": csv_path.name, "person": people["current"], "trial": trial["id"], "kind": trial["kind"],
        "start_t": f"{trial['start_seq'] / SAMPLE_HZ:.2f}", "end_t": f"{trial['stop_seq'] / SAMPLE_HZ:.2f}",
        "net_steps": 2 * count["strikes"],
        "cadence_m": f"{count['cadence_m']:.2f}", "zupt_m": f"{count['zupt_m']:.2f}",
    }
    print(f"Trial {trial['id']} done: net counted {2 * count['strikes']} steps")


def update_matrix():
    """Live mode: matrix shows the active model's answer. Record mode: the label."""
    global shown_matrix
    want = -1
    if mode != "record" and active_model and prediction.get(active_model):
        want = LABELS.index(prediction[active_model])
    if want != shown_matrix:
        Bridge.call("set_matrix", want)
        shown_matrix = want


def loop():
    global sensor_ok, next_seq, dropped, latest, label, requested_label
    global since_classify, _last_print, history_end
    handle_requests()
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
            restart_stream()
            for name in models:
                recent[name].clear()
                prediction[name] = None

    if rows:
        if next_seq is not None and seq > next_seq:
            dropped += seq - next_seq
            restart_stream()
        # Samples are tagged with the trial they belong to (0 = none). Stop
        # ends the trial at the sample it was pressed on.
        trial_id = trial["id"] if trial["state"] in ("walking", "settling") else 0
        for i, r in enumerate(rows):
            s = seq + i
            tid = trial_id if trial["stop_seq"] is None or s < trial["stop_seq"] else 0
            writer.writerow([f"{s / SAMPLE_HZ:.2f}", r[0], tid, people["current"]]
                            + [f"{x:.3f}" for x in r[1:]])
            seconds_recorded[r[0]] += 1 / SAMPLE_HZ
            window.append(r[1:])
            history.append(r[1:])
            if s % PLOT_EVERY == 0:
                plot_samples.append([s] + [round(x, 2) for x in r[1:]])
        # Flush every batch so a battery that runs flat loses at most ~50 ms.
        csv_file.flush()
        next_seq = seq + len(rows)
        history_end = next_seq
        latest = rows[-1]
        if latest[0] != label:
            label = latest[0]
            print(f"Recording label: {label}")

        since_classify += len(rows)
        if since_classify >= CLASSIFY_EVERY:
            since_classify = 0
            if models and len(window) == WINDOW_N:
                classify(next_seq - 1)
            if net:
                run_net()
        if net:
            measure_strides()
            finish_trial()

    try:
        update_matrix()
    except Exception as exc:
        print(f"Bridge call set_matrix failed: {exc}")

    now = time.monotonic()
    if ok and latest and now - _last_print >= PRINT_PERIOD_S:
        _last_print = now
        c = counter_view()
        guesses = "  ".join(f"{k}={v}" for k, v in prediction.items())
        print(f"steps {c['steps']:4d}  {c['zupt_m']:6.1f} m  |  {guesses}  |  "
              f"trial {trial['state']:8}  |  rec {label:5}  |  dropped {dropped}")
    time.sleep(POLL_PERIOD_S)


# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs).
try:
    App.run(user_loop=loop)
finally:
    csv_file.close()
