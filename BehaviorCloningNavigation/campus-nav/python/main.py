# Linux side of the campus navigator (built on the StepCounter).
#
# Pulls batches of raw MPU-6050 samples from the sketch over the Bridge,
# converts them to g and deg/s, and:
#   - counts strides with the StepCounter's neural net (stepnet.json): every
#     0.5 s it runs on the last 4 s and turns heel-strike probabilities into
#     strikes, one per stride of the sensor foot (steps = 2 x strides);
#   - tracks which way you're facing with the gyro (heading.py);
#   - measures every stride once its next foot plant is seen: distance and
#     height change (ZUPT), and how much you turned since the last stride.
#     A stride is the unit the navigator works in: one row per stride;
#   - records demonstration walks. Press Start on the phone, walk the route,
#     press Stop. Each walk is a folder in data/walks/ with the raw IMU
#     samples (so everything can be recomputed on the laptop), one row per
#     stride, and any marks you tapped on the way (optional: turns and
#     stairs are found from the heading and height afterwards).
#
# The web page in assets/ (http://<board IP>:7000) shows all of it.
#
# Axes as mounted on the ankle: X up the leg, Y forward, Z to the left.

import csv
import json
import os
import re
import shutil
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import policy as nav
import stepnet
from heading import HeadingTracker

SAMPLE_HZ = 100
ACCEL_LSB_PER_G = 2048.0    # +-16 g range
GYRO_LSB_PER_DPS = 16.4     # +-2000 deg/s range
# Subtracted from each accel reading. From the axis check on 2026-10-26:
# Z read +1.18 g pointing up and -0.85 g pointing down, so it sits 0.17 g high.
ACCEL_OFFSET_G = (0.0, 0.0, 0.17)
NET_EVERY = 50              # samples between net runs (0.5 s)
NET_WINDOW = 400            # samples the step net sees each time (4 s)
HISTORY_N = 800             # raw samples kept for measuring strides (8 s)
STOP_SETTLE = 150           # samples after Stop before the last strides are final
STAIR_DZ_M = 0.20           # a stride that rises or drops more than this is on stairs
POLL_PERIOD_S = 0.05        # ~5 samples per batch
PRINT_PERIOD_S = 2.0
SYNC_PERIOD_S = 1.0         # how often a walk's files are forced onto the storage card
MARK_KINDS = ["left", "right", "straight", "stairs", "door", "mark"]
# While navigating: "that was wrong, the right thing is ..." (DAgger corrections)
FIX_KINDS = [f"fix_{label}" for label in nav.LABELS]
WALK_KINDS = {"demo": "record", "dagger": "navigate", "test": "navigate"}
APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
WALKS_DIR = DATA_DIR / "walks"
TRASH_DIR = DATA_DIR / "trash"
PEOPLE_PATH = DATA_DIR / "people.json"
DEFAULT_PERSON = "Anne"
MAX_NAME = 40
HERE = Path(__file__).resolve().parent
STEPNET_PATH = HERE / "stepnet.json"
POLICY_PATH = HERE / "policy.json"
STRIDE_FIELDS = ["stride", "t_s", "stride_s", "dist_m", "cadence_m", "zupt_m", "dz_m",
                 "heading_deg", "dheading_deg", "stairs", "running", "said", "confidence"]

# Matrix icons in the sketch (set_matrix)
ICON = {"idle": 0, "rec": 1, "straight": 2, "left": 3, "right": 4, "stairs": 5, "arrive": 6,
        "stairs_up": 5, "stairs_down": 5, "uturn_left": 3, "uturn_right": 4}

net = stepnet.StepNet.load(STEPNET_PATH)
print("Step net loaded")

navigator = {"policy": None, "mtime": None}


def load_policy():
    """(Re)load policy.json if it changed: train_policy.py --deploy can push a new one any time."""
    try:
        mtime = POLICY_PATH.stat().st_mtime
    except OSError:
        navigator.update(policy=None, mtime=None)
        return None
    if mtime != navigator["mtime"]:
        try:
            navigator.update(policy=nav.Policy.load(POLICY_PATH), mtime=mtime)
            info = navigator["policy"].info
            print(f"Navigator loaded: trained {info.get('trained')} on {len(info.get('walks', []))} walks")
        except (OSError, ValueError, KeyError) as exc:
            print(f"Couldn't load {POLICY_PATH.name}: {exc}")
            navigator.update(policy=None, mtime=None)
    return navigator["policy"]


load_policy()
WALKS_DIR.mkdir(parents=True, exist_ok=True)

people = {"current": DEFAULT_PERSON, "known": [DEFAULT_PERSON], "route": ""}
if PEOPLE_PATH.exists():
    try:
        people.update(json.loads(PEOPLE_PATH.read_text()))
    except ValueError:
        print(f"Couldn't read {PEOPLE_PATH}, starting with {DEFAULT_PERSON}")
print(f"Wearer: {people['current']}")


def write_durably(path, text):
    """Write a small file so a power cut leaves either the old file or the whole new one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)                        # the rename itself
    finally:
        os.close(fd)


def recover_meta(d):
    """Rebuild a walk's summary from its files (the power went before it was saved)."""
    def rows(name):
        try:
            with open(d / name, newline="", errors="ignore") as fh:
                return [r for r in csv.DictReader(line.replace("\0", "") for line in fh)]
        except OSError:
            return []
    strides, marks = rows("strides.csv"), rows("marks.csv")
    ts = [float(r["t_s"]) for r in strides + marks if (r.get("t_s") or "").replace(".", "", 1).isdigit()]
    m = re.match(r"(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)_(.*)", d.name)
    started = f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}" if m else ""
    return {"id": d.name, "person": m[7] if m else "", "route": "", "kind": "demo", "started": started,
            "duration_s": round(max(ts, default=0.0), 1), "strides": len(strides),
            "steps": 2 * len(strides),
            "dist_m": round(sum(float(r.get("dist_m") or 0) for r in strides), 1),
            "stair_strides": sum(r.get("stairs") == "1" for r in strides), "marks": len(marks),
            "dropped": 0, "gyro_calibrated": True, "recovered": True}


def load_walks():
    walks = []
    for d in sorted(p for p in WALKS_DIR.iterdir() if p.is_dir()):
        try:
            walks.append(json.loads((d / "meta.json").read_text()))
            continue
        except (OSError, ValueError):
            pass
        meta = recover_meta(d)
        write_durably(d / "meta.json", json.dumps(meta, indent=1))
        walks.append(meta)
        print(f"Walk {d.name} had no summary (power cut?): rebuilt it")
    return walks


walks = load_walks()                # summaries for the page, oldest first

sensor_ok = None
next_seq = None                     # seq of the sample after the newest one received
dropped = 0
_last_print = 0.0
_last_sync = 0.0
last_saved = {"id": None}           # newest walk saved this power-on (the page says "safe to unplug")
since_net = 0

# Step counting state
history = deque(maxlen=HISTORY_N)   # [ax..gz] of the newest samples, for the net and ZUPT
headings = deque(maxlen=HISTORY_N)  # heading (deg) at each sample in history
tracker = stepnet.StrikeTracker(net.threshold)
heading = HeadingTracker()
fed_until = None                    # net output is final up to (not incl.) this seq
last_strike = None
last_strike_heading = None
pending_strides = deque()           # (prev strike, strike) waiting for enough samples
recent_strikes = deque(maxlen=5)
live = {"strides": 0, "dist_m": 0.0, "stairs": 0, "last": None}

# Recording state: idle -> recording (Start) -> settling (Stop) -> idle (saved)
rec = {"state": "idle", "id": None, "dir": None, "start_seq": None, "stop_seq": None,
       "start_time": None, "route": "", "kind": "demo", "marks": [], "strides": 0,
       "dist_m": 0.0, "stairs": 0, "dropped0": 0, "calibrated": False,
       # navigating: the walk's strides as the policy sees them, and what it said last
       "mode": "record", "walk_strides": [], "said": None, "confidence": 0.0, "votes": {}, "fixes": 0}
rec_files = {}
shown_matrix = None

# The page's requests run on the web server's thread, so they only leave
# requests here. loop() owns the Bridge and all the state and acts on them.
requests = deque()


# ---------- web API ----------

def clean(text, n=MAX_NAME):
    return " ".join(str(text).replace(",", " ").split())[:n]


def api_set_person(name: str):
    name = clean(name)
    if not name:
        return {"error": "type a name"}
    if rec["state"] != "idle":
        return {"error": "stop the walk first"}
    requests.append(("person", name))
    return {"ok": True, "person": name}


def api_walk_start(route: str = "", kind: str = "demo"):
    if rec["state"] != "idle":
        return {"error": "a walk is already being recorded"}
    if kind not in WALK_KINDS:
        return {"error": f"kind must be one of {', '.join(WALK_KINDS)}"}
    if WALK_KINDS[kind] == "navigate" and load_policy() is None:
        return {"error": "no navigator on the board yet: train one with train_policy.py --deploy"}
    requests.append(("start", clean(route, 60), kind))
    return {"ok": True}


def api_walk_stop():
    requests.append(("stop",))
    return {"ok": True}


def api_walk_discard():
    requests.append(("discard",))
    return {"ok": True}


def api_mark(kind: str):
    if kind not in MARK_KINDS + FIX_KINDS:
        return {"error": f"kind must be one of {', '.join(MARK_KINDS + FIX_KINDS)}"}
    if rec["state"] != "recording":
        return {"error": "not recording"}
    requests.append(("mark", kind))
    return {"ok": True}


def api_walk_delete(walk_id: str):
    if not re.fullmatch(r"[\w-]+", walk_id) or not (WALKS_DIR / walk_id).is_dir():
        return {"error": "no such walk"}
    requests.append(("delete", walk_id))
    return {"ok": True}


def api_reset():
    requests.append(("reset",))
    return {"ok": True}


def cadence_spm():
    s = list(recent_strikes)
    if len(s) >= 2 and next_seq is not None and next_seq - s[-1] < 3 * SAMPLE_HZ:
        stride = (s[-1] - s[0]) / (len(s) - 1) / SAMPLE_HZ
        return round(120.0 / stride)
    return 0


def api_status():
    elapsed = 0.0
    if rec["state"] != "idle" and next_seq is not None:
        elapsed = ((rec["stop_seq"] or next_seq) - rec["start_seq"]) / SAMPLE_HZ
    return {
        "sensor_ok": bool(sensor_ok),
        "gyro_calibrated": heading.bias_measured,
        "still": heading.is_still(),
        "dropped": dropped,
        "person": people["current"],
        "people": people["known"],
        "route": people.get("route", ""),
        "mark_kinds": MARK_KINDS,
        "live": {"steps": 2 * live["strides"], "strides": live["strides"],
                 "dist_m": round(live["dist_m"], 1), "stairs": live["stairs"],
                 "heading_deg": round(heading.heading, 1), "cadence_spm": cadence_spm(),
                 "last": live["last"]},
        "rec": {"state": rec["state"], "id": rec["id"], "route": rec["route"], "kind": rec["kind"],
                "elapsed_s": round(elapsed), "strides": rec["strides"],
                "dist_m": round(rec["dist_m"], 1), "stairs": rec["stairs"],
                "marks": rec["marks"][-8:], "mode": rec["mode"], "said": rec["said"],
                "confidence": round(rec["confidence"], 2), "votes": rec["votes"], "fixes": rec["fixes"]},
        "walks": walks[-30:],
        "last_saved": last_saved["id"],
        "fix_kinds": FIX_KINDS,
        "policy": navigator["policy"].info if navigator["policy"] else None,
    }


ui = WebUI()
ui.expose_api("POST", "/api/person/{name}", api_set_person)
ui.expose_api("POST", "/api/walk/start", api_walk_start)
ui.expose_api("POST", "/api/walk/stop", api_walk_stop)
ui.expose_api("POST", "/api/walk/discard", api_walk_discard)
ui.expose_api("POST", "/api/walk/delete/{walk_id}", api_walk_delete)
ui.expose_api("POST", "/api/mark/{kind}", api_mark)
ui.expose_api("POST", "/api/reset", api_reset)
ui.expose_api("GET", "/api/status", api_status)


# ---------- sensor ----------

def parse(batch):
    """'<ok>|<seq>|ax,ay,az,gx,gy,gz;...' -> (ok, seq, rows of [g..., deg/s...])."""
    ok, seq, body = batch.split("|", 2)
    rows = []
    for item in body.split(";"):
        if item:
            v = [int(x) for x in item.split(",")]
            rows.append([c / ACCEL_LSB_PER_G - off for c, off in zip(v[0:3], ACCEL_OFFSET_G)]
                        + [c / GYRO_LSB_PER_DPS for c in v[3:6]])
    return ok == "1", int(seq), rows


def restart_stream():
    """Samples were lost or the sensor dropped: the net starts over from here."""
    global fed_until, tracker, last_strike
    history.clear()
    headings.clear()
    fed_until = None
    tracker = stepnet.StrikeTracker(net.threshold)
    pending_strides.clear()
    last_strike = None


# ---------- strides ----------

def run_net():
    """Run the net on the newest samples and queue every new strike for measuring."""
    global fed_until, last_strike
    n = min(len(history), NET_WINDOW)
    if n < 2 * net.context + NET_EVERY:
        return
    start = next_seq - n
    prob = net.probabilities(np.array(list(history)[-n:], dtype=np.float32))
    first = start + net.context if fed_until is None else max(fed_until, start + net.context)
    last = next_seq - net.context
    for seq in range(first, last):
        strike = tracker.add(seq, float(prob[seq - start]))
        if strike is None:
            continue
        if rec["state"] == "settling" and strike >= rec["stop_seq"]:
            continue                        # after Stop: not part of the walk
        recent_strikes.append(strike)
        prev = last_strike if last_strike is not None and \
            (strike - last_strike) / SAMPLE_HZ <= stepnet.MAX_STRIDE_S else None
        pending_strides.append((prev, strike))
        last_strike = strike
    fed_until = last


def heading_at(seq):
    if not headings:
        return heading.heading
    start = next_seq - len(headings)
    return headings[min(max(seq - start, 0), len(headings) - 1)]


def measure_strides(final=False):
    """Measure every stride whose next foot plant has been seen (all of them if final)."""
    while pending_strides:
        prev, strike = pending_strides[0]
        need = stepnet.FIRST_NEED if prev is None else stepnet.stride_samples_needed(prev, strike)
        needs_next = next_seq < strike + need
        if needs_next and not final:
            return
        pending_strides.popleft()
        start = next_seq - len(history)
        if prev is not None and prev - 5 < start:
            prev = None                     # fell out of history: count it as a half stride
        x = np.array(list(history), dtype=float)
        m = stepnet.stride_motion(x, None if prev is None else prev - start, strike - start,
                                  net.step_model, net.zupt_scale, net.run_model, zupt=not needs_next)
        add_stride(strike, prev, m)


def add_stride(strike, prev, m):
    global last_strike_heading
    h = heading_at(strike)
    dh = 0.0 if last_strike_heading is None else h - last_strike_heading
    last_strike_heading = h
    stairs = abs(m["dz_m"]) >= STAIR_DZ_M
    live["strides"] += 1
    live["dist_m"] += m["dist_m"]
    live["stairs"] += stairs
    row = {"stride": live["strides"], "t_s": round(strike / SAMPLE_HZ, 2),
           "stride_s": round((strike - prev) / SAMPLE_HZ, 2) if prev is not None else "",
           "dist_m": round(m["dist_m"], 3), "cadence_m": round(m["cadence_m"], 3),
           "zupt_m": round(m["zupt_m"], 3), "dz_m": round(m["dz_m"], 3),
           "heading_deg": round(h, 1), "dheading_deg": round(dh, 1),
           "stairs": int(stairs), "running": int(m["running"])}
    live["last"] = row
    if rec["state"] in ("recording", "settling") and strike >= rec["start_seq"]:
        rec["strides"] += 1
        rec["dist_m"] += m["dist_m"]
        rec["stairs"] += stairs
        # Within a walk the first stride's turn counts as 0, as in training (walklib)
        ws = rec["walk_strides"]
        wdh = round(h - ws[-1]["heading_deg"], 1) if ws else 0.0
        ws.append({"dist_m": m["dist_m"], "dz_m": m["dz_m"], "heading_deg": h, "dheading_deg": wdh})
        if rec["mode"] == "navigate":
            predict()
        rec_files["strides"].writerow(dict(
            row, stride=rec["strides"], t_s=round((strike - rec["start_seq"]) / SAMPLE_HZ, 2),
            dheading_deg=wdh, said=rec["said"] or "",
            confidence=round(rec["confidence"], 2) if rec["mode"] == "navigate" else ""))


def predict():
    """Ask the navigator what to do next, given the walk's strides so far."""
    p = navigator["policy"]
    if p is None:
        return
    said, conf, votes = p.predict(rec["walk_strides"])
    if said != rec["said"]:
        print(f"Navigator: {said} ({conf:.0%}) after stride {len(rec['walk_strides'])}")
    rec.update(said=said, confidence=conf, votes=votes)


# ---------- recording ----------

def save_people():
    PEOPLE_PATH.write_text(json.dumps(people))


def walk_meta():
    return {"id": rec["id"], "person": people["current"], "route": rec["route"], "kind": rec["kind"],
            "started": rec["start_time"],
            "duration_s": round((rec["stop_seq"] - rec["start_seq"]) / SAMPLE_HZ, 1),
            "strides": rec["strides"], "steps": 2 * rec["strides"], "dist_m": round(rec["dist_m"], 1),
            "stair_strides": rec["stairs"], "marks": len(rec["marks"]),
            "dropped": dropped - rec["dropped0"], "gyro_calibrated": rec["calibrated"],
            "mode": rec["mode"], "fixes": rec["fixes"],
            "policy": navigator["policy"].info if rec["mode"] == "navigate" and navigator["policy"] else None}


def start_walk(route, kind):
    now = datetime.now()
    wid = f"{now:%Y%m%d_%H%M%S}_{re.sub(r'[^A-Za-z0-9]+', '', people['current'])[:12] or 'walk'}"
    d = WALKS_DIR / wid
    d.mkdir(parents=True)
    imu = open(d / "imu.csv", "w", newline="")
    strides = open(d / "strides.csv", "w", newline="")
    marks = open(d / "marks.csv", "w", newline="")
    rec_files.update(
        imu_fh=imu, strides_fh=strides, marks_fh=marks,
        imu=csv.writer(imu), strides=csv.DictWriter(strides, fieldnames=STRIDE_FIELDS),
        marks=csv.writer(marks))
    rec_files["imu"].writerow(["t_s", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps", "heading_deg"])
    rec_files["strides"].writeheader()
    rec_files["marks"].writerow(["t_s", "kind", "stride", "heading_deg"])
    heading.reset()
    rec.update(state="recording", id=wid, dir=d, start_seq=next_seq or 0, stop_seq=None,
               start_time=f"{now:%Y-%m-%d %H:%M:%S}", route=route, kind=kind, marks=[],
               strides=0, dist_m=0.0, stairs=0, dropped0=dropped, calibrated=heading.bias_measured,
               mode=WALK_KINDS[kind], walk_strides=[], said=None, confidence=0.0, votes={}, fixes=0)
    if rec["mode"] == "navigate":
        load_policy()
        predict()                           # what to do before the first stride
    people["route"] = route
    save_people()
    print(f"Walk {wid} started ({kind}, route '{route}')")


def sync_walk_files(force=False):
    """Push the walk's files all the way to the storage card about once a second.

    flush() only hands data to Linux, which keeps it in memory for several
    seconds before writing it out; pulling the power (e.g. swapping the
    battery for USB) loses it. The first practice walk lost its last 20 s
    that way. fsync() makes Linux write it out now.
    """
    global _last_sync
    now = time.monotonic()
    for k in ("imu_fh", "strides_fh", "marks_fh"):
        rec_files[k].flush()
    if force or now - _last_sync >= SYNC_PERIOD_S:
        _last_sync = now
        for k in ("imu_fh", "strides_fh", "marks_fh"):
            os.fsync(rec_files[k].fileno())


def close_walk(keep):
    sync_walk_files(force=True)
    for k in ("imu_fh", "strides_fh", "marks_fh"):
        rec_files[k].close()
    rec_files.clear()
    if keep:
        meta = walk_meta()
        write_durably(rec["dir"] / "meta.json", json.dumps(meta, indent=1))
        walks.append(meta)
        last_saved["id"] = rec["id"]
        print(f"Walk {rec['id']} saved: {meta['steps']} steps, {meta['dist_m']} m, {meta['marks']} marks")
    else:
        shutil.rmtree(rec["dir"], ignore_errors=True)
        print(f"Walk {rec['id']} discarded")
    rec.update(state="idle", id=None, dir=None, start_seq=None, stop_seq=None, marks=[])


def handle_requests():
    while requests:
        req = requests.popleft()
        kind = req[0]
        if kind == "person" and rec["state"] == "idle":
            people["current"] = req[1]
            if req[1] not in people["known"]:
                people["known"].append(req[1])
            save_people()
            print(f"Wearer: {req[1]}")
        elif kind == "start" and rec["state"] == "idle":
            start_walk(req[1], req[2])
        elif kind == "stop" and rec["state"] == "recording":
            rec.update(state="settling", stop_seq=next_seq or 0)
        elif kind == "discard" and rec["state"] != "idle":
            close_walk(keep=False)
        elif kind == "mark" and rec["state"] == "recording":
            t = ((next_seq or 0) - rec["start_seq"]) / SAMPLE_HZ
            m = {"t_s": round(t, 2), "kind": req[1], "stride": rec["strides"],
                 "heading_deg": round(heading.heading, 1)}
            rec["marks"].append(m)
            if req[1].startswith("fix_"):
                rec["fixes"] += 1
                print(f"Correction after stride {rec['strides']}: said {rec['said']}, should be {req[1][4:]}")
            rec_files["marks"].writerow([m["t_s"], m["kind"], m["stride"], m["heading_deg"]])
            rec_files["marks_fh"].flush()
        elif kind == "delete":
            wid = req[1]
            if wid != rec["id"] and (WALKS_DIR / wid).is_dir():
                TRASH_DIR.mkdir(exist_ok=True)
                shutil.move(str(WALKS_DIR / wid), str(TRASH_DIR / wid))
                walks[:] = [w for w in walks if w.get("id") != wid]
                print(f"Walk {wid} moved to data/trash/")
        elif kind == "reset" and rec["state"] == "idle":
            live.update(strides=0, dist_m=0.0, stairs=0, last=None)
            heading.reset()


def finish_walk():
    """After Stop, wait until the net has seen past the stop, then measure what's left and save."""
    if rec["state"] != "settling" or next_seq is None:
        return
    if next_seq < rec["stop_seq"] + STOP_SETTLE or (fed_until or 0) < rec["stop_seq"]:
        return
    measure_strides(final=True)
    close_walk(keep=True)


def update_matrix():
    global shown_matrix
    if rec["state"] == "idle":
        want = ICON["idle"]
    elif rec["mode"] == "navigate" and rec["said"]:
        want = ICON[rec["said"]]
    else:
        want = ICON["rec"]
    if want != shown_matrix:
        Bridge.call("set_matrix", want)
        shown_matrix = want


def loop():
    global sensor_ok, next_seq, dropped, since_net, _last_print
    handle_requests()
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
            restart_stream()

    if rows:
        if next_seq is not None and seq > next_seq:
            dropped += seq - next_seq
            restart_stream()
        recording = rec["state"] in ("recording", "settling")
        for i, r in enumerate(rows):
            s = seq + i
            h = heading.add(r)
            history.append(r)
            headings.append(h)
            if recording and (rec["stop_seq"] is None or s < rec["stop_seq"]):
                rec_files["imu"].writerow([f"{(s - rec['start_seq']) / SAMPLE_HZ:.2f}"]
                                          + [f"{x:.3f}" for x in r] + [f"{h:.2f}"])
        if recording:
            sync_walk_files()
        next_seq = seq + len(rows)

        since_net += len(rows)
        if since_net >= NET_EVERY:
            since_net = 0
            run_net()
        measure_strides()
        finish_walk()

    try:
        update_matrix()
    except Exception as exc:
        print(f"Bridge call set_matrix failed: {exc}")

    now = time.monotonic()
    if ok and now - _last_print >= PRINT_PERIOD_S:
        _last_print = now
        print(f"steps {2 * live['strides']:4d}  {live['dist_m']:6.1f} m  heading {heading.heading:7.1f}  "
              f"stairs {live['stairs']:3d}  |  {rec['state']:9}  |  "
              f"gyro {'ok' if heading.bias_measured else 'uncal'}  |  dropped {dropped}")
    time.sleep(POLL_PERIOD_S)


# App.run() exits via SystemExit on stop, so cleanup must be in a finally
# wrapped around it (code after App.run() never runs). A walk still being
# recorded is saved with what it has.
try:
    App.run(user_loop=loop)
finally:
    if rec_files:
        rec["stop_seq"] = rec["stop_seq"] or next_seq or 0
        close_walk(keep=True)
