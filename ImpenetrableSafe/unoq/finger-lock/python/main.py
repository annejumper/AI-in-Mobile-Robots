# Linux side of the finger-code safe (runs on the UNO Q).
#
# Every frame: webcam -> crop the red square -> 64x64 -> CNN (ONNX) -> finger count.
#
# Two modes:
#   count  shows the live finger count on the LED matrix and prints it with the frame rate.
#   lock   the safe: button -> enter 3 digits -> check or X, servo opens or stays locked.
#          The decisions come from lock_policy.py, the same file the laptop version uses.
#
# Web page: http://<board IP>:7000  - live camera with the crop square, what the network sees,
# the count and confidences, the lock state, and buttons (mode, virtual button, lock/unlock).
# The video is only encoded while someone is watching, to save the board's CPU.
#
# Commands written to cmd.txt in the app folder (tools/latch.py does this from the laptop):
#   mode count | mode lock | lock | unlock | angle <n> | press | snap | record <0-5> | record stop
#
# Recording: "record 3" saves the crop ~6 times a second to record/3/ in the app folder, named
# like lock/collect.py does, so tools/latch.py pull can drop them straight into data/ for training.
#
# Files copied in by lock/export_to_unoq.py on the laptop:
#   finger_cnn.onnx, settings.json (camera size, crop square), lock_policy.py, lock_code.json

import json
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge
from fastapi.responses import StreamingResponse

from lock_policy import LockPolicy

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
CMD_FILE = APP_DIR / "cmd.txt"
MODE_FILE = APP_DIR / "mode.txt"   # remembers the mode across restarts
RECORD_DIR = APP_DIR / "record"     # training pictures taken with the board's camera

S = json.loads((HERE / "settings.json").read_text())
ROI = S["roi"]
IMG = S["img_size"]
CODE = json.loads((HERE / "lock_code.json").read_text())["code"]

sess = ort.InferenceSession(str(HERE / "finger_cnn.onnx"), providers=["CPUExecutionProvider"])


def open_camera():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, S["cap_w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, S["cap_h"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def crop(frame):
    x, y, s = ROI["x"], ROI["y"], ROI["size"]
    return frame[y:y + s, x:x + s]


def predict(bgr_crop):
    """Same steps as lock/model.py preprocess(): resize, BGR->RGB, scale to [-1, 1], channels first."""
    small = cv2.resize(bgr_crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((img - 0.5) / 0.5).transpose(2, 0, 1)[None]
    scores = sess.run(None, {"image": x})[0][0]
    p = np.exp(scores - scores.max())
    p /= p.sum()
    k = int(p.argmax())
    return S["classes"][k], float(p[k]), p, small


def pack(digits, preview=None, underscores=0):
    """Digits for show_digits(): slot = digit + 1, dim preview = digit + 8, 7 = '_', 0 = blank."""
    slots = [d + 1 for d in digits] + ([preview + 8] if preview is not None else []) + [7] * underscores
    slots = ([0] * (3 - len(slots)) + slots) if len(slots) == 1 else (slots + [0] * (3 - len(slots)))
    return (slots[0] << 8) | (slots[1] << 4) | slots[2]


# ── State shared with the web server's threads ──────────────────────────────
# The web handlers only read `status`/the latest JPEG and queue requests; loop() owns
# the camera, the Bridge and the policy.

requests = deque()
status = {}
view = threading.Condition()
view_jpeg, view_seq, viewers = None, 0, 0


def api_status():
    return status


def api_press():
    requests.append("press")
    return {"ok": True}


def api_mode(name: str):
    if name not in ("count", "lock"):
        return {"error": "mode must be count or lock"}
    requests.append(f"mode {name}")
    return {"ok": True}


def api_latch(action: str):
    if action not in ("lock", "unlock"):
        return {"error": "use lock or unlock"}
    requests.append(action)
    return {"ok": True}


def api_record(label: str):
    if label != "stop" and label not in [str(c) for c in S["classes"]]:
        return {"error": "use 0-5 or stop"}
    requests.append(f"record {label}")
    return {"ok": True}


def api_stream():
    def frames():
        global viewers
        with view:
            viewers += 1
        try:
            seen = -1
            while True:
                with view:
                    view.wait_for(lambda: view_seq != seen, timeout=2.0)
                    jpeg, seen = view_jpeg, view_seq
                if jpeg is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        finally:
            with view:
                viewers -= 1

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


ui = WebUI()
ui.expose_api("GET", "/api/status", api_status)
ui.expose_api("GET", "/stream", api_stream)
ui.expose_api("POST", "/api/press", api_press)
ui.expose_api("POST", "/api/mode/{name}", api_mode)
ui.expose_api("POST", "/api/latch/{action}", api_latch)
ui.expose_api("POST", "/api/record/{label}", api_record)


# ── Main loop ───────────────────────────────────────────────────────────────

mode = MODE_FILE.read_text().strip() if MODE_FILE.exists() else "count"
policy = LockPolicy(CODE)
cap = open_camera()
last_presses = None
shown = None          # last thing sent to the matrix, so it's only redrawn on a change
servo_locked = None
last_state = None
frames, t_report, fps = 0, time.time(), 0.0
last_frame = None
rec = {"label": None, "session": "", "last": 0.0, "count": 0}


def record_counts():
    return {str(c): len(list((RECORD_DIR / str(c)).glob("*.jpg"))) for c in S["classes"]}


print(f"Mode: {mode}.  Camera open: {cap.isOpened()}.  Code has {len(CODE)} digits.")


def show(kind, value):
    global shown
    if shown != (kind, value):
        Bridge.call("show_icon" if kind == "icon" else "show_digits", value)
        shown = (kind, value)


def set_servo(locked):
    global servo_locked
    if servo_locked != locked:
        Bridge.call("set_locked", 1 if locked else 0)
        servo_locked = locked


def run_command(cmd):
    """Returns True if the command was a (virtual) button press."""
    global mode, shown, servo_locked
    words = cmd.split()
    if not words:
        return False
    if words[0] == "press":
        print("command: press (virtual button)")
        return True
    if words[0] == "mode" and len(words) == 2 and words[1] in ("count", "lock"):
        mode = words[1]
        MODE_FILE.write_text(mode)
        policy.__init__(CODE)
        shown = servo_locked = None
    elif words[0] in ("lock", "unlock"):
        Bridge.call("set_locked", 1 if words[0] == "lock" else 0)
        servo_locked = words[0] == "lock"
    elif words[0] == "angle" and len(words) == 2 and words[1].isdigit():
        Bridge.call("set_angle", int(words[1]))
        servo_locked = None
    elif words[0] == "record" and len(words) == 2 and words[1] == "stop":
        print(f"recording stopped: {rec['count']} pictures of {rec['label']}  (on the board: {record_counts()})")
        rec["label"] = None
    elif words[0] == "record" and len(words) == 2 and words[1] in [str(c) for c in S["classes"]]:
        # a new session each time, so training can hold whole recordings out for validation
        rec.update(label=words[1], session="b" + time.strftime("%m%d-%H%M%S"), count=0)
        (RECORD_DIR / words[1]).mkdir(parents=True, exist_ok=True)
    elif words[0] == "snap" and last_frame is not None:
        cv2.imwrite(str(APP_DIR / "snap_frame.jpg"), last_frame)
        cv2.imwrite(str(APP_DIR / "snap_crop.jpg"), crop(last_frame))
    else:
        print(f"Unknown command: {cmd}")
        return False
    print(f"command: {cmd}")
    return False


def pending_commands():
    cmds = []
    if CMD_FILE.exists():
        cmds.append(CMD_FILE.read_text().strip())
        CMD_FILE.unlink()
    while requests:
        cmds.append(requests.popleft())
    return cmds


def publish_view(frame, small, fingers, conf, state_text, color):
    """Annotated half-size frame with the network's 64x64 input in the corner, as JPEG."""
    global view_jpeg, view_seq
    v = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2), interpolation=cv2.INTER_AREA)
    x, y, s = ROI["x"] // 2, ROI["y"] // 2, ROI["size"] // 2
    cv2.rectangle(v, (x, y), (x + s, y + s), (0, 0, 255), 2)
    inset = cv2.resize(small, (128, 128), interpolation=cv2.INTER_NEAREST)
    v[8:136, v.shape[1] - 136:v.shape[1] - 8] = inset
    cv2.rectangle(v, (v.shape[1] - 137, 7), (v.shape[1] - 7, 137), (255, 255, 255), 1)
    for text, org, scale, col in ((f"{fingers} ({conf:.0%})", (10, 34), 1.0, (0, 255, 255)),
                                  (state_text, (10, v.shape[0] - 14), 0.8, color)):
        cv2.putText(v, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(v, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)
    ok, jpeg = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with view:
            view_jpeg, view_seq = jpeg.tobytes(), view_seq + 1
            view.notify_all()


def loop():
    global cap, last_presses, last_state, frames, t_report, fps, last_frame

    virtual_press = any([run_command(c) for c in pending_commands()])

    ok, frame = cap.read()
    if not ok:
        print("No camera frame; reopening the camera")
        cap.release()
        time.sleep(1)
        cap = open_camera()
        return
    last_frame = frame
    fingers, conf, probs, small = predict(crop(frame))
    frames += 1

    if rec["label"] is not None and time.time() - rec["last"] > 0.16:
        rec["last"] = time.time()
        cv2.imwrite(str(RECORD_DIR / rec["label"] / f"{rec['session']}_{int(rec['last'] * 1000)}.jpg"), crop(frame))
        rec["count"] += 1

    presses = Bridge.call("get_presses")
    button = (last_presses is not None and presses != last_presses) or virtual_press
    last_presses = presses

    now = time.time()
    if now - t_report >= 1.0:
        fps = frames / (now - t_report)

    if mode == "count":
        show("digits", pack([fingers]) if conf >= policy.MIN_CONF else 0)
        state_text, color = "COUNT MODE", (255, 255, 255)
        if now - t_report >= 1.0:
            print(f"{fingers} fingers ({conf:.0%})   {fps:.1f} frames/s   "
                  f"[{' '.join(f'{p:.2f}' for p in probs)}]")
    else:
        policy.update(fingers, conf, button, now)
        st = policy.state
        if st == "ENTERING":
            pre = policy.candidate if policy.CONFIRM_WITH_BUTTON else None
            show("digits", pack(policy.entered, pre, 3 - len(policy.entered) - (pre is not None)))
        elif st == "UNLOCKED":
            show("icon", 1)
        else:                      # LOCKED, DENIED, LOCKOUT
            show("icon", 0)
        set_servo(st != "UNLOCKED")
        state_text = f"{st}  {'* ' * len(policy.entered)}"   # the stream never shows the digits
        color = {"UNLOCKED": (0, 255, 0), "ENTERING": (0, 255, 255)}.get(st, (0, 0, 255) if st != "LOCKED" else (255, 255, 255))
        if st != last_state:
            print(f"{st:8}  entered {policy.entered}  seeing {fingers} ({conf:.0%})  - {policy.message()}")
            last_state = st

    if now - t_report >= 1.0:
        frames, t_report = 0, now

    status.update({
        "mode": mode, "fingers": fingers, "confidence": round(conf, 3),
        "probs": [round(float(p), 3) for p in probs], "fps": round(fps, 1),
        "state": policy.state if mode == "lock" else "COUNT",
        "message": policy.message() if mode == "lock" else "showing the live count on the LED matrix",
        "entered": len(policy.entered) if mode == "lock" else 0,
        "min_conf": policy.MIN_CONF,
        "hold": policy.hold / policy.HOLD_FRAMES if mode == "lock" and policy.candidate else 0,
        "recording": rec["label"], "rec_count": rec["count"],
    })

    if rec["label"] is not None:
        state_text = f"RECORDING {rec['label']}  ({rec['count']})"
        color = (0, 0, 255)
    if viewers:
        publish_view(frame, small, fingers, conf, state_text, color)


App.run(user_loop=loop)
