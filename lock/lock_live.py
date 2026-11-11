"""
The lock, running on the computer. SPACE stands in for the button.

    python lock/lock_live.py --set-code 315     (once: save a 3-digit code of different numbers 1-5)
    python lock/lock_live.py                     (run the lock)
    python lock/lock_live.py --count-only        (just show the finger count, for testing the model)

Lock policy (what lock_policy.py decides):
  LOCKED     waiting for the button.
  ENTERING   for each of the 3 digits: the same count (1-5) must be predicted with at least
             MIN_CONF confidence for HOLD_FRAMES frames in a row. Then the hand must leave
             (0 fingers / empty square) before the next digit is read, so holding up "3"
             doesn't type 3-3-3. If 20 s pass with no new digit, it gives up and stays locked.
  UNLOCKED   all 3 digits matched: servo opens; SPACE locks it again.
  DENIED     wrong code: stays locked. 3 wrong codes in a row -> 30 s lockout.
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch

import config
from lock_policy import LockPolicy
from model import load_model, predict


def draw_bar(view, probs, x0, y0):
    """Probability bar for each class, so you can see how sure the network is."""
    for k, p in enumerate(probs):
        y = y0 + 34 * k
        cv2.rectangle(view, (x0, y), (x0 + int(260 * p), y + 26), (0, 200, 255), -1)
        cv2.rectangle(view, (x0, y), (x0 + 260, y + 26), (255, 255, 255), 1)
        cv2.putText(view, f"{k}: {p:.0%}", (x0 + 270, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def text(view, s, org, scale=1.0, color=(255, 255, 255)):
    cv2.putText(view, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), int(5 * scale))
    cv2.putText(view, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, int(2 * scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-code", help="save a new 3-digit code, e.g. 315")
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--cam", type=int, default=config.CAM_INDEX)
    args = ap.parse_args()

    if args.set_code:
        digits = [int(c) for c in args.set_code]
        if len(digits) != 3 or len(set(digits)) != 3 or not all(1 <= d <= 5 for d in digits):
            raise SystemExit("The code must be 3 different numbers from 1 to 5, e.g. 315")
        config.CODE_PATH.write_text(json.dumps({"code": digits}))
        print(f"Saved code to {config.CODE_PATH}")
        return

    if not args.count_only and not config.CODE_PATH.exists():
        raise SystemExit("No code yet: run  python lock/lock_live.py --set-code 315  first")
    code = json.loads(config.CODE_PATH.read_text())["code"] if not args.count_only else None

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(config.MODEL_PATH, device)
    policy = LockPolicy(code) if code else None

    cap = cv2.VideoCapture(args.cam, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAP_H)
    x, y, s = config.load_roi()
    win = "lock"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        fingers, conf, probs = predict(model, frame[y:y + s, x:x + s], device)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        button = key == ord(" ")

        view = frame.copy()
        cv2.rectangle(view, (x, y), (x + s, y + s), (0, 0, 255), 3)
        text(view, f"{fingers} fingers ({conf:.0%})", (15, 45), 1.2, (0, 255, 255))
        draw_bar(view, probs, 15, 80)

        if policy:
            policy.update(fingers, conf, button, time.time())
            color = {"LOCKED": (255, 255, 255), "ENTERING": (0, 255, 255),
                     "UNLOCKED": (0, 255, 0), "DENIED": (0, 0, 255), "LOCKOUT": (0, 0, 255)}[policy.state]
            text(view, policy.state, (15, view.shape[0] - 110), 1.6, color)
            text(view, policy.message(), (15, view.shape[0] - 60), 1.0, color)
            # what the UNO Q's LED matrix will show: the digits so far, then a check or an X
            entered = " ".join(str(d) for d in policy.entered) + " _" * (3 - len(policy.entered))
            text(view, entered, (view.shape[1] - 330, 80), 2.0, color)
            if policy.state == "ENTERING" and policy.candidate:
                frac = policy.hold / policy.HOLD_FRAMES
                cv2.rectangle(view, (x, y + s + 10), (x + int(s * frac), y + s + 30), (0, 255, 255), -1)

        cv2.imshow(win, view)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
