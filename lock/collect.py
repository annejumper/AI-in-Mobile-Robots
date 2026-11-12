"""
Collect training images of 0-5 fingers inside the red square.

    python lock/collect.py

  Drag with the mouse     set the square to your red square (saved to roi.json)
  0 1 2 3 4 5             choose which label you're recording
  SPACE                   start / stop recording (saves ~6 images a second)
  n                       next camera (if the window isn't showing the webcam)
  q                       quit

While recording, keep moving: slide the hand around, tilt it, use different finger
combinations (e.g. 2 = index+middle AND thumb+pinky), left and right hands, other people's hands.
For 0, record both a fist and an empty square.
"""
import time

import cv2

import config

drag = {"start": None, "end": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        drag["start"], drag["end"] = (x, y), (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drag["start"]:
        drag["end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drag["start"]:
        (x0, y0), (x1, y1) = drag["start"], (x, y)
        s = max(abs(x1 - x0), abs(y1 - y0))
        if s > 40:
            config.save_roi(min(x0, x1), min(y0, y1), s)
        drag["start"] = None


def counts():
    return {c: len(list((config.DATA_DIR / str(c)).glob("*.jpg"))) for c in config.CLASSES}


def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAP_H)
    return cap


def main():
    for c in config.CLASSES:
        (config.DATA_DIR / str(c)).mkdir(parents=True, exist_ok=True)
    cam = config.CAM_INDEX
    cap = open_cam(cam)
    win = "collect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    label, recording, last_save, n = 1, False, 0.0, counts()
    session = time.strftime("%m%d-%H%M%S")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        x, y, s = config.load_roi()
        now = time.time()
        if recording and now - last_save > 0.16:
            path = config.DATA_DIR / str(label) / f"{session}_{int(now * 1000)}.jpg"
            cv2.imwrite(str(path), frame[y:y + s, x:x + s])
            n[label] += 1
            last_save = now

        view = frame.copy()
        cv2.rectangle(view, (x, y), (x + s, y + s), (0, 0, 255) if recording else (0, 255, 0), 3)
        if drag["start"]:
            cv2.rectangle(view, drag["start"], drag["end"], (255, 255, 0), 2)
        status = f"label {label}  {'RECORDING' if recording else 'paused (SPACE)'}   camera {cam}"
        tally = "  ".join(f"{c}:{n[c]}" for c in config.CLASSES)
        for i, t in enumerate([status, tally]):
            cv2.putText(view, t, (15, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(view, t, (15, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        preview = cv2.resize(frame[y:y + s, x:x + s], (config.IMG_SIZE * 3, config.IMG_SIZE * 3),
                             interpolation=cv2.INTER_NEAREST)   # roughly what the network sees
        view[-preview.shape[0]:, -preview.shape[1]:] = preview
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("5"):
            label, recording = key - ord("0"), False
        if key == ord(" "):
            recording = not recording
            if recording:   # each recording is its own session, so training can hold whole ones out
                session = time.strftime("%m%d-%H%M%S")
        if key == ord("n"):
            cap.release()
            cam = (cam + 1) % 3
            cap = open_cam(cam)

    cap.release()
    cv2.destroyAllWindows()
    print("images per class:", counts())


if __name__ == "__main__":
    main()
