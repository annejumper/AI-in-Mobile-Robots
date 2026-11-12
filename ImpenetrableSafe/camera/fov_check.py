"""
Field-of-view check for the lock's webcam.

Shows the live camera with a grid, then measures the field of view from a ruler:

  1. Mount the camera on the stand, pointing straight down at the desk.
  2. Measure the height H from the lens to the desk (cm) and lay a ruler flat in view.
  3. Run:   python3 fov_check.py --height-cm 30
  4. Click two marks on the ruler a known distance apart (default 10 cm, change with --ruler-cm).

From the clicks:
    cm per pixel        = ruler_cm / pixel distance between clicks
    visible width  (cm) = image width  (px) × cm per pixel
    visible height (cm) = image height (px) × cm per pixel
    horizontal FOV      = 2 · atan( visible width  / (2H) )
    vertical FOV        = 2 · atan( visible height / (2H) )

The largest square the camera can see is (visible height) × (visible height): the most the
red square on the desk can be. Draw it a bit smaller so the edges stay in view.

Keys:   n  next camera     g  grid on/off     c  clear clicks
        s  save snapshot   q  quit
"""
import argparse
import math
import time
from pathlib import Path

import cv2

HERE = Path(__file__).parent
clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) == 2:
            clicks.clear()
        clicks.append((x, y))


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def measure(w_px, h_px, ruler_cm, height_cm):
    """The FOV numbers from the two clicks, or None if there aren't two yet."""
    if len(clicks) < 2:
        return None
    (x1, y1), (x2, y2) = clicks
    px = math.hypot(x2 - x1, y2 - y1)
    if px < 1:
        return None
    cm_per_px = ruler_cm / px
    vis_w, vis_h = w_px * cm_per_px, h_px * cm_per_px
    out = {"cm/px": cm_per_px, "visible_w": vis_w, "visible_h": vis_h}
    if height_cm:
        out["hfov"] = math.degrees(2 * math.atan(vis_w / (2 * height_cm)))
        out["vfov"] = math.degrees(2 * math.atan(vis_h / (2 * height_cm)))
    return out


def draw(frame, cam, show_grid, m, ruler_cm):
    h, w = frame.shape[:2]
    if show_grid:
        for k in range(1, 10):   # grid every 10%
            cv2.line(frame, (w * k // 10, 0), (w * k // 10, h), (200, 200, 200), 1)
            cv2.line(frame, (0, h * k // 10), (w, h * k // 10), (200, 200, 200), 1)
        cv2.line(frame, (w // 2 - 30, h // 2), (w // 2 + 30, h // 2), (0, 255, 255), 2)
        cv2.line(frame, (w // 2, h // 2 - 30), (w // 2, h // 2 + 30), (0, 255, 255), 2)
        # largest centred square: what a square crop for the network would keep
        s = min(w, h)
        x0, y0 = (w - s) // 2, (h - s) // 2
        cv2.rectangle(frame, (x0, y0), (x0 + s - 1, y0 + s - 1), (0, 0, 255), 2)

    for p in clicks:
        cv2.circle(frame, p, 8, (0, 255, 0), 2)
    if len(clicks) == 2:
        cv2.line(frame, clicks[0], clicks[1], (0, 255, 0), 2)

    lines = [f"camera {cam}   {w}x{h}   (n = next camera)"]
    if m:
        lines.append(f"{ruler_cm:g} cm = {ruler_cm / m['cm/px']:.0f} px   ->  {m['cm/px'] * 10:.3f} mm/px")
        lines.append(f"visible area: {m['visible_w']:.1f} x {m['visible_h']:.1f} cm")
        if "hfov" in m:
            lines.append(f"FOV: {m['hfov']:.1f} deg horizontal, {m['vfov']:.1f} deg vertical")
        lines.append(f"largest square on desk: {m['visible_h']:.1f} x {m['visible_h']:.1f} cm")
    else:
        lines.append(f"click two ruler marks {ruler_cm:g} cm apart")
    for i, text in enumerate(lines):
        y = 35 + 35 * i
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index (press n to cycle)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--ruler-cm", type=float, default=10.0, help="distance between your two clicks")
    ap.add_argument("--height-cm", type=float, default=None, help="lens-to-desk height, for the FOV angles")
    args = ap.parse_args()

    cam = args.cam
    cap = open_camera(cam, args.width, args.height)
    win = "FOV check"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    show_grid, last_print = True, None

    while True:
        ok, frame = cap.read()
        if not ok:
            print(f"camera {cam} gave no frame, trying the next one")
            cap.release()
            cam = (cam + 1) % 4
            cap = open_camera(cam, args.width, args.height)
            time.sleep(0.5)
            continue
        raw = frame.copy()
        h, w = frame.shape[:2]
        m = measure(w, h, args.ruler_cm, args.height_cm)
        draw(frame, cam, show_grid, m, args.ruler_cm)
        cv2.imshow(win, frame)

        if m and m != last_print:
            print(f"\ncamera {cam}, {w}x{h}:  {m['cm/px'] * 10:.3f} mm per pixel, "
                  f"sees {m['visible_w']:.1f} x {m['visible_h']:.1f} cm")
            if "hfov" in m:
                print(f"  FOV {m['hfov']:.1f}° horizontal, {m['vfov']:.1f}° vertical "
                      f"(diagonal {math.degrees(2 * math.atan(math.hypot(m['visible_w'], m['visible_h']) / (2 * args.height_cm))):.1f}°)")
            last_print = m

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("g"):
            show_grid = not show_grid
        if key == ord("c"):
            clicks.clear()
        if key == ord("s"):
            path = HERE / f"snapshot_cam{cam}_{time.strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(path), raw)
            print(f"saved {path}")
        if key == ord("n"):
            cap.release()
            cam = (cam + 1) % 4
            clicks.clear()
            cap = open_camera(cam, args.width, args.height)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
