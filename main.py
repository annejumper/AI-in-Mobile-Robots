"""PoseRace: control a LEGO Education race car's wheels with arm gestures.

Gesture scheme: hold your arms out and raise/lower each wrist relative to
your shoulder. Left wrist height sets the left wheel's speed, right wrist
height sets the right wheel's speed (both -100% to 100%, reverse included).
Push one arm up and the other down to spin in place -- handy for backing
out of tight spots.

Usage:
    python main.py                 # connect to real hardware over BLE
    python main.py --dry-run       # run the vision + gesture pipeline only,
                                    # print wheel speeds instead of driving
"""

import argparse
import time

import cv2

from poserace.car import Car
from poserace.gestures import GestureController, WheelSpeeds
from poserace.vision import PoseCamera

SEND_INTERVAL_SECONDS = 1 / 15  # cap BLE commands to ~15 Hz


def draw_hud(frame, landmarks, speeds):
    h, w = frame.shape[:2]

    if landmarks is not None:
        points_of_interest = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
        for idx in points_of_interest:
            lm = landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, (0, 255, 0), -1)
        # Arm lines: shoulder->elbow->wrist for each side
        for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
            pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in (shoulder, elbow, wrist)]
            cv2.line(frame, pts[0], pts[1], (0, 255, 0), 2)
            cv2.line(frame, pts[1], pts[2], (0, 255, 0), 2)

    # Speed bars
    for label, speed, x in (("L", speeds.left, 30), ("R", speeds.right, w - 60)):
        bar_top, bar_bottom, bar_mid = 60, h - 60, (60 + h - 60) // 2
        cv2.rectangle(frame, (x, bar_top), (x + 30, bar_bottom), (80, 80, 80), 2)
        fill_height = int(abs(speed) / 100 * (bar_bottom - bar_mid))
        if speed >= 0:
            cv2.rectangle(frame, (x, bar_mid - fill_height), (x + 30, bar_mid), (0, 200, 0), -1)
        else:
            cv2.rectangle(frame, (x, bar_mid), (x + 30, bar_mid + fill_height), (0, 0, 200), -1)
        cv2.line(frame, (x, bar_mid), (x + 30, bar_mid), (255, 255, 255), 1)
        cv2.putText(frame, f"{label} {speed:+.0f}%", (x - 10, bar_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, "q: quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Skip hardware, just show the vision pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    car = None
    if not args.dry_run:
        car = Car()
        if not car.connect():
            print("Exiting. Run with --dry-run to test the gesture pipeline without hardware.")
            return

    gesture_controller = GestureController()
    last_send_time = 0.0

    try:
        with PoseCamera(camera_index=args.camera, mirrored=True) as cam:
            while True:
                frame, landmarks = cam.read()
                if frame is None:
                    print("Camera read failed.")
                    break

                if landmarks is not None:
                    speeds = gesture_controller.update(landmarks, mirrored=cam.mirrored)
                else:
                    gesture_controller.reset()
                    speeds = WheelSpeeds(left=0.0, right=0.0)

                now = time.time()
                if car is not None and now - last_send_time >= SEND_INTERVAL_SECONDS:
                    car.drive_tank(speeds.left, speeds.right)
                    last_send_time = now

                draw_hud(frame, landmarks, speeds)
                cv2.imshow("PoseRace", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if car is not None:
            car.stop()
            car.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
