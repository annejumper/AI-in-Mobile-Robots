"""Center the car in front of a stationary AprilTag using a phone camera
mounted on the car, driven by two LEGO Education motors.

Physical setup:
  - A Double Motor drives the car back and forth on the ground.
  - A Single Motor sits on top and pans the phone mount so the camera
    stays aimed at the (stationary) AprilTag as the car moves.
  - The AprilTag itself is stationary (e.g. taped near the computer's
    webcam), NOT on the car.
  - The phone streams video back to this computer, which finds the tag
    and drives both motors.

Two nested proportional control loops:
  - Pan loop (fast, visual): the tag's horizontal pixel error in the phone's
    camera frame drives the pan motor, keeping the tag centered in frame
    regardless of where the car currently is.
  - Drive loop (slow, mechanical): the pan motor's accumulated rotation
    away from its zeroed "facing forward" position drives the car. If the
    pan motor has had to turn to keep tracking the tag, the car itself
    must be off to one side -- driving the car to bring the pan angle back
    to zero re-centers the car under the tag.

IMPORTANT: before running (without --dry-run), physically aim the phone
straight ahead (perpendicular to the car's forward axis, facing the tag)
-- the script zeroes the pan motor's relative position at startup and
assumes that starting orientation is "straight ahead."

Camera feed into OpenCV, one of two ways:
  * IP-camera app over Wi-Fi (e.g. "IP Webcam" on Android) -- pass its
    stream URL with --url.
  * macOS Continuity Camera (iPhone) or any local webcam -- pass its
    device index with --camera-index (use view_stream.py --list to find
    the right index, and --rotate if it's mounted vertically).

Usage:
    python park_control.py --camera-index 1 --rotate 90 --dry-run   # vision only, no motors
    python park_control.py --camera-index 1 --rotate 90             # full run, both motors
"""

import argparse
import sys
import time

import cv2
import legoeducation as le

DEFAULT_DICT = "DICT_APRILTAG_36h11"

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class ProportionalController:
    """output = clip(gain * error, -max_out, max_out), zero inside a
    deadzone around error=0, floored above a minimum magnitude so the
    motor can actually overcome static friction."""

    def __init__(self, deadzone, max_out, min_out, gain, invert=False):
        self.deadzone = deadzone
        self.max_out = max_out
        self.min_out = min_out
        self.gain = -gain if invert else gain

    def output_for(self, error):
        if abs(error) < self.deadzone:
            return 0
        out = max(-self.max_out, min(self.max_out, self.gain * error))
        if 0 < abs(out) < self.min_out:
            out = self.min_out if out > 0 else -self.min_out
        return int(out)


def find_tag(gray, detector, tag_id):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for pts, found_id in zip(corners, ids.flatten()):
        if tag_id is not None and found_id != tag_id:
            continue
        return pts.reshape(4, 2)
    return None


def resolve_card_color(name):
    if name is None:
        return None
    attr = f"LEGO_COLOR_{name.upper()}"
    if not hasattr(le, attr):
        raise SystemExit(f"Unknown card color '{name}'. See le.LEGO_COLOR_NAME_MAP for valid names.")
    return getattr(le, attr)


def connect_motor(motor, label, card_color, card_serial):
    print(f"Connecting to {label} over Bluetooth...")
    motor.connect(card_color=card_color, card_serial=card_serial)
    if not motor.connected:
        sys.exit(f"Error connecting to {label}. Check it's powered on and card_color/card_serial match its Connection Card.")
    print(f"{label} connected.")


def read_tag_cx(cap, detector, tag_id, rotation, attempts=30):
    """Grab frames until the tag is found, and return its centroid x. None if not found."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if not ok:
            continue
        if rotation is not None:
            frame = cv2.rotate(frame, rotation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tag = find_tag(gray, detector, tag_id)
        if tag is not None:
            return tag.mean(axis=0)[0]
    return None


def calibrate_pan_sign(cap, detector, tag_id, rotation, pan_motor, fallback_invert, test_speed=25, test_duration=0.4):
    """Nudge the pan motor clockwise by a known amount and measure which way
    the tag's pixel position actually moves, instead of guessing --pan-invert
    by hand. Returns True/False for invert, or fallback_invert if the tag
    couldn't be tracked through the test."""
    print("Calibrating pan direction (keep the tag in view)...")
    cx0 = read_tag_cx(cap, detector, tag_id, rotation)
    if cx0 is None:
        print("Could not see the tag to calibrate; falling back to --pan-invert as given.")
        return fallback_invert

    pan_motor.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=test_speed, blocking=False)
    time.sleep(test_duration)
    pan_motor.motor_stop()
    time.sleep(0.2)

    cx1 = read_tag_cx(cap, detector, tag_id, rotation)
    if cx1 is None:
        print("Lost the tag during calibration; falling back to --pan-invert as given.")
        return fallback_invert

    delta = cx1 - cx0
    print(f"Calibration: clockwise moved the tag by {delta:+.1f}px in frame.")
    if abs(delta) < 3:
        print("Movement too small to calibrate reliably; falling back to --pan-invert as given.")
        return fallback_invert

    # Positive error (tag right of center) should drive the pan speed in
    # whatever direction makes cx decrease. If clockwise increased cx, then
    # clockwise is the wrong response to positive error, so invert.
    return delta > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="Phone IP-camera stream URL")
    source_group.add_argument("--camera-index", type=int, help="Local camera device index (e.g. Continuity Camera)")
    parser.add_argument("--rotate", type=int, default=0, choices=sorted(ROTATIONS), help="Rotate the feed clockwise by this many degrees, to match a vertically-mounted camera (default 0)")
    parser.add_argument("--dict", default=DEFAULT_DICT, help="AprilTag dictionary name")
    parser.add_argument("--tag-id", type=int, default=None, help="Only track this tag ID (default: first tag found)")

    parser.add_argument("--pan-max-speed", type=int, default=40, help="Max pan motor speed percent (default 40)")
    parser.add_argument("--pan-min-speed", type=int, default=15, help="Minimum effective pan motor speed percent (default 15)")
    parser.add_argument("--pan-deadzone", type=int, default=15, help="Pixel deadzone around frame-center for the pan loop (default 15)")
    parser.add_argument("--pan-invert", action="store_true", help="Flip pan direction if it turns away from the tag instead of toward it")

    parser.add_argument("--car-max-speed", type=int, default=60, help="Max car motor speed percent (default 60)")
    parser.add_argument("--car-min-speed", type=int, default=20, help="Minimum effective car motor speed percent (default 20)")
    parser.add_argument("--car-deadzone-deg", type=float, default=3, help="Pan-angle deadzone (degrees) around zero for the drive loop (default 3)")
    parser.add_argument("--car-full-speed-deg", type=float, default=30, help="Pan angle (degrees) at which the car reaches max speed (default 30)")
    parser.add_argument("--car-invert", action="store_true", help="Flip car direction if it drives away from center instead of toward it")

    parser.add_argument("--command-hz", type=float, default=10, help="Max rate to send motor commands (default 10 Hz)")
    parser.add_argument("--pan-card-color", default=None, help="Single Motor connection card color name (e.g. AZURE)")
    parser.add_argument("--pan-card-serial", default="6065", help="Single Motor connection card serial number (default 6065)")
    parser.add_argument("--car-card-color", default=None, help="Double Motor connection card color name (e.g. AZURE)")
    parser.add_argument("--car-card-serial", default="1096", help="Double Motor connection card serial number (default 1096)")
    parser.add_argument("--dry-run", action="store_true", help="Vision only; do not connect to or drive either motor")
    args = parser.parse_args()

    source = args.url if args.url is not None else args.camera_index
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {source}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    rotation = ROTATIONS[args.rotate]

    pan_motor = None
    car_motor = None
    pan_invert = args.pan_invert
    if not args.dry_run:
        pan_motor = le.SingleMotor()
        connect_motor(pan_motor, "Single Motor (pan)", resolve_card_color(args.pan_card_color), args.pan_card_serial)
        pan_motor.motor_set_end_state(le.MOTOR_END_STATE_BRAKE)
        pan_motor.motor_reset_relative_position(position=0)
        print("Pan motor zeroed -- make sure the phone was aimed straight ahead just now.")
        pan_invert = calibrate_pan_sign(cap, detector, args.tag_id, rotation, pan_motor, args.pan_invert)
        print(f"Pan invert = {pan_invert}")

        car_motor = le.DoubleMotor()
        connect_motor(car_motor, "Double Motor (drive)", resolve_card_color(args.car_card_color), args.car_card_serial)
        car_motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)

    def drive(pan_speed, car_speed):
        if pan_motor is not None:
            if pan_speed == 0:
                pan_motor.motor_stop(blocking=False)
            else:
                direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if pan_speed > 0 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                pan_motor.motor_run(direction=direction, speed=abs(pan_speed), blocking=False)
        if car_motor is not None:
            car_motor.movement_move_tank(car_speed, car_speed, blocking=False)

    pan_controller = None
    drive_controller = ProportionalController(
        deadzone=args.car_deadzone_deg, max_out=args.car_max_speed,
        min_out=args.car_min_speed, gain=args.car_max_speed / args.car_full_speed_deg,
        invert=args.car_invert,
    )

    last_command_time = 0.0
    command_interval = 1.0 / args.command_hz

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost video stream.")
                break
            if rotation is not None:
                frame = cv2.rotate(frame, rotation)

            if pan_controller is None:
                center_x = frame.shape[1] / 2
                pan_controller = ProportionalController(
                    deadzone=args.pan_deadzone, max_out=args.pan_max_speed,
                    min_out=args.pan_min_speed, gain=args.pan_max_speed / center_x,
                    invert=pan_invert,
                )
            else:
                center_x = frame.shape[1] / 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tag = find_tag(gray, detector, args.tag_id)

            pan_speed = 0
            if tag is not None:
                cx, cy = tag.mean(axis=0)
                cv2.polylines(frame, [tag.astype(int)], True, (0, 0, 255), 3)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({int(cx)}, {int(cy)})", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                pan_speed = pan_controller.output_for(cx - center_x)

            pan_position = pan_motor.motor.position if pan_motor is not None else 0
            car_speed = drive_controller.output_for(pan_position) if tag is not None else 0

            cv2.line(frame, (int(center_x), 0), (int(center_x), frame.shape[0]), (255, 0, 0), 1)
            cv2.putText(frame, f"pan speed: {pan_speed}  pan angle: {pan_position:.1f} deg  car speed: {car_speed}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            now = time.time()
            if now - last_command_time >= command_interval:
                drive(pan_speed, car_speed)
                last_command_time = now

            cv2.imshow("AprilTag Parking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        drive(0, 0)
        if pan_motor is not None:
            pan_motor.motor_stop()
            pan_motor.disconnect()
        if car_motor is not None:
            car_motor.movement_stop()
            car_motor.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
