"""Center the robot in the middle of a stationary computer camera's frame,
using an AprilTag mounted on top of the robot, driven by two LEGO
Education motors.

Physical setup (current -- Windows computer camera):
  - The camera (this computer's webcam) is STATIONARY, sitting on the
    desk/tripod, NOT on the robot.
  - The AprilTag is mounted on the robot, on top of the Single Motor, so
    the Single Motor can rotate the tag's mount independently of the car.
  - A Double Motor drives the car straight forward/backward (both wheels
    always at the same speed and direction -- it never pivots/turns;
    any rotation is the Single Motor's job, see below).

Two independent PID control loops:
  - Car loop (centering): the tag's horizontal pixel error in the
    (stationary) camera's frame drives the Double Motor straight
    forward/backward (both wheels together), until the tag -- and
    therefore the robot -- is centered in frame.
  - Pan loop (squaring up): compares the apparent length of the tag's
    left edge vs. its right edge in the image. Foreshortening makes the
    farther edge look shorter whenever the tag isn't square-on to the
    camera, so driving that left/right edge-length difference to zero
    with the Single Motor keeps the tag (and therefore the car, even if
    it isn't perfectly parallel to the camera) perpendicular to the
    camera. This gives the car some leeway -- it doesn't have to line up
    perfectly on its own. The car is also held still (never driven) until
    the tag has been square-on for a moment, since a steep angle plus car
    motion at the same time is when the tag is most likely to drop out of
    detection.

If the tag is lost (--tag-lost-timeout seconds with no detection), the
car stops for good and the pan motor searches: it spins a full 360
degrees, in whichever direction it was last turning to correct the
angle, at --search-speed. If found mid-spin, it stops and resumes normal
squaring/centering. If not, it tries up to 3 spins total, each slower
than the last; if the tag still isn't found, the program stops and exits.

IMPORTANT: before running (without --dry-run), physically aim the tag
roughly at the camera -- the car's direction calibration briefly nudges
the Double Motor to figure out which way is which, and needs the tag in
view to do that. The pan motor starts wherever it happens to be (no
homing) and is not auto-calibrated either: it just holds still until the
camera loop starts commanding it; use --pan-invert manually if it turns
the wrong way.

---
ORIGINAL setup (teammate's Mac, phone mounted on the robot, stationary
tag near the computer) is preserved below, commented out rather than
deleted. In that version the camera moved (via the phone on the pan
motor) to track a stationary tag, and the car drove to bring the pan
motor's accumulated angle back to zero. That is the opposite of the
current setup, where the camera is fixed and the tag is on the robot.

At startup, you'll be prompted to type 0 (webcam state) or 1 (phone
state), which picks a sensible default camera source for that setup:
  * webcam state (0): this computer's local webcam, opened via
    DirectShow on Windows for fast/reliable startup (device index 0).
  * phone state (1): a phone acting as the camera, either an IP-camera
    app over Wi-Fi (--url) or a local device index (e.g. macOS
    Continuity Camera, index 1).
Pass --url or --camera-index explicitly to override the prompt's default
for either state.

Usage:
    python park_control.py --dry-run   # vision only, no motors
    python park_control.py             # full run, both motors
    python park_control.py --camera-index 2   # override the state prompt's default camera index
"""

import argparse
import math
import sys
import time

import cv2
import legoeducation as le

DEFAULT_DICT = "DICT_APRILTAG_36h11"
SEARCH_SPIN_COUNT = 3  # number of 360-degree search spins to try before giving up

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class PIDController:
    """output = clip(Kp*error + Ki*integral(error) + Kd*derivative(error),
    -max_out, max_out), zero inside a deadzone around error=0.

    The integral resets whenever the error is inside the deadzone (so it
    doesn't wind up once the target is reached). It uses clamping
    ("conditional integration") anti-windup: while a large error already
    saturates the output at max_out, further integral accumulation is
    paused, since it wouldn't change the output yet anyway -- it would
    only build up a backlog that has to unwind later, overshooting the
    target and needing a reverse correction to come back. With this,
    the integral only meaningfully builds once the actuator has room to
    use it, i.e., once the error is already fairly small -- which is
    exactly the situation Ki is for (closing a small residual error a
    plain P term settles for), without disturbing the smooth, monotonic
    slow-down of the approach itself.
    """

    def __init__(self, kp, ki, kd, deadzone, max_out, invert=False):
        sign = -1 if invert else 1
        self.kp = sign * kp
        self.ki = sign * ki
        self.kd = sign * kd
        self.deadzone = deadzone
        self.max_out = max_out
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def output_for(self, error, now=None):
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._prev_time is None else max(now - self._prev_time, 1e-3)
        self._prev_time = now

        if abs(error) < self.deadzone:
            self._integral = 0.0
            self._prev_error = error
            return 0

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        # Would this step's output already be pushed past the limit, in
        # the same direction the error is pushing it? If so, don't add to
        # the integral -- accumulating further can't help (output is
        # already clipped) and would only have to unwind later.
        provisional = self.kp * error + self.ki * self._integral + self.kd * derivative
        saturating_further = (provisional >= self.max_out and error > 0) or (provisional <= -self.max_out and error < 0)
        if not saturating_further:
            self._integral += error * dt
            if self.ki:
                max_integral = self.max_out / abs(self.ki)
                self._integral = max(-max_integral, min(max_integral, self._integral))

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return int(max(-self.max_out, min(self.max_out, out)))


class LowPassFilter:
    """Exponential moving average: smooths a noisy per-frame measurement
    before it reaches a PID controller. alpha (0-1] is the weight given
    to each new sample -- closer to 1 means less smoothing/more
    responsive, closer to 0 means more smoothing/slower to react.

    This matters more now that commands go out near the camera's own
    frame rate: without it, every bit of frame-to-frame corner-detection
    noise gets forwarded straight to the motors, which is what turns
    into jitter/stutter at low speed instead of a smooth crawl to zero.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self._value = None

    def reset(self):
        self._value = None

    def update(self, raw):
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1 - self.alpha) * self._value
        return self._value


def find_tag(gray, detector, tag_id):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for pts, found_id in zip(corners, ids.flatten()):
        if tag_id is not None and found_id != tag_id:
            continue
        return pts.reshape(4, 2)
    return None


def tag_skew(tag):
    """Perspective skew of the tag's quadrilateral, roughly -1 to 1. Zero
    means the tag is square-on (perpendicular) to the camera; positive
    means the tag's left edge appears longer than its right (the right
    side is angled away from the camera); negative is the opposite.

    Corner order from cv2.aruco.detectMarkers is top-left, top-right,
    bottom-right, bottom-left, so the tag's left/right sides are the
    (top-left, bottom-left) and (top-right, bottom-right) edges.
    """
    tl, tr, br, bl = tag
    left_edge = math.hypot(tl[0] - bl[0], tl[1] - bl[1])
    right_edge = math.hypot(tr[0] - br[0], tr[1] - br[1])
    total = left_edge + right_edge
    if total < 1e-6:
        return 0.0
    return (left_edge - right_edge) / total


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


def read_tag_metrics(cap, detector, tag_id, rotation, attempts=30):
    """Grab frames until the tag is found, and return its (centroid_x,
    skew). None if not found."""
    for _ in range(attempts):
        ok, frame = cap.read()
        if not ok:
            continue
        if rotation is not None:
            frame = cv2.rotate(frame, rotation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tag = find_tag(gray, detector, tag_id)
        if tag is not None:
            cx = tag.mean(axis=0)[0]
            return cx, tag_skew(tag)
    return None


def calibrate_car_direction(cap, detector, tag_id, rotation, car_motor, fallback_invert, test_speed=25, test_duration=0.4):
    """Drive the car forward, then backward, measuring the tag's pixel-x
    shift from each -- run every time the tag is freshly (re)acquired
    (not just once at cold startup), so it works regardless of which way
    the robot happens to be facing when it's placed down or re-finds the
    tag after a search. Forward and backward give two independent
    estimates of the same signed gain (driving backward should undo
    driving forward), averaged for a more robust measurement than a
    single nudge -- and the net movement is approximately zero, so this
    doesn't drift the robot away from where it was found."""

    def measure():
        m = read_tag_metrics(cap, detector, tag_id, rotation, attempts=5)
        return None if m is None else m[0]

    print("Calibrating car direction (keep the tag in view)...")
    m0 = measure()
    if m0 is None:
        print("Could not see the tag to calibrate; falling back to --car-invert as given.")
        return fallback_invert

    car_motor.movement_move_tank(test_speed, test_speed, blocking=False)
    time.sleep(test_duration)
    car_motor.movement_stop()
    time.sleep(0.2)
    m1 = measure()

    car_motor.movement_move_tank(-test_speed, -test_speed, blocking=False)
    time.sleep(test_duration)
    car_motor.movement_stop()
    time.sleep(0.2)
    m2 = measure()

    if m1 is None or m2 is None:
        print("Lost the tag during calibration; falling back to --car-invert as given.")
        return fallback_invert

    forward_delta = m1 - m0
    backward_delta = m2 - m1
    if abs(forward_delta) < 3 and abs(backward_delta) < 3:
        print("Movement too small to calibrate reliably; falling back to --car-invert as given.")
        return fallback_invert

    gain = ((forward_delta / (test_speed * test_duration)) + (-backward_delta / (test_speed * test_duration))) / 2
    print(f"Calibration: forward {forward_delta:+.1f}px, backward {backward_delta:+.1f}px -> gain {gain:+.4f} px/(speed*sec)")

    # Positive pixel error (tag right of center) should drive the car in
    # whatever direction makes cx decrease. If driving forward increased
    # cx, forward is the wrong response to positive error, so invert.
    return gain > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Camera source: either a phone IP-camera stream (--url, e.g. Android's
    # "IP Webcam" app or macOS Continuity Camera reached via device index),
    # or a local webcam device index. Mutually exclusive. Left unset
    # (None), the interactive state prompt at startup picks a sensible
    # default (0 for webcam state, 1 for phone state) -- pass one of these
    # explicitly only to override that default.
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--url", help="Phone IP-camera stream URL")
    source_group.add_argument("--camera-index", type=int, default=None, help="Local camera device index, overriding the state prompt's default. Use view_stream.py --list to find the right index if you have more than one camera.")

    parser.add_argument("--rotate", type=int, default=0, choices=sorted(ROTATIONS), help="Rotate the feed clockwise by this many degrees, to match a vertically-mounted camera (default 0)")
    parser.add_argument("--dict", default=DEFAULT_DICT, help="AprilTag dictionary name")
    parser.add_argument("--tag-id", type=int, default=None, help="Only track this tag ID (default: first tag found)")

    parser.add_argument("--pan-kp", type=float, default=60.0, help="Pan PID proportional gain: pan speed percent per unit of tag skew (default 60.0 -- a fixed constant, independent of camera/frame size)")
    parser.add_argument("--pan-ki", type=float, default=0.1, help="Pan PID integral gain -- accumulates a small sustained skew that the proportional term alone settles for instead of driving to 0 (default 0.1)")
    parser.add_argument("--pan-kd", type=float, default=10.0, help="Pan PID derivative gain -- damps oscillation/overshoot around square (default 10.0)")
    parser.add_argument("--pan-max-speed", type=int, default=40, help="Hard cap on pan motor (tag-angle) speed percent, regardless of what the PID computes (default 40)")
    parser.add_argument("--pan-skew-deadzone", type=float, default=0.01, help="Tag left/right edge-length skew deadzone around zero (perpendicular) for the pan loop (default 0.01, tightened for more sensitivity -- raise it if this causes jittering/hunting from corner-detection noise)")
    parser.add_argument("--car-hold-skew", type=float, default=0.03, help="Don't drive the car until the tag's skew is within this much of square -- pan corrects the angle first, every time, before the car is allowed to move, since a steep angle plus car motion at the same time is when the tag is most likely to drop out of detection (default 0.03)")
    parser.add_argument("--car-hold-settle-time", type=float, default=0.3, help="How long skew must stay continuously within --car-hold-skew before the car is released to drive (default 0.3s) -- without this, a single noisy frame that happens to dip below the threshold (e.g. the very first frame the tag is (re)detected) would release the car before pan has actually had a chance to correct anything")
    parser.add_argument("--pan-smoothing", type=float, default=0.3, help="Low-pass filter alpha (0-1] applied to the raw skew reading before the pan PID sees it -- lower means more smoothing/less jitter but slower to react (default 0.3)")
    parser.add_argument("--pan-invert", action="store_true", help="Flip pan direction if it turns the tag away from square instead of toward it")

    parser.add_argument("--car-kp", type=float, default=0.05, help="Car PID proportional gain: car speed percent per pixel of centering error (default 0.05 -- a fixed constant, independent of camera/frame size)")
    parser.add_argument("--car-ki", type=float, default=0.1, help="Car PID integral gain -- closes the small residual centering error a P-only response settles for. Uses clamping anti-windup (won't build up during the initial fast approach); lower toward 0 if you see a speed-up hump near the end (default 0.1)")
    parser.add_argument("--car-kd", type=float, default=0.02, help="Car PID derivative gain -- damps oscillation/overshoot around center (default 0.02)")
    parser.add_argument("--car-max-speed", type=int, default=25, help="Hard cap on car drive speed percent, regardless of what the PID computes (default 25)")
    parser.add_argument("--car-deadzone", type=int, default=5, help="Pixel deadzone around frame-center for the car centering loop (default 5)")
    parser.add_argument("--car-smoothing", type=float, default=0.3, help="Low-pass filter alpha (0-1] applied to the raw pixel-position reading before the car PID sees it -- lower means more smoothing/less jitter but slower to react (default 0.3)")
    parser.add_argument("--car-invert", action="store_true", help="Flip car drive direction if it drives away from center instead of toward it")

    parser.add_argument("--command-hz", type=float, default=30, help="Max rate to send motor commands (default 30 Hz). Raised from an earlier 10 Hz default -- at 10 Hz, commands lagged well behind the camera's own ~30fps, so each motor command was a stale, chunky update instead of a steady stream of small corrections, which shows up as jitter/stutter at low speed. Lower this only if you see BLE command backlog or disconnects from sending too fast.")
    parser.add_argument("--tag-lost-timeout", type=float, default=0.25, help="Seconds to keep driving at the last commanded speed after losing sight of the tag (a momentary dropout shouldn't cause a stop/restart jitter cycle) before stopping the car and starting a search spin (default 0.25)")
    parser.add_argument("--search-speed", type=float, default=6.0, help="Pan motor speed for the first 360-degree search spin once the tag has been lost for --tag-lost-timeout seconds. Spins in whatever direction pan was last correcting toward. Each subsequent spin (up to %d total) is slower: speed = search-speed / spin-number. The car never moves during a search; if the tag isn't found after all spins, the program stops and exits (default 6.0)" % SEARCH_SPIN_COUNT)
    parser.add_argument("--pan-card-color", default=None, help="Single Motor connection card color name (e.g. AZURE)")
    parser.add_argument("--pan-card-serial", default="6065", help="Single Motor connection card serial number (default 6065)")
    parser.add_argument("--car-card-color", default=None, help="Double Motor connection card color name (e.g. AZURE)")
    parser.add_argument("--car-card-serial", default="1096", help="Double Motor connection card serial number (default 1096)")
    parser.add_argument("--dry-run", action="store_true", help="Vision only; do not connect to or drive either motor")
    args = parser.parse_args()

    print("Select camera:")
    print("  0 = computer webcam (AprilTag mounted on robot)")
    print("  1 = phone camera (mounted on robot, tracking a stationary AprilTag)")
    choice = input("Enter 0 or 1: ").strip()
    while choice not in ("0", "1"):
        choice = input("Please enter 0 or 1: ").strip()
    state = "webcam" if choice == "0" else "phone"

    camera_index = args.camera_index if args.camera_index is not None else (1 if state == "phone" else 0)
    source = args.url if args.url is not None else camera_index
    if isinstance(source, str):
        cap = cv2.VideoCapture(source)
    elif sys.platform.startswith("win"):
        # cv2.CAP_DSHOW opens the local webcam via DirectShow, which starts
        # up faster and more reliably on Windows than OpenCV's default
        # backend. Only applies to a local device index, and only on
        # Windows -- DirectShow isn't available elsewhere.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source {source}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    rotation = ROTATIONS[args.rotate]

    pan_motor = None
    car_motor = None
    pan_invert = args.pan_invert
    car_invert = args.car_invert
    if not args.dry_run:
        pan_motor = le.SingleMotor()
        connect_motor(pan_motor, "Single Motor (tag angle)", resolve_card_color(args.pan_card_color), args.pan_card_serial)
        pan_motor.motor_set_end_state(le.MOTOR_END_STATE_BRAKE)
        # No homing and no auto-calibration nudge here -- pan starts
        # wherever it happens to be and just holds still until the
        # camera-driven PID loop starts commanding it (searching/squaring
        # both work from any starting angle). Set --pan-invert manually if
        # it turns the wrong way.

        car_motor = le.DoubleMotor()
        connect_motor(car_motor, "Double Motor (drive)", resolve_card_color(args.car_card_color), args.car_card_serial)
        car_motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
        # Car direction is NOT calibrated here -- it's calibrated fresh
        # every time the tag is (re)acquired, in the main loop below, so
        # it works regardless of which way the robot is facing when
        # placed down, rather than only whichever way it happened to
        # face at cold startup.

    def drive(pan_speed, car_speed):
        if pan_motor is not None:
            if pan_speed == 0:
                pan_motor.motor_stop(blocking=False)
            else:
                direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if pan_speed > 0 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                pan_motor.motor_run(direction=direction, speed=abs(pan_speed), blocking=False)
        if car_motor is not None:
            # Both wheels always the same speed/direction -- straight
            # forward/backward, never a pivot/turn. (motor_set_duty_cycle
            # was tried here to avoid speed-regulation jitter, but it sends
            # the same raw value to both physical motors with no awareness
            # that they're mounted mirrored, which turned the car instead
            # of driving it straight -- reverted to movement_move_tank,
            # which handles that mirroring correctly.)
            car_motor.movement_move_tank(car_speed, car_speed, blocking=False)

    # Kp/Ki/Kd are fixed constants you tune directly -- they no longer
    # depend on camera resolution or on each other (max_speed is now only
    # an output cap, not a term used to compute Kp).
    pan_controller = PIDController(
        kp=args.pan_kp, ki=args.pan_ki, kd=args.pan_kd,
        deadzone=args.pan_skew_deadzone, max_out=args.pan_max_speed,
        invert=pan_invert,
    )
    car_controller = PIDController(
        kp=args.car_kp, ki=args.car_ki, kd=args.car_kd,
        deadzone=args.car_deadzone, max_out=args.car_max_speed,
        invert=car_invert,
    )
    car_filter = LowPassFilter(args.car_smoothing)
    pan_filter = LowPassFilter(args.pan_smoothing)

    last_command_time = 0.0
    command_interval = 1.0 / args.command_hz

    # These persist across frames (not reset to 0 every loop) so that a
    # brief tag dropout can keep coasting at the last commanded values
    # instead of immediately slamming to a stop.
    car_speed = 0
    pan_speed = 0
    skew = 0.0
    last_seen_time = None
    squared_since = None  # when skew most recently entered the car-hold band
    car_direction_calibrated_this_cycle = False  # reset every time the tag is freshly (re)acquired

    # Search-spin state, entered once the tag's been lost longer than
    # --tag-lost-timeout. "normal" is everything else (tracking/coasting).
    mode = "normal"
    search_spin_index = 0
    search_direction = 1  # +1 = clockwise, -1 = counterclockwise

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost video stream.")
                break
            if rotation is not None:
                frame = cv2.rotate(frame, rotation)

            center_x = frame.shape[1] / 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tag = find_tag(gray, detector, args.tag_id)

            status = "tracking"

            if mode == "searching":
                # Car never moves during a search -- only the pan motor
                # spins, hunting for the tag.
                car_speed = 0
                pan_speed = 0
                if tag is not None:
                    # Found it -- stop the spin and hand back to normal
                    # control. squared_since is cleared, so the existing
                    # pan-first/car-hold logic above will square the tag up
                    # again before the car is allowed to move.
                    pan_motor.motor_stop(blocking=False)
                    mode = "normal"
                    last_seen_time = time.monotonic()
                    car_controller.reset()
                    pan_controller.reset()
                    car_filter.reset()
                    pan_filter.reset()
                    squared_since = None
                    car_direction_calibrated_this_cycle = False
                    status = "tag reacquired -- squaring up"
                else:
                    spin_speed = args.search_speed / (search_spin_index + 1)
                    if abs(pan_motor.motor.position) >= 360:
                        # This spin completed a full rotation without
                        # finding the tag -- start the next, slower one.
                        search_spin_index += 1
                        if search_spin_index >= SEARCH_SPIN_COUNT:
                            pan_motor.motor_stop(blocking=False)
                            print(f"Tag not found after {SEARCH_SPIN_COUNT} search spins -- stopping.")
                            break
                        spin_speed = args.search_speed / (search_spin_index + 1)
                        pan_motor.motor_reset_relative_position(position=0)
                        direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if search_direction > 0 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                        pan_motor.motor_run(direction=direction, speed=spin_speed, blocking=False)
                    status = f"tag lost -- searching (spin {search_spin_index + 1}/{SEARCH_SPIN_COUNT}, speed {spin_speed:.0f})"

            elif tag is not None:
                last_seen_time = time.monotonic()

                # Car direction is calibrated fresh every time the tag is
                # (re)acquired -- once per acquisition, not every frame --
                # so it works regardless of which way the robot happens to
                # be facing right now, rather than only whichever way it
                # faced at cold startup.
                if not car_direction_calibrated_this_cycle:
                    car_direction_calibrated_this_cycle = True
                    if car_motor is not None:
                        car_invert = calibrate_car_direction(cap, detector, args.tag_id, rotation, car_motor, args.car_invert)
                        print(f"Car invert = {car_invert}")
                        car_controller = PIDController(
                            kp=args.car_kp, ki=args.car_ki, kd=args.car_kd,
                            deadzone=args.car_deadzone, max_out=args.car_max_speed,
                            invert=car_invert,
                        )

                cx, cy = tag.mean(axis=0)
                skew = tag_skew(tag)
                cv2.polylines(frame, [tag.astype(int)], True, (0, 0, 255), 3)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({int(cx)}, {int(cy)})", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Smooth the raw per-frame readings before they hit the PID --
                # otherwise sending commands at near camera frame rate just
                # forwards every bit of corner-detection noise to the motors.
                smoothed_cx = car_filter.update(cx)
                smoothed_skew = pan_filter.update(skew)
                pan_speed = pan_controller.output_for(smoothed_skew)

                # Pan (squaring) goes first: hold the car still until the
                # tag has been close enough to square-on continuously for
                # --car-hold-settle-time -- not just for one frame, since a
                # single noisy reading dipping under the threshold (e.g.
                # the very first frame the tag is (re)detected, before pan
                # has had any chance to act) would otherwise release the
                # car immediately. A steep viewing angle combined with car
                # motion is exactly when the tag is most likely to drop out
                # of detection.
                now_mono = time.monotonic()
                if abs(smoothed_skew) < args.car_hold_skew:
                    if squared_since is None:
                        squared_since = now_mono
                else:
                    squared_since = None

                settled = squared_since is not None and (now_mono - squared_since) >= args.car_hold_settle_time
                if settled:
                    car_speed = car_controller.output_for(smoothed_cx - center_x)
                else:
                    car_speed = 0
                    car_controller.reset()
                    status = "squaring up (car held)"
            else:
                lost_for = float("inf") if last_seen_time is None else time.monotonic() - last_seen_time
                if lost_for < args.tag_lost_timeout:
                    # Briefly lost the tag (motion blur, momentary occlusion) --
                    # keep coasting at the last commanded car_speed/pan_speed
                    # instead of stopping and restarting, which is what
                    # caused a jitter cycle on every brief dropout.
                    status = f"tag lost, coasting ({args.tag_lost_timeout - lost_for:.1f}s left)"
                else:
                    # Lost long enough -- stop the car for good (it never
                    # moves again until the tag is reacquired) and start a
                    # search spin, continuing whichever way pan was last
                    # turning to correct the angle.
                    search_direction = 1 if pan_speed >= 0 else -1
                    car_speed = 0
                    pan_speed = 0
                    car_controller.reset()
                    pan_controller.reset()
                    car_filter.reset()
                    pan_filter.reset()
                    squared_since = None
                    if pan_motor is None:
                        status = "tag lost, stopped (search needs real hardware, not --dry-run)"
                    else:
                        mode = "searching"
                        # drive() is never called while mode == "searching"
                        # (search sends its own direct pan_motor commands
                        # below), so the car needs an explicit one-time stop
                        # here -- otherwise it just keeps coasting at
                        # whatever speed/direction it was last sent.
                        car_motor.movement_stop()
                        search_spin_index = 0
                        spin_speed = args.search_speed / (search_spin_index + 1)
                        pan_motor.motor_reset_relative_position(position=0)
                        direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if search_direction > 0 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                        pan_motor.motor_run(direction=direction, speed=spin_speed, blocking=False)
                        status = f"tag lost -- searching (spin 1/{SEARCH_SPIN_COUNT}, speed {spin_speed:.0f})"

            cv2.line(frame, (int(center_x), 0), (int(center_x), frame.shape[0]), (255, 0, 0), 1)
            cv2.putText(frame, f"car: {car_speed}  pan: {pan_speed}  skew: {skew:+.2f}  [{status}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            now = time.time()
            if mode != "searching" and now - last_command_time >= command_interval:
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
