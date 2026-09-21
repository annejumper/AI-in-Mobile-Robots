"""Isolate the pan (Single Motor) direction convention, no camera involved.

Spins the pan motor clockwise for 2 seconds, stops, then counter-clockwise
for 2 seconds. Watch the turret and note which physical direction each one
produces (e.g. "clockwise turns it toward the AprilTag" or "away").

Usage:
    python test_pan_motor.py
    python test_pan_motor.py --card-serial 6065
"""

import argparse
import time

import legoeducation as le


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card-color", default=None, help="Single Motor connection card color name (e.g. AZURE)")
    parser.add_argument("--card-serial", default="6065", help="Single Motor connection card serial number (default 6065)")
    parser.add_argument("--speed", type=int, default=30, help="Test speed percent (default 30)")
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds to run each direction (default 2.0)")
    args = parser.parse_args()

    card_color = None
    if args.card_color is not None:
        attr = f"LEGO_COLOR_{args.card_color.upper()}"
        card_color = getattr(le, attr)

    motor = le.SingleMotor()
    print("Connecting to Single Motor...")
    motor.connect(card_color=card_color, card_serial=args.card_serial)
    if not motor.connected:
        raise SystemExit("Error connecting to Single Motor. Check it's powered on and the card serial matches.")
    print("Connected.")

    try:
        print(f"Spinning CLOCKWISE at {args.speed}% for {args.seconds}s -- watch which way it turns.")
        motor.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=args.speed, blocking=False)
        time.sleep(args.seconds)
        motor.motor_stop()
        time.sleep(1)

        print(f"Spinning COUNTERCLOCKWISE at {args.speed}% for {args.seconds}s -- watch which way it turns.")
        motor.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=args.speed, blocking=False)
        time.sleep(args.seconds)
        motor.motor_stop()
    finally:
        motor.motor_stop()
        motor.disconnect()


if __name__ == "__main__":
    main()
