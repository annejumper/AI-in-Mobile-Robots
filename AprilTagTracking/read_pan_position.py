"""Print the Single Motor's live absolute position (0-359 deg), so you can
physically aim the camera/tag mount by hand and read off the angle to use
for --pan-start-position-deg.

Usage:
    python read_pan_position.py
    python read_pan_position.py --card-serial 6065

Press Ctrl+C to quit.
"""

import argparse
import time

import legoeducation as le


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--card-color", default=None, help="Single Motor connection card color name (e.g. AZURE)")
    parser.add_argument("--card-serial", default="6065", help="Single Motor connection card serial number (default 6065)")
    args = parser.parse_args()

    card_color = None
    if args.card_color is not None:
        card_color = getattr(le, f"LEGO_COLOR_{args.card_color.upper()}")

    motor = le.SingleMotor()
    print("Connecting to Single Motor...")
    motor.connect(card_color=card_color, card_serial=args.card_serial)
    if not motor.connected:
        raise SystemExit("Error connecting to Single Motor. Check it's powered on and the card serial matches.")
    print("Connected. Rotate the mount by hand and watch the angle. Ctrl+C to quit.\n")

    try:
        while True:
            print(f"absolute position: {motor.motor.absolutePosition:6.1f} deg", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        motor.disconnect()


if __name__ == "__main__":
    main()
