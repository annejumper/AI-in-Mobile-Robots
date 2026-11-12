# Linux side of the safe hardware test.
# - prints what the sketch reports (button presses, latch state, servo angle)
# - runs commands written to cmd.txt in the app folder by tools/latch.py on the laptop:
#     lock | unlock | angle <0-180>
# - lists any cameras the board can see (for when the webcam moves to the UNO Q)
import os
import time
from pathlib import Path

from arduino.app_utils import App, Bridge

CMD_FILE = Path(__file__).resolve().parent.parent / "cmd.txt"

cams = sorted(d for d in os.listdir("/dev") if d.startswith("video"))
print(f"Cameras visible to the app: {cams or 'none'}")
print(f"Watching {CMD_FILE} for commands")

last = None


def run_command():
    if not CMD_FILE.exists():
        return
    words = CMD_FILE.read_text().split()
    CMD_FILE.unlink()
    if not words:
        return
    try:
        if words[0] == "lock":
            Bridge.call("set_locked", 1)
        elif words[0] == "unlock":
            Bridge.call("set_locked", 0)
        elif words[0] == "angle" and len(words) == 2:
            Bridge.call("set_angle", int(words[1]))
        else:
            print(f"Unknown command: {' '.join(words)}")
            return
        print(f"command: {' '.join(words)}")
    except Exception as exc:
        print(f"Command {' '.join(words)} failed: {exc}")


def loop():
    global last
    run_command()
    try:
        status = Bridge.call("get_status")
    except Exception as exc:
        print(f"Bridge call failed: {exc}")
        time.sleep(1)
        return
    if status != last:
        presses, locked, down, angle = status.split(",")
        state = "LOCKED" if locked == "1" else "UNLOCKED"
        button = "held" if down == "1" else "up"
        print(f"button {button:4}  presses {presses:>3}  latch {state:8}  servo {angle} deg")
        last = status
    time.sleep(0.05)


App.run(user_loop=loop)
