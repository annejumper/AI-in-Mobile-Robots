# AprilTag Parking

A robot that uses an AprilTag and a camera to autonomously center (park) itself, driven by two LEGO Education motors.

## How it works

The robot can run in either of two physical setups:

- **Mode 0 (webcam):** a stationary computer webcam watches an AprilTag mounted on top of the robot.
- **Mode 1 (phone):** a camera mounted on the robot (e.g. a phone) watches a stationary AprilTag.

Two independent PID control loops drive the robot:

- **Pan loop (aligning):** a Single Motor rotates the tag's mount (or the camera) until it's perpendicular to the other side, measured by comparing the apparent length of the tag's left and right edges — foreshortening makes the farther edge look shorter whenever the tag isn't square-on.
- **Car loop (centering):** a Double Motor drives the robot straight forward/backward until the tag is centered in the camera's frame. The car is held still until the tag has been square-on for a moment, since a steep angle plus car motion at the same time is when the tag is most likely to drop out of detection.

If the tag is lost, the robot briefly coasts (in case it's just a momentary dropout), then stops the car for good and searches for the tag by spinning the pan motor through up to three full rotations, each slower than the last. If the tag still isn't found, the program exits. Once reacquired, it re-aligns and re-centers automatically.

See the code flow diagram for the full state machine.

## Files

| File | Purpose |
|---|---|
| `park_control.py` | Main control loop: vision, PID control, and motor driving. |
| `generate_tag.py` | Generates a printable AprilTag image. |
| `view_stream.py` | Sanity-checks a camera source (webcam or phone) before running the full control loop. |
| `test_pan_motor.py` | Isolates the pan motor's direction convention, no camera involved. |
| `tags/` | Generated AprilTag images. |
| `requirements.txt` | Python dependencies. |

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Print an AprilTag (or generate one):
   ```
   python generate_tag.py --id 0 --size 600 --out tags/tag_0.png
   ```
   Measure the printed tag's black square side length — you'll need it for distance estimation.
3. Mount the tag and camera per your chosen mode, and confirm the camera source works:
   ```
   python view_stream.py --list          # find the right camera index
   python view_stream.py --camera-index 1
   ```
4. (Optional) Check the pan motor's direction convention:
   ```
   python test_pan_motor.py
   ```

## Running

```
python park_control.py --dry-run   # vision only, no motors
python park_control.py             # full run, both motors
```

At startup you'll be prompted to select `0` (webcam mode) or `1` (phone mode), which picks a sensible default camera source. Pass `--url` or `--camera-index` to override it.

**Before running (without `--dry-run`):** physically aim the tag roughly at the camera — the car's direction calibration briefly nudges the Double Motor to figure out which way is which, and needs the tag in view to do so.

Run `python park_control.py --help` for the full list of tuning flags (PID gains, speed caps, deadzones, timeouts, etc).
