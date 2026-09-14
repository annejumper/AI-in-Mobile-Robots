# PoseRace

Control a LEGO Education race car's two drive wheels independently by
raising and lowering your arms in front of a webcam.

## Gesture scheme

Each wrist's height relative to your own shoulder sets that side's wheel
speed, continuously from -100% (full reverse) to +100% (full forward):

- Arm at shoulder height -> that wheel is stopped (small dead zone so a
  resting arm doesn't drift).
- Raise an arm -> that wheel drives forward, faster the higher you raise it.
- Lower an arm below shoulder height -> that wheel reverses.
- Left wrist controls the left wheel, right wrist controls the right wheel.
- Raise one arm and lower the other to spin in place -- useful for backing
  out of a tight spot.

The mapping is normalized by shoulder width, so it adapts to how far
you're standing from the camera without recalibration (shoulder width was
chosen over shoulder-to-hip torso length because it stays in frame even
when a webcam only frames head-to-waist). Reaching full speed takes a
bigger raise going forward than going backward, since a lowered arm tends
to leave the camera's view sooner than a raised one.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py             # connects to the LEGO Double Motor over Bluetooth and drives it
python main.py --dry-run   # runs the webcam + gesture pipeline only, no hardware required
python main.py --card-serial 0049   # connect to a different hub than the hard-coded default
```

`main.py` hard-codes `CARD_SERIAL` to this car's **Connection Card serial
number** (printed on the physical card that ships with each LEGO CS/AI
hub) so it always pairs with the same hub. This matters in a classroom:
with many hubs powered on at once, connecting without a card filter can
grab someone else's motor instead of yours. If you ever swap to a
different hub, update `CARD_SERIAL` at the top of `main.py` (or pass
`--card-serial` to override it for one run).

## Obstacle motor (hand gesture triggered)

A second, independent LEGO Single Motor swings an obstacle on top of the
car (`poserace/obstacle.py`), controlled by hand gestures rather than the
arms/pose tracking that drives the wheels:

- Hold up **one finger** ("1", MediaPipe's `Pointing_Up`) -> swings the
  obstacle 90 degrees clockwise.
- Hold up **two fingers in a peace sign** ("2", MediaPipe's `Victory`) ->
  swings the obstacle 90 degrees counterclockwise.

Each gesture triggers exactly one move the moment it's newly shown;
holding the same gesture doesn't repeat the move. This uses a second
pretrained MediaPipe model (Gesture Recognizer), downloaded automatically
alongside the pose model on first run.

`SingleMotor` and `DoubleMotor` are separate physical hub products, each
with their own Connection Card, so the obstacle hub needs its own card
serial:

```bash
python main.py --obstacle-card-serial 0123   # if it's a different physical hub
python main.py --no-obstacle                 # skip it (and skip loading the gesture model)
```

`OBSTACLE_CARD_SERIAL` in `main.py` is hard-coded to this project's
obstacle hub. If the obstacle hub fails to connect, the program keeps
running with just the drive wheels rather than exiting.

Press `q` in the video window to quit. On first run, the MediaPipe pose
model (~6MB) is downloaded automatically into `poserace/models/`.

## How is Python talking to the LEGO hardware?

Over **Bluetooth Low Energy** via the `legoeducation` package, which wraps
`bleak` (cross-platform BLE) and speaks LEGO's binary RPC protocol to the
Double Motor hub. We use `DoubleMotor.movement_move_tank(speed_left,
speed_right)` to set both wheel speeds in a single BLE command per frame.

## Is it synchronous or asynchronous?

Both. Internally, `legoeducation` runs everything through a single
`asyncio` event loop (`background_worker.TransportManager`) that owns the
BLE connection -- all actual I/O is async. The public API exposed to us,
though, is synchronous by default: each call takes a `blocking` argument,
and a synchronous-looking call is bridged onto the async loop under the
hood. We call `movement_move_tank(..., blocking=False)` so sending a new
speed command every video frame never stalls the webcam loop waiting for a
BLE acknowledgment.

## How did you train it, and what are its limitations?

We didn't train a model -- this project uses two pretrained MediaPipe
models (Pose Landmarker for arm tracking, Gesture Recognizer for the
obstacle motor's hand gestures), both from Google, downloaded automatically
as `.task` files. Our own code (`poserace/gestures.py`) is a hand-written,
rule-based mapping (arm height -> speed, no learning) applied to the pose
model's landmark output; the obstacle motor just reacts to the gesture
model's fixed, canned gesture categories (`Pointing_Up`, `Victory`).
Limitations:

- Requires decent, even lighting and the user's shoulders/wrists visible
  in frame; occlusion or being partially off-screen breaks tracking.
- Tracks one person/one hand at a time; a crowded frame can confuse
  detection.
- The Gesture Recognizer only knows its fixed vocabulary of canned
  gestures -- there's no way to add a custom gesture without training a
  new classifier, which we didn't do.
- Raw landmark coordinates are jittery frame-to-frame, so we smooth wheel
  speed with an exponential moving average, which trades a little
  responsiveness for stability.
- The dead zone, full-speed range, and gesture confidence threshold are
  fixed constants tuned by hand-testing, not per-user calibrated -- they
  may feel slightly off for very different body proportions or camera
  angles.
- End-to-end latency (camera capture -> model inference -> BLE write ->
  motor response) adds a small but real control delay.
