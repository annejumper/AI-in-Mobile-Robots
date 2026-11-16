# Running the trained policies on the real Silly Walker

These scripts run the three **hardware-ready** policies (`Walk-Real-Default`, `-Laminate`, `-Carpet`) on the real
robot: a LEGO Education Double Motor with two beam legs. The laptop runs the policy and talks to the motor over
Bluetooth with the official [`legoeducation`](https://pypi.org/project/legoeducation/) API.

## Why "hardware-ready" policies

The first policies (`Drive-Straight-*`, in `../results/default`, `laminate`, `carpet`) were trained seeing their exact
body velocity and exact distance from the line. Only the simulator knows those. The hardware-ready policies were
trained, still entirely in simulation, seeing only what the Double Motor can measure, with the imperfections of the
real thing:

| The policy sees (9 numbers) | From | Trained with |
|---|---|---|
| turning rates x, y, z | IMU gyroscope | +-0.15 rad/s noise, up to 40 ms old |
| heading (sin, cos) | IMU yaw, relative to the start | +-0.03 noise, up to 40 ms old |
| left and right leg speed | motor encoders | +-0.5 rad/s noise, up to 40 ms old |
| its last command | itself | |

It acts every 40 ms (25 Hz). Each simulated robot also got a different floor friction (+-30%), motor mass (+-10%),
motor centre of mass (+-5 mm), motor strength (+-30%) and starting leg angle (+-0.3 rad), so the policy cannot rely on
any one exact value. The rewards were unchanged: they are only used during training, so they still use the true speed
and position.

## Files

| File | What it is |
|---|---|
| `silly_walker_real.py` | Everything: `verify`, `calibrate`, `testdrive`, `run` (and `--fake`) |
| `walk_default.py`, `walk_laminate.py`, `walk_carpet.py` | Shortcuts for `silly_walker_real.py run <floor>` |
| `policies/` | The three exported policies: `.onnx` (network), `.json` (what the numbers mean), `_check.npz` (reference answers) |
| `calibration.json` | Written by `calibrate` (not there until you run it) |
| `real_runs/` | One CSV per run (every control step), plus `summary.csv` with one line per run |
| `requirements.txt` | `legoeducation`, `onnxruntime`, `numpy` (and `mujoco` for `--fake`) |

## Setup (once)

1. Python 3.11+ and the packages: `python -m pip install -r requirements.txt`
2. Put the number on your Double Motor's Connection Card in `CARD_SERIAL` at the top of `silly_walker_real.py`.
3. Check the policies: `python silly_walker_real.py verify laminate` (and `default`, `carpet`). It should print
   `OK: the laptop runs the same policy.`

## Calibrate (once per robot, about 2 minutes)

```
python silly_walker_real.py calibrate
```

It walks you through six steps and saves `calibration.json`:

1. Robot still on the floor: measures the gyroscope's zero offset.
2. Turn it **exactly a quarter turn to the left** (counterclockwise from above): finds the gyro axis for turning, its
   units, and whether the IMU yaw works in this mounting (if not, the heading comes from the gyro instead).
3. Lift its **front** (the end the legs point to) and put it down: finds the pitch axis.
4. Lift its **left side** and put it down: finds the roll axis.
5. Hold it in the air: the legs run at 40% for 2 s to measure leg speed per % (and the motor's top speed).
6. On the floor at 40%: you say whether it walked toward its front. If not, the script flips its idea of front, and of
   left and right.
7. Set the legs by hand to the Onshape pose: saves that as the legs' home position. Every run moves the legs there first.

## Run

Mark a start line on the floor (tape), point the robot along a second piece of tape, then:

```
python walk_laminate.py          # or walk_default.py / walk_carpet.py; 10 s
python walk_laminate.py 20       # 20 s
```

It moves the legs home, waits for Enter, sits still for 1.5 s (measuring gyro offset and the "straight ahead"
direction), then walks. Each second it prints the heading, leg speeds and commands. At the end it asks you to measure
how far forward and sideways it went (cm), and appends everything to `real_runs/summary.csv`.

`python silly_walker_real.py testdrive` repeats `drive.py`'s straight phase (both legs at 8 rad/s for 3 s, no policy):
the simplest sim-vs-real check of the robot itself. The simulation goes about 23 cm.

## Comparing with the simulation

| Floor | Simulation, 10 s (`../results/hardware_ready/<floor>/`) | Script against the MuJoCo model (`--fake`) | Real robot |
|---|---|---|---|
| Default (friction 1.0) | 0.72 m, ends 7 cm left, heading -13 to +24 deg | 0.72 m, +2 cm, -20 to +19 deg | *your runs* |
| Laminate (0.3) | 0.56 m, ends 0 cm, heading -18 to +15 deg | 0.63 m, +2 cm, -20 to +13 deg | *your runs* |
| Carpet (0.6) | 0.50 m, ends 0 cm, heading -23 to +12 deg | 0.56 m, -8 cm, -21 to +10 deg | *your runs* |

Each start is slightly different (random starting angle in simulation, hand placement for real), so run each floor 3
or more times and compare averages. "Somewhat close" means: similar forward distance (same order, tens of cm in 10 s),
heading swinging but staying within a few tens of degrees, and drift of centimetres rather than tens of centimetres.

## Without the robot: `--fake`

`python walk_laminate.py --fake` runs exactly the same loop, but against the MuJoCo model on this laptop instead of the
Bluetooth motor. The fake motor even reports its readings in scrambled axes and LEGO-like units, which a built-in
calibration undoes. That tests the input conversion, the policy, the command conversion and the logging. It does not
test the real calibration or the Bluetooth timing: the first real runs check those.

## If something goes wrong

| What you see | What to do |
|---|---|
| `Set CARD_SERIAL...` | Put your Connection Card number at the top of `silly_walker_real.py`. |
| `Could not connect` | Motor on? Card number right? Bluetooth on? Close the LEGO app (it may hold the connection). |
| `Two motions used the same gyro axis` | Run `calibrate` again with bigger, cleaner motions (one axis at a time). |
| It walks backwards | Run `calibrate` again and answer step 6 carefully: "front" is the end the legs point to in Onshape. |
| It turns steadily one way | Check the yaw step: turn exactly a quarter turn **left**. A wrong sign makes it steer the wrong way. |
| Prints `NOTE: the policy may ask for up to 10 rad/s` | The real motor is slower than the policy's top speed; fast commands get capped at 100%. |
