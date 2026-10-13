# T-anne-is: ME193 Midterm (Anne Jumper)

*A virtual tennis game.*

Rally against a virtual opponent using a **LEGO Education Double Motor as your racket**
(Connection Card **6065**). Choose clay, grass or hard court. Each surface bounces
differently, so the height where your racket has to be changes. Twist the racket for spin.
Your record rally is posted live to MQTT topic `ME193/Rogers/AnneJumper`.

## Setup (once)

```bash
cd ~/PingPong-Midterm
/opt/homebrew/bin/python3.12 -m venv .venv      # MediaPipe needs Python 3.12
.venv/bin/pip install -r requirements.txt
mkdir -p models && curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
.venv/bin/python generate_tags.py                # print tags/sheet.png, the bigger the better
```

(This folder already has the `.venv`, the model and the `tags/`.)

## Play

```bash
.venv/bin/python imu_monitor.py   # 1st time: check the swing (pitch) and twist (yaw) axes; tune thresholds
.venv/bin/python tennis.py        # the game: opens the 3D court in your browser
.venv/bin/python tennis.py --sim  # no hub: SPACE flat, T topspin, B slice, F full power
```

Options: `--view cv` (old OpenCV window instead of the 3D view), `--no-browser`, `--low-graphics` (if the 3D view stutters), `--hand left`, `--level 1|2|3`, `--surface clay|grass|hard`, `--swing-axis` / `--spin-axis x|y|z`, `--no-mqtt`.

| AprilTag | Meaning | Key |
|---|---|---|
| 0 | **START** (hold it up for about 0.2 s) | S |
| 1 / 2 / 3 | Level: Club 9 m/s, Varsity 12 m/s, Pro 16 m/s | 1 / 2 / 3 |
| 4 / 5 / 6 | Court: Clay, Grass, Hard | C / G / H |

**How to play:** stand far enough back that the camera sees you from head to hips. Hold the
hub like a racket handle. When the ball arrives, put your hand on the white **"hit here"** ring
and swing (rotate the hub about its **pitch** axis).
- **Twist** the hub on its **yaw** axis during the swing for spin: one way gives topspin, the other gives slice.
- No twist gives a clean **flat** shot.
- If topspin reads as slice, set `SPIN_SIGN = -1` in `racket.py`.
- Big flat swings at low balls fly **long**. Topspin brings them down.

**Haptic tip:** press a LEGO piece onto one motor shaft so it sits off-center. That makes it a
vibration motor (an eccentric rotating mass, like a phone's), and the hit pulses feel much stronger.

## Optional: train your own stroke classifier

```bash
.venv/bin/python train_strokes.py                          # all 6: forehand & backhand x flat / topspin / slice
.venv/bin/python train_strokes.py --labels backhand_topspin  # re-record or add to one class
```

It finishes with a table showing which strokes the model mixes up. When `strokes_model.json` exists,
the game uses the KNN to decide **forehand or backhand** and the **spin type**; the twist sets how much
spin. Delete it to go back to twist-only spin. Without a model, if your backhand topspin twist reads
as slice, set `BACKHAND_SPIN_SIGN = -1` in `racket.py`.

## Files

| File | What it does |
|---|---|
| `tennis.py` | Main game: hub connection, game state machine, the hit/miss policy, MQTT |
| `webview.py` | Local web server + websocket that streams the game state and webcam to the browser |
| `web/` | The Three.js view: `venues.js` (the three Grand Slam-style stadiums), `game.js` (ball, racket, effects, sound), HUD (`index.html`, `style.css`), and a local copy of Three.js |
| `racket.py` | IMU swing detection (pitch rate gives power, yaw twist gives signed spin) and haptics, beeps and light |
| `court.py` | Ball physics: gravity, Magnus effect from spin, surface bounce, aiming |
| `vision.py` | MediaPipe pose (racket hand), AprilTag reader, body-relative screen mapping |
| `strokes.py`, `train_strokes.py` | k-nearest-neighbors stroke classifier (6 classes) |
| `imu_monitor.py` | Live IMU readout for tuning |
| `generate_tags.py` | Printable AprilTags |
| `lelib.py`, `mqttlib.py`, `camlib.py` | Class libraries (unchanged) |
