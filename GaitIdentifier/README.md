# GaitIdentifier

A wearable built from an Arduino UNO Q and an MPU-6050 IMU that classifies what the wearer is doing:

- standing still
- walking
- running
- biking (or lying on your back and air-biking)

Two models are compared (no neural nets):

1. **Single-feature model**: uses only the peak acceleration along one axis.
2. **Full-IMU model**: uses all of the IMU outputs (3-axis accelerometer + 3-axis gyroscope).

## Hardware

Everything is worn at the ankle in one 3D-printed housing (attached with an elbow brace): UNO Q, battery and a GY-521 (MPU-6050) board, mounted rigidly.

| GY-521 | UNO Q |
|---|---|
| VCC | 5V (power header) |
| GND | GND (digital header, below AREF) |
| SDA | SDA / D20 |
| SCL | SCL / D21 |

Optional activity button: a pushbutton between **D2** and **GND** (the UNO Q has no user button of its own). Not needed: the phone page below does the same job.

Axes as mounted: **X up the leg, Y forward (direction of travel), Z to the left.** Checked on 2026-10-26: each axis reads about +1 g pointing up. Z reads 0.17 g high, which `main.py` subtracts.

## Plan

1. **Wiring test** (done 2026-10-26). The IMU answers as a genuine MPU-6050 (WHO_AM_I 0x68) and streams at 100 Hz (±16 g, ±2000 °/s) with no dropped samples.
2. **Record labeled data** for each activity (1–2 min each, ideally several people). The app in [gait-identifier/](gait-identifier/) is the default app on the board, so it starts on power-up with no laptop. Pick the activity on a phone at **http://10.247.137.125:7000** (phone on Tufts Wi-Fi), which also shows how much of each has been recorded. The LED matrix shows it as ‖, S, W, R, B (✗ if the sensor drops). Every sample is saved with its label to `data/session_<time>.csv` on the board, one file per power-on.
3. **Features over 2 s windows** ([features.py](gait-identifier/python/features.py), shared by training and the board): mean, spread, max and min per axis, acceleration and rotation magnitude, and stride/pedal rate.
4. **Models** ([training/train.py](training/train.py)). **Model 1**: a depth-2 tree on peak X acceleration only, which works out to three thresholds. **Model 2**: a depth-4 decision tree on all features, compared against KNN (k=5). No neural nets. Each recording is split by time (first 70% train, last 30% test). Results from the 2026-10-26 data (326 test windows):

   | Model | Test accuracy |
   |---|---|
   | Model 1: peak X accel only (cutoffs 1.26 / 2.32 / 4.12 g) | 96.3% |
   | Model 2: decision tree, full IMU | 98.2% |
   | KNN (k=5), full IMU | 98.2% |

   Plots are in [training/results/](training/results/). The bike data is real (upright) biking. Record lying-down air-biking too before demoing that.
5. **Live display** (done). The board classifies the last 2 s every 0.5 s with both models. On the phone page, the **Live** tab shows the answer large, a switch picks which model drives it and the LED matrix, and both models' answers are shown side by side. The **Record** tab labels new training data.

### Retraining

```
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn matplotlib   # once
.venv/bin/python training/train.py
```

It reads `data/session_*.csv` (copied off the board from `ArduinoApps/gait-identifier/data/`), prints accuracy and the learned rules, saves plots, and writes `gait-identifier/python/model.json`. Deploy the app again to put the new model on the board.

## Running it

App Lab app in [gait-identifier/](gait-identifier/). Deploy with `tools/deploy.py` from the `AI_in_mobile_robotics` repo, over USB (`brew install android-platform-tools` for adb) or Wi-Fi (board at 10.247.137.125 on Tufts_Robot).
