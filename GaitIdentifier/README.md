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

Activity button: a pushbutton between **D2** and **GND** (the UNO Q has no user button of its own).

Axes as mounted: **X up the leg, Y forward (direction of travel), Z to the left.** Checked on 2026-10-26: each axis reads about +1 g pointing up. Z reads 0.17 g high, which `main.py` subtracts.

## Plan

1. **Wiring test** (done 2026-10-26). The IMU answers as a genuine MPU-6050 (WHO_AM_I 0x68) and streams at 100 Hz (±16 g, ±2000 °/s) with no dropped samples.
2. **Record labeled data** for each activity (1–2 min each, ideally several people). The app in [gait-identifier/](gait-identifier/) is the default app on the board, so it starts on power-up with no laptop. Each press of the D2 button steps PAUSE → STAND → WALK → RUN → BIKE, shown on the LED matrix as ‖, S, W, R, B (✗ if the sensor drops). Every sample is saved with its label to `data/session_<time>.csv` on the board, one file per power-on.
3. **Features over 2 s windows**: peak/mean/spread per axis, step frequency, gyro energy, gravity direction (catches lying-down air-biking).
4. **Model 1**: thresholds on peak X acceleration only. **Model 2**: small decision tree on all features, compared against KNN. No neural nets.
5. **Live display**: activity icon on the LED matrix (animated at step rate, error icon if the sensor drops), plus a web page served by the UNO Q showing both models' answers side by side and live graphs.

## Running it

App Lab app in [gait-identifier/](gait-identifier/). Deploy with `tools/deploy.py` from the `AI_in_mobile_robotics` repo, over USB (`brew install android-platform-tools` for adb) or Wi-Fi (board at 10.247.137.125 on Tufts_Robot).
