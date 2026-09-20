# lego_robot_gestures

Yash's gesture-recognition prototypes, developed independently in [YAKU-2003/LEGO-ROBOT](https://github.com/YAKU-2003/LEGO-ROBOT). Kept in this separate folder so it doesn't collide with the existing `poserace/` package.

Two approaches, both working end-to-end on live video, neither wired to a motor yet:

- **Continuous wrist-height control** -- `pose_gesture_controller.py` (full body) / `hand_gesture_controller.py` (hands only) map an arm's height to a smoothed -100%..+100% wheel speed. `run_pose_gesture.py` / `run_hand_gesture.py` run each live via webcam. `pose_camera.py` bundles pose + a canned-gesture (`Closed_Fist`/`Open_Palm`) emergency-stop recognizer.
- **Discrete hand-shape commands** -- `hand_command_controller.py` classifies fist/open-palm/finger-count each frame into one command (forward/backward/turn/spin an auxiliary motor). `run_hand_commands.py` runs it live, hands only (no face/body tracking).

`test_single_motor.py` is the one script here that actually talks to LEGO hardware (a Single Motor over BLE via `legoeducation`).

See the [LEGO-ROBOT README](https://github.com/YAKU-2003/LEGO-ROBOT#readme) for the full gesture-scheme writeup and known limitations.
