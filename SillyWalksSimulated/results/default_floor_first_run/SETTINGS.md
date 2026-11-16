# First experiment: default floor, solid-ABS masses (2026-11-15)

The first Drive-Straight training, kept for comparison with the later `default/`, `laminate/` and `carpet/` runs.
Its raw job folder (intermediate checkpoints, TensorBoard logs, config snapshot) was deleted; this file records
the settings it used.

| Setting | This run | Later runs |
|---|---|---|
| Masses | every part solid ABS: motor 100.4 g, each hub + leg 5.5 g (no pins yet) | measured: 95.0 g total |
| Floor | MuJoCo default plane, friction 1.0, solref (0.02, 1) | per surface (`SURFACES` in `drive_task.py`) |
| Chassis friction | set to 0.01, but had no effect (MuJoCo uses the larger friction of the two geoms) | removed; robot geoms get contact priority |
| `START_HEIGHT` | 0.02 m (robot dropped ~17 mm at the start of each run) | 0.0032 m (rests on the floor) |
| `njmax` | mjlab default (some contacts were dropped: "nefc overflow") | 400 (no overflow) |
| Task name | `Drive-Straight` | `Drive-Straight-Default` / `-Laminate` / `-Carpet` |
| Unchanged | `TARGET_SPEED` 0.07, `MAX_WHEEL_SPEED` 10, `WHEEL_KD` 0.05, `WHEEL_TORQUE_LIMIT` 0.2, `START_YAW` 0.4, `FACE_YAW` +90°, rewards and weights, 4096 robots, 2000 iterations | same |

Files: `trained_policy_walk.mp4` (policy, 10 s), `test_drive.mp4` (scripted drive.py plan), `model_1999.pt`,
`trained_policy_track.txt`, `training_reward.csv`, `training_output.txt` (full training log), `trained_policy_frames.png`.
