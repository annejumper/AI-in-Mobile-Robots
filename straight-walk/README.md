# straight-walk

Tabular Q-learning that teaches a two-legged LEGO® Education CS & AI walker
(driven by one Double Motor) to walk in a straight line.

## How it works

[walk_rl.py](walk_rl.py) uses the Double Motor's built-in IMU yaw as its only
feedback. Each step is one full leg rotation (one stride), so every heading
reading is taken at the same point in the gait.

- **State (21):** heading error from the start direction (7 ranges, from
  < -30° to >= 30°) x turn during the last stride (turning one way / steady /
  turning the other way).
- **Actions (5):** a speed bias between the legs: left/right =
  `50 - b` / `50 + b` for b in -20, -10, 0, +10, +20.
- **Reward:** highest when pointed straight; penalties for swinging and for
  large corrections; -10 and the episode ends if heading drifts past 45°.
- **Learning:** Q-learning (alpha 0.2, gamma 0.9), epsilon-greedy exploration
  decaying from 0.5 to 0.05. The table is saved to `q_table.json` after every
  episode, so training can be stopped and resumed.

After training, the robot corrects in real time once per stride: read
heading -> look up state -> take the best steering action.

## Setup

1. Install Python 3.11+ and the library:

   ```
   pip install -r requirements.txt
   ```

2. Set `CARD_SERIAL` in `walk_rl.py` to the number on your Double Motor's
   Connection Card.
3. Turn on the Double Motor and run:

   ```
   python walk_rl.py check    # connect, print IMU readings, walk 8 strides
   python walk_rl.py train    # learn (or keep learning) the Q-table
   python walk_rl.py run      # walk using the learned Q-table, no exploring
   python walk_rl.py show     # print the Q-table (no robot needed)
   ```

During `train`, point the robot straight and press Enter before each episode;
Ctrl+C stops and saves.

## Tuning

- `YAW_SCALE`: set to 0.1 if `check` shows yaw in tenths of a degree.
- `YAW_FACE`: set (e.g. `le.DEVICE_FACE_RIGHT`) if yaw doesn't change when the
  robot turns, which happens when the motor is mounted on its side.
- `BASE_SPEED`, `STEERING`, `STEP_DEGREES`: walking speed, correction strength,
  and how often it corrects.

## Reference

Uses the official LEGO Education Python API
([GitHub](https://github.com/LEGO/LEGOEducation),
[PyPI](https://pypi.org/project/legoeducation/)), distributed under the
Business Source License 1.1.
