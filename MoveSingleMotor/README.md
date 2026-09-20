# MoveSingleMotor

A small Python script for driving a LEGO® Education Single Motor using the
official `legoeducation` Python API.

## What this is

[single_motor.py](single_motor.py) connects to a LEGO Education Single Motor over
Bluetooth, runs it 180° clockwise at 50% speed, then disconnects.

## Setup

1. Install Python 3.14+.
2. Install the library:

   ```
   pip install legoeducation
   ```

3. Update `card_color` and `card_serial` in `single_motor.py` to match the
   Connection Card printed on your Single Motor.
4. Power on / charge the motor so it's broadcasting over Bluetooth, then run:

   ```
   python single_motor.py
   ```

## Reference

This project uses the official LEGO Education Python API:

- Source: https://github.com/LEGO/LEGOEducation
- PyPI: https://pypi.org/project/legoeducation/

See that repository's `singlemotor.md` and `constants.md` for the full list of
motor functions (`motor_run`, `motor_run_for_degrees`, `motor_run_for_time`,
`motor_set_speed`, `motor_stop`, etc.) and available constants
(colors, directions, light/sound patterns).

The `legoeducation` package is distributed by The LEGO Group under the
Business Source License 1.1, which permits non-commercial academic,
research, and personal use.
