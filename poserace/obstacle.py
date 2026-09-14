"""A LEGO Education Single Motor that sweeps an obstacle back and forth
on top of the car, independent of the drive wheels."""

import legoeducation as le

# Degrees the motor turns each leg of the sweep before reversing direction.
SWEEP_DEGREES = 180

# Speed as a percentage (0-100) for the sweeping motion.
SWEEP_SPEED = 40


class ObstacleMotor:
    """Connects to a Single Motor and continuously sweeps it back and forth."""

    def __init__(self):
        self._motor = le.SingleMotor()
        self._connected = False
        self._direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE

    def connect(self, card_serial: str | None = None) -> bool:
        if card_serial:
            print(f"Scanning for LEGO Single Motor with Connection Card {card_serial!r}...")
        else:
            print("Scanning for LEGO Single Motor over Bluetooth (no card filter)...")
        result = self._motor.connect(card_serial=card_serial)
        self._connected = result is not None or self._motor.done()
        if self._connected:
            print("Connected to LEGO Single Motor.")
        else:
            print("Could not connect to a LEGO Single Motor.")
        return self._connected

    def start(self):
        """Kick off the first sweep. Call update() regularly afterward to
        keep it reversing direction once each sweep completes."""
        if not self._connected:
            return
        self._motor.motor_run_for_degrees(
            SWEEP_DEGREES, direction=self._direction, speed=SWEEP_SPEED, blocking=False
        )

    def update(self):
        """Reverse direction once the current sweep leg finishes. Call this
        once per main loop iteration; it's a no-op most of the time."""
        if not self._connected:
            return
        if self._motor.done():
            self._direction = (
                le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
                if self._direction == le.MOTOR_MOVE_DIRECTION_CLOCKWISE
                else le.MOTOR_MOVE_DIRECTION_CLOCKWISE
            )
            self._motor.motor_run_for_degrees(
                SWEEP_DEGREES, direction=self._direction, speed=SWEEP_SPEED, blocking=False
            )

    def stop(self):
        if not self._connected:
            return
        self._motor.motor_stop(blocking=False)

    def disconnect(self):
        if self._connected:
            self._motor.disconnect()
            self._connected = False
