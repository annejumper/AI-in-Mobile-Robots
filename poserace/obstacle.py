"""A LEGO Education Single Motor that swings an obstacle on top of the
car, triggered by hand gestures rather than running continuously."""

import legoeducation as le

# Degrees the motor turns for each triggered move.
TRIGGER_DEGREES = 90

# Speed as a percentage (0-100) for the triggered move.
TRIGGER_SPEED = 80


class ObstacleMotor:
    """Connects to a Single Motor and rotates it on command."""

    def __init__(self):
        self._motor = le.SingleMotor()
        self._connected = False

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

    def trigger_cw(self):
        """Rotate TRIGGER_DEGREES clockwise. Non-blocking."""
        if not self._connected:
            return
        self._motor.motor_run_for_degrees(
            TRIGGER_DEGREES, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, speed=TRIGGER_SPEED, blocking=False
        )

    def trigger_ccw(self):
        """Rotate TRIGGER_DEGREES counterclockwise. Non-blocking."""
        if not self._connected:
            return
        self._motor.motor_run_for_degrees(
            TRIGGER_DEGREES, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, speed=TRIGGER_SPEED, blocking=False
        )

    def stop(self):
        if not self._connected:
            return
        self._motor.motor_stop(blocking=False)

    def disconnect(self):
        if self._connected:
            self._motor.disconnect()
            self._connected = False
