"""Thin wrapper around legoeducation's DoubleMotor for the race car."""

import legoeducation as le


class Car:
    """Connects to a LEGO Education Double Motor and drives it tank-style."""

    def __init__(self):
        self._motor = le.DoubleMotor()
        self._connected = False

    def connect(self, timeout: int = 10) -> bool:
        print("Scanning for LEGO Double Motor over Bluetooth...")
        result = self._motor.connect()
        self._connected = result is not None or self._motor.done()
        if self._connected:
            print("Connected to LEGO Double Motor.")
        else:
            print("Could not connect to a LEGO Double Motor.")
        return self._connected

    def drive_tank(self, speed_left: float, speed_right: float):
        """Set both wheel speeds at once. Non-blocking so the vision loop
        never stalls waiting on a BLE acknowledgment."""
        if not self._connected:
            return
        self._motor.movement_move_tank(int(speed_left), int(speed_right), blocking=False)

    def stop(self):
        if not self._connected:
            return
        self._motor.movement_stop(blocking=False)

    def disconnect(self):
        if self._connected:
            self._motor.disconnect()
            self._connected = False
