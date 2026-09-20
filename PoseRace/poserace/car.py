"""Thin wrapper around legoeducation's DoubleMotor for the race car."""

import legoeducation as le


class Car:
    """Connects to a LEGO Education Double Motor and drives it tank-style."""

    def __init__(self):
        self._motor = le.DoubleMotor()
        self._connected = False

    def connect(self, card_serial: str | None = None) -> bool:
        """Connect to a Double Motor, optionally restricted to the hub
        paired with a specific Connection Card serial number. Filtering by
        card is important in a classroom: with many hubs powered on at
        once, an unfiltered scan can connect to someone else's car.
        """
        if card_serial:
            print(f"Scanning for LEGO Double Motor with Connection Card {card_serial!r}...")
        else:
            print("Scanning for LEGO Double Motor over Bluetooth (no card filter)...")
        result = self._motor.connect(card_serial=card_serial)
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
