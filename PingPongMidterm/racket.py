"""
The racket: a LEGO Education Double Motor held in the hand.

  * IMU thread  - reads the hub's gyro/accelerometer as fast as notifications
                  arrive and turns swings into SwingEvents.
  * Haptics thread - the only thread that sends commands to the hub (motor
                  pulses, beeps, light), so the game loop never waits on BLE.

Swing physics (two gyro axes, like a real racket):
  PITCH rate  (the swing itself)        -> did you swing, and how hard   (power)
  YAW twist   (torque about the handle) -> spin, with a direction:
                twist one way = TOPSPIN, the other way = SLICE, none = FLAT
    A torque applied for a moment changes the angular velocity
    (angular impulse: integral of torque dt = I * delta-omega), so the peak
    yaw rate during the swing measures how much twisting torque you applied.

Run imu_monitor.py to see your own numbers and tune the thresholds below.
"""

import math
import queue
import threading
import time
from dataclasses import dataclass

import legoeducation as le

# --- swing detection tuning (raw gyro units are ~deg/s) ---------------------
SWING_AXIS = "y"        # gyro axis of the swing (pitch, the way Anne holds the hub)
SPIN_AXIS = "z"         # gyro axis of the twist (yaw)
SWING_START = 250       # |pitch rate| that starts a swing
SWING_END_FRAC = 0.4    # swing ends when |rate| drops below SWING_START * this
SWING_FULL = 1200       # |pitch rate| that counts as a 100% power swing
SPIN_DEAD = 120         # |yaw rate| below this during a swing = FLAT (no twist)
SPIN_FULL = 600         # |yaw rate| that gives maximum spin
SPIN_SIGN = 1           # set to -1 if your topspin twist reads as slice
BACKHAND_SPIN_SIGN = 1  # set to -1 if your BACKHAND topspin twist reads as slice (untrained only)
SWING_MAX_S = 0.6       # a "swing" longer than this is cut off
REFRACTORY_S = 0.06     # short: a windup is followed almost at once by the real swing

LIGHTS = {
    "orange": le.LEGO_COLOR_ORANGE, "green": le.LEGO_COLOR_GREEN,
    "blue": le.LEGO_COLOR_BLUE, "white": le.LEGO_COLOR_WHITE,
    "red": le.LEGO_COLOR_RED, "magenta": le.LEGO_COLOR_MAGENTA,
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def spin_amount(twist_rate):
    """Signed yaw rate -> spin in [-1, 1] (+ topspin, - slice), 0 inside the dead zone."""
    mag = clamp((abs(twist_rate) - SPIN_DEAD) / (SPIN_FULL - SPIN_DEAD), 0.0, 1.0)
    return math.copysign(mag, twist_rate * SPIN_SIGN) if mag > 0 else 0.0


@dataclass
class SwingEvent:
    t_start: float
    t_peak: float
    t_end: float
    peak_rate: float       # signed pitch rate at the fastest point of the swing
    peak_twist: float      # signed yaw rate with the largest magnitude during the swing
    other_peak: float      # signed peak rate on the remaining gyro axis
    peak_accel: float      # largest accelerometer magnitude
    label: str = None      # e.g. "forehand_topspin" once the game has judged a hit with the KNN

    @property
    def power(self):
        return clamp(abs(self.peak_rate) / SWING_FULL, 0.0, 1.0)

    @property
    def spin(self):
        return spin_amount(self.peak_twist)

    def features(self):
        """Feature vector for the stroke KNN. The swing rate keeps its sign,
        since forehand and backhand swing the hub in different directions."""
        return [self.peak_rate, self.peak_twist * SPIN_SIGN, self.other_peak,
                self.peak_accel, self.t_end - self.t_start]


class SwingDetector:
    """Feed it IMU samples; it returns a SwingEvent when a swing finishes."""

    def __init__(self, swing_axis=SWING_AXIS, spin_axis=SPIN_AXIS):
        self.axis = "xyz".index(swing_axis)
        self.spin_axis = "xyz".index(spin_axis)
        self.other_axis = ({0, 1, 2} - {self.axis, self.spin_axis}).pop()
        self.rate = 0.0
        self.twist = 0.0
        self.any_axis = False   # serve: an overhead swing turns the hub on other axes than pitch
        self._swing = None
        self._quiet_until = 0.0

    @property
    def in_swing(self):
        return self._swing is not None

    def feed(self, t, gyro, accel):
        rate = float(gyro[self.axis])
        if self.any_axis:      # count the total turning speed (unsigned) as the swing
            rate = math.sqrt(sum(float(g) ** 2 for g in gyro))
        twist = float(gyro[self.spin_axis])
        other = float(gyro[self.other_axis])
        self.rate, self.twist = rate, twist
        amag = math.sqrt(sum(float(a) ** 2 for a in accel))

        s = self._swing
        if s is None:
            if abs(rate) >= SWING_START and t >= self._quiet_until:
                self._swing = dict(t_start=t, t_peak=t, peak_rate=rate, twist=twist,
                                   other=other, peak_accel=amag)
            return None

        if abs(rate) > abs(s["peak_rate"]):
            s["t_peak"], s["peak_rate"] = t, rate
        if abs(twist) > abs(s["twist"]):
            s["twist"] = twist
        if abs(other) > abs(s["other"]):
            s["other"] = other
        s["peak_accel"] = max(s["peak_accel"], amag)

        if abs(rate) < SWING_START * SWING_END_FRAC or t - s["t_start"] > SWING_MAX_S:
            self._swing = None
            self._quiet_until = t + REFRACTORY_S
            return SwingEvent(s["t_start"], s["t_peak"], t, s["peak_rate"],
                              s["twist"], s["other"], s["peak_accel"])
        return None


class Racket:
    """Wraps a connected doubleMotor: swing events in, haptic feedback out."""

    def __init__(self, dm, swing_axis=SWING_AXIS, spin_axis=SPIN_AXIS, knn=None):
        self.dm = dm
        self.detector = SwingDetector(swing_axis, spin_axis)
        self.knn = knn
        self.events = queue.Queue()
        self._cmds = queue.Queue()
        self._running = True
        self.light_name = "white"
        threading.Thread(target=self._imu_loop, daemon=True).start()
        threading.Thread(target=self._haptic_loop, daemon=True).start()

    # --- inputs -------------------------------------------------------------
    @property
    def in_swing(self):
        return self.detector.in_swing or not self.events.empty()

    @property
    def live_rate(self):
        return self.detector.rate

    @property
    def live_twist(self):
        return self.detector.twist

    def serve_mode(self, on):
        self.detector.any_axis = on

    def poll_swings(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get_nowait())
        return out

    def _imu_loop(self):
        last = None
        while self._running:
            imu = self.dm.imu_device
            if imu is not last:  # the library swaps in a new object per notification
                last = imu
                gyro = (imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ)
                accel = (imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ)
                if not any(math.isnan(v) for v in gyro):
                    ev = self.detector.feed(time.monotonic(), gyro, accel)
                    if ev:
                        self.events.put(ev)
            time.sleep(0.002)

    # --- haptics / sound / light -------------------------------------------
    def _haptic_loop(self):
        while self._running:
            fn = self._cmds.get()
            if fn is None:
                break
            try:
                fn()
            except Exception as e:
                print(f"[racket] hub command failed: {e}")

    def _do(self, fn):
        if self._cmds.qsize() < 4:  # never let feedback pile up behind the game
            self._cmds.put(fn)

    def _pulse(self, speed, ms, reverse=True):
        """One haptic 'kick': jerk both motors one way, then back, so the
        reaction torque is felt in the hand and the shafts end where they began."""
        speed = int(clamp(speed, -100, 100))
        self.dm.movement_move_tank(speed, speed)
        time.sleep(ms / 1000)
        if reverse:
            self.dm.movement_move_tank(-speed, -speed)
            time.sleep(ms / 1000)
        self.dm.movement_stop()

    def _beep(self, freq, pattern=le.SOUND_PATTERN_BEEP_SINGLE):
        self.dm.beep(pattern=pattern, frequency=int(freq), blocking=False)

    def light(self, name, pattern=le.LIGHT_PATTERN_SOLID):
        self.light_name = name
        self._do(lambda: self.dm.light_color(LIGHTS[name], pattern=pattern))

    def bounce_tick(self):
        """Tiny tap when the incoming ball bounces -- a timing cue you can feel."""
        self._do(lambda: self._pulse(35, 25))

    def hit(self, power, perfect, spin_label):
        speed = 45 + 55 * power
        restore = self.light_name

        def go():
            self.dm.light_color(le.LEGO_COLOR_WHITE)
            self._beep(1319 if perfect else 880)
            if spin_label == "TOPSPIN":       # brushy: rapid triple buzz
                for _ in range(3):
                    self._pulse(speed, 18)
            elif spin_label == "SLICE":       # smooth: one long, soft push
                self._pulse(speed * 0.6, 110)
            else:                             # flat / sidespin: one solid thump
                self._pulse(speed, 45)
            if perfect:
                self._pulse(100, 15)
            self.dm.light_color(LIGHTS[restore])
        self._do(go)

    def landed_in(self):
        self._do(lambda: self._beep(1568))

    def miss(self):
        restore = self.light_name

        def go():
            self.dm.light_color(le.LEGO_COLOR_RED, pattern=le.LIGHT_PATTERN_SHORT_BLINK)
            self._beep(196, le.SOUND_PATTERN_BEEP_DOUBLE)
            for _ in range(3):
                self._pulse(60, 70)
            time.sleep(0.6)
            self.dm.light_color(LIGHTS[restore])
        self._do(go)

    def new_record(self):
        self._do(lambda: (self._beep(1047, le.SOUND_PATTERN_BEEP_UP_MIDDLE_DOWN),
                          self.dm.light_color(le.LEGO_COLOR_MAGENTA, pattern=le.LIGHT_PATTERN_DOUBLE_BLINK)))

    def countdown(self, n):
        self._do(lambda: self._beep(1320 if n == 0 else 660))

    def close(self):
        self._running = False
        self._cmds.put(None)
        try:
            self.dm.movement_stop()
        except Exception:
            pass



class SimRacket:
    """Keyboard stand-in for testing without the hub:
    SPACE = flat swing, T = topspin, B = slice, F = full-power flat."""

    KEYS = {ord(" "): (0.6, 0.0), ord("t"): (0.85, 0.8),
            ord("b"): (0.45, -0.6), ord("f"): (1.0, 0.0)}   # (power, spin)

    def __init__(self):
        self.events = []
        self.in_swing = False
        self._shown = (0.0, 0.0, 0.0)  # rate, twist, time

    def key(self, k):
        if k not in self.KEYS:
            return False
        power, spin = self.KEYS[k]
        t = time.monotonic() - 0.05
        rate = power * SWING_FULL
        twist = math.copysign(SPIN_DEAD + abs(spin) * (SPIN_FULL - SPIN_DEAD), spin * SPIN_SIGN) if spin else 0.0
        self.events.append(SwingEvent(t - 0.1, t, t + 0.1, rate, twist, 0.0, 0.0))
        self._shown = (rate, twist, time.monotonic())
        return True

    def _fade(self, v):
        return v * max(0.0, 1 - (time.monotonic() - self._shown[2]) / 0.4)

    @property
    def live_rate(self):
        return self._fade(self._shown[0])

    @property
    def live_twist(self):
        return self._fade(self._shown[1])

    def poll_swings(self):
        out, self.events = self.events, []
        return out

    def serve_mode(self, on): pass
    def light(self, name, pattern=None): pass
    def bounce_tick(self): pass
    def hit(self, power, perfect, spin_label): pass
    def landed_in(self): pass
    def miss(self): pass
    def new_record(self): pass
    def countdown(self, n): pass
    def close(self): pass
