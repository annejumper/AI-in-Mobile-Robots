# Which way you're facing, from the ankle gyro alone (no compass, no GPS).
#
# Turning your body turns your ankle about the vertical, so the heading rate
# is the gyro's rotation about "up". "Up" in the sensor's frame changes all
# the time, because the shank swings back and forth with every step. So it's
# tracked: every sample the gyro says how far the sensor rotated, and "up" is
# rotated the opposite way to match. That alone would slowly drift, so
# whenever the accelerometer reads about 1 g (the foot is planted and only
# gravity is acting) "up" is nudged toward the direction gravity pushes.
#
# (The first version averaged the accelerometer over a second instead. On
# the first practice walk that drifted 6-12 degrees to the right on every
# straight stride, because the average lags the swinging shank. Tracking
# "up" with the gyro brought it to about 1 degree per stride.)
#
# The MPU-6050's gyro reads a degree or two per second even when it's still
# (its bias). Integrated over a 10 minute walk that would be hundreds of
# degrees, so the bias is measured whenever you stand still for half a
# second and subtracted. Stand still for a couple of seconds before a walk
# so it's measured at least once.
#
# Heading is in degrees, left turns positive (counterclockwise seen from
# above), counted from wherever you faced when it was last reset. It is not
# wrapped to +-180, so a full turn reads 360.

from collections import deque

import numpy as np

SAMPLE_HZ = 100
DT = 1.0 / SAMPLE_HZ
UP_GAIN = 0.05              # how far "up" moves toward gravity per still-ish sample
UP_ACC_TOL_G = 0.15         # |accel| within this of 1 g counts as "only gravity"
STILL_N = 50                # samples (0.5 s) that must all be still to measure the bias
STILL_GYRO_STD_DPS = 1.0    # per-axis spread allowed while "still"
STILL_ACC_G = 0.05          # |accel| within this of 1 g while "still"
BIAS_BLEND = 0.2            # each new still stretch moves the bias this far


class HeadingTracker:
    def __init__(self):
        self.up = None              # unit vector, sensor frame
        self.bias = np.zeros(3)     # deg/s
        self.bias_measured = False
        self.heading = 0.0          # degrees, left positive
        self.still = deque(maxlen=STILL_N)
        self._since_check = 0

    def reset(self):
        """Call the current direction 0 degrees."""
        self.heading = 0.0

    def is_still(self):
        """True if the last STILL_N samples were all still (you're standing)."""
        if len(self.still) < STILL_N:
            return False
        s = np.array(list(self.still))      # list() first: the web thread calls this too
        acc_ok = np.all(np.abs(np.linalg.norm(s[:, :3], axis=1) - 1.0) < STILL_ACC_G)
        return bool(acc_ok and np.all(s[:, 3:].std(axis=0) < STILL_GYRO_STD_DPS))

    def add(self, sample):
        """Feed one [ax, ay, az (g), gx, gy, gz (deg/s)] sample; returns the heading."""
        a = np.asarray(sample[:3], dtype=float)
        n = np.linalg.norm(a)
        self.still.append(sample[:6])
        self._since_check += 1
        if self._since_check >= STILL_N // 2:
            self._since_check = 0
            if self.is_still():
                mean = np.array(list(self.still))[:, 3:].mean(axis=0)
                self.bias = mean if not self.bias_measured else \
                    (1 - BIAS_BLEND) * self.bias + BIAS_BLEND * mean
                self.bias_measured = True
        if self.up is None:
            if n > 0.5:
                self.up = a / n
            return self.heading
        w = np.radians(np.asarray(sample[3:6], dtype=float) - self.bias)   # rad/s
        # A direction fixed in the world, seen from a sensor rotating at w,
        # turns at -w: d(up)/dt = up x w.
        up = self.up + np.cross(self.up, w) * DT
        if abs(n - 1.0) < UP_ACC_TOL_G:
            up = up + UP_GAIN * (a / n - up)
        self.up = up / np.linalg.norm(up)
        self.heading += float(np.degrees(np.dot(w, self.up))) * DT
        return self.heading
