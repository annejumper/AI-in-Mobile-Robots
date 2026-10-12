"""
Tennis physics for Virtual Tennis: court geometry, court surfaces, player
levels and ball flight (gravity + Magnus effect from spin + surface bounce).

World coordinates (meters), all measured from the player:
    x  sideways   (+ = player's right)
    y  height     (0 = ground)
    z  depth      (0 = the player's contact plane, + = toward the opponent)
"""

from dataclasses import dataclass

import numpy as np

G = 9.81
MAGNUS = 7.0        # extra dip (topspin) / lift (slice) in m/s^2 at full spin and 25 m/s
MIN_G_EFF = 2.0     # even a huge slice can't make the ball float forever
BALL_RADIUS = 0.033

# Court geometry (singles court; the player stands 1 m behind their baseline)
BASELINE_Z = 1.0
NET_Z = BASELINE_Z + 11.885
FAR_BASELINE_Z = NET_Z + 11.885
OPPONENT_Z = FAR_BASELINE_Z + 1.0
SERVICE_LINE = 6.40   # distance from the net
HALF_WIDTH = 4.115
NET_HEIGHT = 0.914
NET_MARGIN = 0.10     # how far over the net every shot is aimed
HIP_HEIGHT = 0.95     # used to map ball heights onto the player's body


@dataclass(frozen=True)
class Surface:
    name: str
    restitution: float   # how much vertical speed survives the bounce (bounce height)
    friction: float      # how much forward speed survives the bounce (slow vs. skid)
    spin_grip: float     # how strongly spin changes the bounce
    court_bgr: tuple
    light: str           # hub light color for this surface
    blurb: str


SURFACES = {
    "clay":  Surface("Clay",  0.80, 0.68, 1.4, (45, 95, 190),  "orange", "high, slow bounce - spin bites"),
    "grass": Surface("Grass", 0.58, 0.92, 0.6, (50, 140, 55),  "green",  "low, fast skid - get down low"),
    "hard":  Surface("Hard",  0.72, 0.82, 1.0, (150, 85, 35),  "blue",   "true, medium bounce"),
}


@dataclass(frozen=True)
class Level:
    name: str
    speed: float        # opponent ball speed, m/s
    hit_radius: float   # how close (m) the racket must be to the ball
    z_early: float      # earliest ball depth (m in front of you) a swing counts
    z_late: float       # latest ball depth (m behind you) a swing counts
    bounce_lo: float    # where the incoming ball bounces (m in front of you)
    bounce_hi: float
    spin_max: float     # how much spin the opponent puts on the ball
    ghost_early: bool   # show the "hit here" ring before the bounce


LEVELS = {
    1: Level("Club",    13.0, 0.45, 3.0, -1.5, 6.5, 9.0, 0.3, True),
    2: Level("Varsity", 18.0, 0.35, 2.4, -1.2, 6.0, 8.5, 0.6, False),
    3: Level("Pro",     25.0, 0.27, 1.8, -0.9, 6.0, 9.0, 1.0, False),
}


class Ball:
    """A tennis ball in flight. spin > 0 is topspin, < 0 is slice (backspin);
    side is sidespin, which curves the ball left/right."""

    def __init__(self, pos, vel, spin=0.0, side=0.0):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.spin = float(spin)
        self.side = float(side)
        self.bounces = 0
        self._set_magnus()

    def copy(self):
        b = Ball(self.pos, self.vel, self.spin, self.side)
        b.bounces = self.bounces
        return b

    def _set_magnus(self):
        # The Magnus force is fixed for each flight segment (recomputed after
        # every bounce), so the launch solvers below can aim exactly.
        speed = float(np.hypot(self.vel[0], self.vel[2]))
        self.ay = -max(G + MAGNUS * self.spin * speed / 25.0, MIN_G_EFF)
        self.ax = MAGNUS * self.side * speed / 25.0 * np.sign(self.vel[2])

    def _bounce(self, surface):
        grip = surface.spin_grip * self.spin
        # Topspin kicks up and forward; slice stays low and checks up.
        self.vel[1] = -self.vel[1] * surface.restitution * (1 + 0.12 * grip)
        self.vel[2] *= surface.friction * (1 + 0.15 * grip)
        self.vel[0] *= surface.friction
        self.spin *= 0.5
        self.side *= 0.5
        self.bounces += 1
        self._set_magnus()

    def step(self, dt, surface, h=0.002):
        """Advance by dt seconds. Returns a list of events:
        ("net", height when crossing the net) and ("bounce", z)."""
        events = []
        for _ in range(max(1, int(round(dt / h)))):
            prev_z = self.pos[2]
            self.vel[0] += self.ax * h
            self.vel[1] += self.ay * h
            self.pos += self.vel * h
            if (prev_z - NET_Z) * (self.pos[2] - NET_Z) < 0:
                events.append(("net", self.pos[1]))
            if self.pos[1] <= 0 and self.vel[1] < 0:
                self.pos[1] = 0.0
                self._bounce(surface)
                events.append(("bounce", self.pos[2]))
        return events


def predict_contact(ball, surface, max_t=6.0):
    """Fly a copy of an incoming ball until it reaches the player's contact
    plane (z = 0). Returns (x, y, seconds_from_now, bounces_before_contact)."""
    b = ball.copy()
    t = 0.0
    while b.pos[2] > 0 and t < max_t:
        b.step(0.004, surface, h=0.004)
        t += 0.004
    return b.pos[0], b.pos[1], t, b.bounces


REACH_LO, REACH_HI = 0.30, 1.75   # contact heights a person can actually hit


def launch_incoming(level, surface, rng, speed_mult=1.0, z0=OPPONENT_Z, x_target=0.6):
    """The opponent's shot: aimed to bounce in front of the player and arrive
    at sideways position x_target, at a height the player can reach.
    Returns (ball, predicted (x, y))."""
    for _ in range(15):
        ball, (cx, cy) = _aim_incoming(level, surface, rng, speed_mult, z0, x_target)
        if REACH_LO <= cy <= REACH_HI:
            break
    return ball, (cx, cy)


def _aim_incoming(level, surface, rng, speed_mult, z0, x_target):
    v = level.speed * speed_mult
    spin = rng.uniform(-0.4, 1.0) * level.spin_max
    zb = rng.uniform(level.bounce_lo, level.bounce_hi)
    h0 = 1.2
    g_eff = max(G + MAGNUS * spin * v / 25.0, MIN_G_EFF)
    t_b = (z0 - zb) / v
    vy = (0.5 * g_eff * t_b ** 2 - h0) / t_b
    if z0 > NET_Z:  # make sure it clears the net
        t_n = (z0 - NET_Z) / v
        vy = max(vy, (NET_HEIGHT + NET_MARGIN - h0 + 0.5 * g_eff * t_n ** 2) / t_n)

    x0 = rng.uniform(-2.5, 2.5)
    vx = (x_target - x0) / (z0 / v)
    for _ in range(4):  # friction at the bounce bends x, so correct a few times
        cx, cy, ct, _ = predict_contact(Ball((x0, h0, z0), (vx, vy, -v), spin), surface)
        vx += (x_target - cx) / max(ct, 0.1)
    ball = Ball((x0, h0, z0), (vx, vy, -v), spin)
    cx, cy, _, _ = predict_contact(ball, surface)
    return ball, (cx, cy)


def shot_speed(power):
    """Player's racket power (0-1) -> ball speed in m/s."""
    return 12.0 + 22.0 * power


def launch_return(x, y, power, spin, side, rng, aim_z=20.0):
    """The player's shot from contact point (x, y). The 'racket face' is
    aimed to land at aim_z for this speed and spin, but always clears the
    net by NET_MARGIN -- so a fast, flat ball struck from low down has to
    go up to clear the net and then sails LONG. Topspin pulls it back in."""
    v = shot_speed(power)
    h0 = max(y, 0.15)
    g_eff = max(G + MAGNUS * spin * v / 25.0, MIN_G_EFF)
    t = aim_z / v
    vy = (0.5 * g_eff * t ** 2 - h0) / t
    t_n = NET_Z / v
    vy = max(vy, (NET_HEIGHT + NET_MARGIN - h0 + 0.5 * g_eff * t_n ** 2) / t_n)
    aim_x = float(np.clip(-0.8 * x + rng.uniform(-2.5, 2.5), -3.3, 3.3))
    vx = (aim_x - x) / (FAR_BASELINE_Z / v)
    return Ball((x, h0, 0.0), (vx, vy, v), spin, side)
