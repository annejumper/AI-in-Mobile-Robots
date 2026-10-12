"""
VIRTUAL TENNIS  -  ME193 Midterm  -  Anne Jumper

Your racket is a LEGO Education Double Motor (Connection Card 6065).

  POSE (MediaPipe)  - where your racket hand is: is the racket where the ball is?
  IMU  (Double Motor) - pitch swing = are you swinging, and how hard?
                        yaw twist (torque) = spin: one way topspin, other way slice
  APRILTAGS         - pick your level (ball speed) and court surface, and start.
  MQTT              - record rally (most continuous hits) -> ME193/Rogers/AnneJumper
  HAPTICS           - the hub's motors kick in your hand on every bounce and hit.

AprilTags (36h11):  0 = START    1 = Club   2 = Varsity   3 = Pro
                    4 = Clay     5 = Grass  6 = Hard court
Keys: Q quit | 1/2/3 level | C/G/H surface | S start
      (--sim, no hub: SPACE flat, T topspin, B slice, F full power)

Run:  .venv/bin/python tennis.py            (add --sim to play without the hub)
"""

import legoeducation as le
from lelib import doubleMotor

DOUBLE_MOTOR_CARD_SERIAL = 6065

import argparse
import os
import random
import time
from collections import deque

import cv2
import numpy as np

import racket as rk
from court import (BALL_RADIUS, BASELINE_Z, FAR_BASELINE_Z, HALF_WIDTH, LEVELS, NET_HEIGHT,
                   NET_Z, OPPONENT_Z, SERVICE_LINE, SURFACES, launch_incoming, launch_return,
                   shot_speed)
from vision import SKELETON, BodyFrame, PoseTracker, TagReader

TOPIC = "ME193/Rogers/AnneJumper"
TAG_START = 0
TAG_LEVELS = {1: 1, 2: 2, 3: 3}
TAG_SURFACES = {4: "clay", 5: "grass", 6: "hard"}
START_FRAMES = 6          # frames the START tag must be seen (debounce)
PERFECT_Z = 0.6           # swing within this many meters of the contact plane = PERFECT
SETTLE_S = 0.08           # wait this long after a swing for a stronger one (windup -> real swing)
HIT_NOW_POWER = 0.45      # weaker candidates wait until the ball has passed, in case a real swing follows
EARLY_FLASH_POWER = 0.5   # only scold 'too early' for a real swing, not a windup

WHITE, BLACK = (255, 255, 255), (0, 0, 0)
YELLOW, GREEN, RED, CYAN = (0, 230, 255), (90, 220, 90), (70, 70, 240), (230, 200, 60)
BALL_BGR = (0, 255, 204)  # optic yellow
FONT = cv2.FONT_HERSHEY_DUPLEX


# --- small helpers ------------------------------------------------------------

def interp(hist, t):
    """Linear interpolation into a deque of (time, *values). None if t is
    outside the recorded range."""
    if len(hist) < 2 or t < hist[0][0] or t > hist[-1][0] + 0.05:
        return None
    prev = hist[0]
    for cur in hist:
        if cur[0] >= t:
            if cur[0] - prev[0] < 1e-6:
                return cur[1:]
            f = (t - prev[0]) / (cur[0] - prev[0])
            return tuple(a + f * (b - a) for a, b in zip(prev[1:], cur[1:]))
        prev = cur
    return hist[-1][1:]


def text(img, s, org, scale=0.7, color=WHITE, thick=1, center=False):
    if center:
        (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
        org = (org[0] - w // 2, org[1] + h // 2)
    org = (int(org[0]), int(org[1]))
    cv2.putText(img, s, (org[0] + 2, org[1] + 2), FONT, scale, BLACK, thick + 1, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def shade(img, x1, y1, x2, y2, alpha=0.55):
    h, w = img.shape[:2]
    x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)


def ipt(p):
    return int(round(p[0])), int(round(p[1]))


class Scoreboard:
    """Publishes the record rally to MQTT as a float, e.g. '7.0'."""

    def __init__(self, enabled, topic):
        self.topic = topic
        self.client = None
        if enabled:
            try:
                from mqttlib import MQTTClient
                self.client = MQTTClient()
                self.client.connect()
                print(f"MQTT connected - posting record to {topic}")
            except Exception as e:
                print(f"MQTT unavailable ({e}); playing offline")
                self.client = None

    def post(self, record):
        if self.client:
            self.client.publish(self.topic, str(float(record)), retain=True)

    def close(self):
        if self.client:
            self.client.disconnect()


# --- the game -----------------------------------------------------------------

class Game:
    def __init__(self, racket, scoreboard, frame, hand, level, surface, sim):
        self.racket = racket
        self.board = scoreboard
        self.frame = frame
        self.hand_sign = 1 if hand == "right" else -1
        self.level_id = level
        self.surface_key = surface
        self.sim = sim
        self.rng = random.Random()
        self.state = "MENU"
        self.state_t = time.monotonic()
        self.ball = None
        self.ghost = None             # predicted contact point (x, y)
        self.candidate = None         # strongest swing in the hitting window: (SwingEvent, ball pos)
        self.show_ghost = False
        self.streak = 0
        self.record = 0
        self.record_this_rally = False
        self.start_seen = 0
        self.ball_hist = deque(maxlen=400)    # (t, x, y, z)
        self.racket_hist = deque(maxlen=400)  # (t, sx, sy, torso_px)
        self.trail = deque(maxlen=10)
        self.msg = None                       # (title, detail, color, until)
        self.last_shot = None
        self.last_swing = None
        self.fx = []                          # one-shot events for the 3D view (sounds, dust)
        self.body_seen = False
        self.racket_world = None              # racket hand in world meters, for the 3D view
        self._count = 4
        self.racket.light(self.surface.light)
        self.board.post(self.record)

    @property
    def level(self):
        return LEVELS[self.level_id]

    @property
    def surface(self):
        return SURFACES[self.surface_key]

    def set_state(self, s):
        self.state = s
        self.state_t = time.monotonic()

    def flash(self, title, detail="", color=WHITE, secs=1.2):
        self.msg = (title, detail, color, time.monotonic() + secs)

    # --- settings from AprilTags / keys --------------------------------------
    def choose(self, level=None, surface=None):
        if self.state not in ("MENU", "OVER"):
            return
        if level and level != self.level_id:
            self.level_id = level
            self.flash(f"Level: {self.level.name}", f"{self.level.speed:.0f} m/s balls", CYAN)
        if surface and surface != self.surface_key:
            self.surface_key = surface
            self.racket.light(self.surface.light)
            self.flash(f"{self.surface.name} court", self.surface.blurb, CYAN)

    def handle_tags(self, tags):
        for tid in tags:
            self.choose(level=TAG_LEVELS.get(tid), surface=TAG_SURFACES.get(tid))
        self.start_seen = self.start_seen + 1 if TAG_START in tags else 0
        if self.start_seen >= START_FRAMES:
            self.start()

    def start(self):
        if self.state in ("MENU", "OVER"):
            self.streak = 0
            self.record_this_rally = False
            self.ball = None
            self.set_state("COUNTDOWN")
            self._count = 4

    # --- per-frame update -----------------------------------------------------
    def update(self, t, dt, body, swings):
        self.frame.update(body)
        self.body_seen = body is not None
        self.racket_world = None
        if body is not None and body.racket is not None:
            self.racket_hist.append((t, body.racket[0], body.racket[1], body.torso_px))
            self.racket_world = self.frame.to_world(*body.racket)
        for sw in swings:
            self.last_swing = sw

        if self.state == "COUNTDOWN":
            n = 3 - int(t - self.state_t)
            if n != self._count:
                self._count = n
                self.racket.countdown(max(n, 0))
                self.fx.append({"type": "count", "n": max(n, 0)})
            if n <= 0:
                self.serve(OPPONENT_Z)
        elif self.state == "INCOMING":
            self.update_incoming(t, dt, swings)
        elif self.state == "OUTGOING":
            self.update_outgoing(dt)
        elif self.state == "OVER":
            if self.ball is not None and abs(self.ball.pos[2]) < 40:
                self.ball.step(dt, self.surface)   # let the dead ball roll away in the 3D view
            if t - self.state_t > 2.5:
                self.set_state("MENU")

    def serve(self, z0):
        side = self.rng.choice([1, -1])
        x_target = side * self.rng.uniform(0.35, 0.75)
        mult = min(1.0 + 0.03 * self.streak, 1.35)   # rallies heat up
        self.ball, self.ghost = launch_incoming(self.level, self.surface, self.rng, mult, z0, x_target)
        self.show_ghost = self.level.ghost_early
        self.candidate = None
        self.ball_hist.clear()
        self.trail.clear()
        self.fx.append({"type": "oppHit", "pos": self.ball.pos.tolist()})
        self.set_state("INCOMING")

    def update_incoming(self, t, dt, swings):
        for kind, _ in self.ball.step(dt, self.surface):
            if kind == "bounce":
                self.fx.append({"type": "bounce", "pos": self.ball.pos.tolist()})
                if self.ball.bounces == 1:
                    self.racket.bounce_tick()
                    self.show_ghost = True
        self.ball_hist.append((t, *self.ball.pos))
        self.trail.append(self.ball.pos.copy())

        for sw in swings:
            if self.state == "INCOMING":
                self.consider(sw)

        # Judge the strongest swing once no stronger one can follow it
        c = self.candidate
        if c and self.state == "INCOMING" and not self.racket.in_swing:
            settled = t - c[0].t_end > SETTLE_S
            passed = self.ball.pos[2] < self.level.z_late
            if (settled and c[0].power >= HIT_NOW_POWER) or passed:
                self.candidate = None
                self.judge(*c)

        if (self.state == "INCOMING" and self.ball.pos[2] < self.level.z_late - 0.4
                and not self.racket.in_swing and self.candidate is None):
            self.end_rally("No swing!", "Swing the racket as the ball reaches you")

    # --- THE POLICY: was that swing a hit? -------------------------------------
    def consider(self, sw):
        """1. TIMING (IMU): where was the ball at the fastest instant of the
        swing? Swings inside the hitting window become candidates and the
        STRONGEST one wins -- so a slow windup or reset never beats the real swing."""
        p = interp(self.ball_hist, sw.t_peak)
        if p is None:
            return
        z = p[2]
        lv = self.level
        if z > lv.z_early:
            if sw.power >= EARLY_FLASH_POWER:
                self.flash("Too early!", f"ball was still {z:.1f} m away", YELLOW, 0.8)
            return  # a free practice swing (or a windup); you can still hit this ball
        if z < lv.z_late:
            if self.candidate is None:
                self.end_rally("Too late!", f"ball was {-z:.1f} m behind you")
            return  # otherwise it's just the follow-through
        if self.candidate is None or sw.power > self.candidate[0].power:
            self.candidate = (sw, p)

    def judge(self, sw, p):
        x, y, z = p
        lv = self.level
        # 2. POSITION (pose): was the racket where the ball was?
        r = interp(self.racket_hist, sw.t_peak)
        if r is None:
            self.end_rally("Can't see your racket hand", "stay in view of the camera")
            return
        bx, by = self.frame.to_screen(x, y)
        dx = (r[0] - bx) / self.frame.ppm
        dy = (by - r[1]) / self.frame.ppm          # + means racket above the ball
        if np.hypot(dx, dy) > lv.hit_radius:
            if abs(dy) >= abs(dx):
                where = f"{abs(dy) * 100:.0f} cm too {'high' if dy > 0 else 'low'}"
                if dy > 0 and self.surface_key == "grass":
                    where += " - grass stays low!"
                elif dy < 0 and self.surface_key == "clay":
                    where += " - clay kicks up!"
            else:
                where = f"{abs(dx) * 100:.0f} cm too far {'right' if dx > 0 else 'left'}"
            self.end_rally("Missed!", f"racket {where}")
            return
        self.hit(sw, x, y, z)

    def spin_from(self, sw, stroke):
        """Spin comes from YAW TWIST during the swing (signed: + topspin,
        - slice, inside the dead zone = flat). If a trained KNN is loaded,
        it decides the stroke (forehand/backhand) and spin type, and the
        twist sets how much spin."""
        how = "yaw twist"
        topspin = sw.spin * (rk.BACKHAND_SPIN_SIGN if stroke == "backhand" else 1)
        if sw.label:                      # e.g. "backhand_slice"
            *knn_stroke, kind = sw.label.split("_")
            stroke = knn_stroke[0] if knn_stroke else stroke
            d = {"topspin": 1, "slice": -1}.get(kind, 0)
            topspin, how = d * max(abs(sw.spin), 0.4), "KNN"
        label = "TOPSPIN" if topspin > 0.15 else "SLICE" if topspin < -0.15 else "FLAT"
        return stroke, topspin, label, how

    def hit(self, sw, x, y, z):
        perfect = bool(abs(z) < PERFECT_Z)
        side = "forehand" if x * self.hand_sign > 0 else "backhand"
        stroke, topspin, label, how = self.spin_from(sw, side)
        power = sw.power
        self.racket.hit(power, perfect, label)
        self.ball = launch_return(x, y, power, topspin, 0.0, self.rng)
        self.trail.clear()
        self.last_shot = dict(power=power, speed=shot_speed(power), label=label, spin=topspin,
                              how=how, height=y, perfect=perfect, stroke=stroke)
        detail = f"{stroke}  {label.lower()}"
        if stroke != side:
            detail += f"  (a {stroke} on your {side} side!)"
        self.flash("PERFECT!" if perfect else "Nice hit!", detail, GREEN, 0.9)
        self.fx.append({"type": "hit", "pos": [x, y, 0.0], "power": power, "perfect": perfect,
                        "label": label, "stroke": stroke})
        self.show_ghost = False
        self.set_state("OUTGOING")

    def update_outgoing(self, dt):
        for kind, val in self.ball.step(dt, self.surface):
            if self.state != "OUTGOING":
                break
            if kind == "bounce":
                self.fx.append({"type": "bounce", "pos": self.ball.pos.tolist()})
            if kind == "net" and val < NET_HEIGHT:
                self.fx.append({"type": "net", "pos": self.ball.pos.tolist()})
                self.end_rally("Into the net!", "lift it more - less topspin from low balls")
            elif kind == "bounce" and self.ball.bounces == 1:
                if val > FAR_BASELINE_Z:
                    tip = ("add TOPSPIN to bring a big swing down" if self.last_shot["label"] != "TOPSPIN"
                           else "swing a little softer")
                    self.end_rally("LONG!", f"{val - FAR_BASELINE_Z:.1f} m out - {tip}")
                else:
                    self.rally_point()
        self.trail.append(self.ball.pos.copy())
        if self.state == "OUTGOING" and (self.ball.pos[2] >= OPPONENT_Z or self.ball.bounces >= 2):
            self.serve(max(self.ball.pos[2], NET_Z + 6))

    def rally_point(self):
        self.streak += 1
        self.racket.landed_in()
        self.fx.append({"type": "in", "pos": self.ball.pos.tolist()})
        if self.streak > self.record:
            if not self.record_this_rally and self.record > 0:
                self.fx.append({"type": "record"})
            self.record = self.streak
            self.record_this_rally = True
        self.board.post(self.record)          # real-time: every ball that lands in

    def end_rally(self, title, detail):
        self.candidate = None
        self.racket.miss()
        self.fx.append({"type": "miss", "title": title})
        if self.record_this_rally:
            self.racket.new_record()
            detail = f"{detail}   ***  NEW RECORD: {self.record}  ***"
        self.flash(title, detail, RED, 2.5)
        self.ghost = None
        self.set_state("OVER")

    # --- state for the Three.js view ---------------------------------------------
    MSG_KIND = {GREEN: "good", RED: "bad", YELLOW: "warn", CYAN: "info", WHITE: "info"}

    def snapshot(self):
        """Everything the browser needs to draw one frame (all positions in
        world meters). One-shot events in fx are handed over and cleared."""
        now = time.monotonic()
        lv, sf = self.level, self.surface
        ball = None
        if self.ball is not None and self.state in ("INCOMING", "OUTGOING", "OVER"):
            ball = {"pos": self.ball.pos.tolist(), "vel": self.ball.vel.tolist(),
                    "spin": self.ball.spin, "incoming": self.state == "INCOMING"}
        ghost = None
        if self.state == "INCOMING" and self.show_ghost and self.ghost:
            ghost = [float(self.ghost[0]), float(self.ghost[1])]
        msg = None
        if self.msg and now < self.msg[3]:
            msg = {"title": self.msg[0], "detail": self.msg[1],
                   "kind": self.MSG_KIND.get(self.msg[2], "info")}
        sw = self.last_swing
        fx, self.fx = self.fx, []
        return {
            "state": self.state,
            "countdown": max(0, 3 - int(now - self.state_t)) if self.state == "COUNTDOWN" else None,
            "surface": self.surface_key, "surfaceName": sf.name, "blurb": sf.blurb,
            "level": self.level_id, "levelName": lv.name, "speed": lv.speed, "hitRadius": lv.hit_radius,
            "streak": self.streak, "record": self.record, "topic": self.board.topic,
            "mqtt": self.board.client is not None,
            "ball": ball, "ghost": ghost,
            "racket": list(self.racket_world) if self.racket_world else None,
            "swinging": bool(self.racket.in_swing), "tracking": self.body_seen,
            "hand": "right" if self.hand_sign > 0 else "left",
            "imu": {"swing": abs(self.racket.live_rate) / rk.SWING_FULL,
                    "swingTick": rk.SWING_START / rk.SWING_FULL,
                    "twist": self.racket.live_twist / rk.SPIN_FULL,
                    "twistDead": rk.SPIN_DEAD / rk.SPIN_FULL},
            "lastSwing": {"power": sw.power, "spin": sw.spin, "label": sw.label} if sw else None,
            "lastShot": self.last_shot,
            "msg": msg,
            "startSeen": min(self.start_seen, START_FRAMES) / START_FRAMES,
            "sim": self.sim,
            "fx": fx,
        }

    # --- drawing (OpenCV view, and the webcam inset) ------------------------------
    def draw(self, img, body, tags):
        self.draw_court(img)
        if self.ball is not None and self.state in ("INCOMING", "OUTGOING", "OVER"):
            self.draw_ball(img)
        if self.state == "INCOMING" and self.show_ghost and self.ghost:
            c = ipt(self.frame.to_screen(*self.ghost))
            cv2.circle(img, c, int(self.level.hit_radius * self.frame.ppm), WHITE, 2, cv2.LINE_AA)
            text(img, "hit here", (c[0], c[1] - int(self.level.hit_radius * self.frame.ppm) - 12),
                 0.55, WHITE, center=True)
        self.draw_body(img, body)
        self.draw_tags(img, tags)
        self.draw_hud(img)

    def draw_court(self, img):
        f = self.frame
        P = lambda x, y, z: ipt(f.project(x, y, z)[0])
        over = img.copy()
        quad = np.int32([P(-HALF_WIDTH, 0, BASELINE_Z), P(HALF_WIDTH, 0, BASELINE_Z),
                         P(HALF_WIDTH, 0, FAR_BASELINE_Z), P(-HALF_WIDTH, 0, FAR_BASELINE_Z)])
        cv2.fillPoly(over, [quad], self.surface.court_bgr)
        net = np.int32([P(-HALF_WIDTH - 0.9, 0, NET_Z), P(HALF_WIDTH + 0.9, 0, NET_Z),
                        P(HALF_WIDTH + 0.9, NET_HEIGHT, NET_Z), P(-HALF_WIDTH - 0.9, NET_HEIGHT, NET_Z)])
        cv2.fillPoly(over, [net], (40, 40, 40))
        cv2.addWeighted(over, 0.4, img, 0.6, 0, img)
        lines = [((-HALF_WIDTH, BASELINE_Z), (HALF_WIDTH, BASELINE_Z)),
                 ((-HALF_WIDTH, FAR_BASELINE_Z), (HALF_WIDTH, FAR_BASELINE_Z)),
                 ((-HALF_WIDTH, BASELINE_Z), (-HALF_WIDTH, FAR_BASELINE_Z)),
                 ((HALF_WIDTH, BASELINE_Z), (HALF_WIDTH, FAR_BASELINE_Z)),
                 ((-HALF_WIDTH, NET_Z - SERVICE_LINE), (HALF_WIDTH, NET_Z - SERVICE_LINE)),
                 ((-HALF_WIDTH, NET_Z + SERVICE_LINE), (HALF_WIDTH, NET_Z + SERVICE_LINE)),
                 ((0, NET_Z - SERVICE_LINE), (0, NET_Z + SERVICE_LINE))]
        for (x1, z1), (x2, z2) in lines:
            cv2.line(img, P(x1, 0, z1), P(x2, 0, z2), WHITE, 2, cv2.LINE_AA)
        cv2.line(img, P(-HALF_WIDTH - 0.9, NET_HEIGHT, NET_Z), P(HALF_WIDTH + 0.9, NET_HEIGHT, NET_Z),
                 WHITE, 2, cv2.LINE_AA)

    def draw_ball(self, img):
        f = self.frame
        for i, p in enumerate(self.trail):
            c, s = f.project(*p)
            cv2.circle(img, ipt(c), max(1, int(2 + 3 * s * i / len(self.trail))), (0, 200, 170), -1)
        x, y, z = self.ball.pos
        (sx, sy), _ = f.project(x, 0, z)
        (bx, by), s = f.project(x, y, z)
        r = max(4, int(BALL_RADIUS * 3 * f.ppm * s))
        cv2.ellipse(img, ipt((sx, sy)), (r, max(2, r // 3)), 0, 0, 360, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, ipt((bx, by)), r, BALL_BGR, -1, cv2.LINE_AA)
        cv2.circle(img, ipt((bx, by)), r, (0, 120, 90), 2, cv2.LINE_AA)

    def draw_body(self, img, body):
        if body is None:
            if self.state != "MENU":
                text(img, "Step into view!", (img.shape[1] // 2, 120), 1.0, YELLOW, 2, center=True)
            return
        for a, b in SKELETON:
            pa, pb = body.points[a], body.points[b]
            if pa[2] > 0.5 and pb[2] > 0.5:
                cv2.line(img, ipt(pa), ipt(pb), (220, 220, 220), 3, cv2.LINE_AA)
        if body.racket:
            swinging = self.racket.in_swing
            col = RED if swinging else WHITE
            if self.state == "INCOMING" and self.show_ghost and self.ghost:
                g = self.frame.to_screen(*self.ghost)
                if np.hypot(body.racket[0] - g[0], body.racket[1] - g[1]) <= self.level.hit_radius * self.frame.ppm:
                    col = GREEN if not swinging else RED
            cv2.circle(img, ipt(body.racket), 18, col, 3, cv2.LINE_AA)
            cv2.circle(img, ipt(body.racket), 5, col, -1, cv2.LINE_AA)

    def draw_tags(self, img, tags):
        names = {TAG_START: "START"}
        names.update({k: LEVELS[v].name for k, v in TAG_LEVELS.items()})
        names.update({k: SURFACES[v].name for k, v in TAG_SURFACES.items()})
        for tid, c in tags.items():
            cv2.polylines(img, [np.int32(c)], True, CYAN, 3, cv2.LINE_AA)
            label = f"#{tid} {names.get(tid, '?')}"
            if tid == TAG_START and self.state in ("MENU", "OVER"):
                label += f" {min(self.start_seen, START_FRAMES)}/{START_FRAMES}"
            text(img, label, (c[:, 0].mean(), c[:, 1].min() - 18), 0.7, CYAN, 2, center=True)

    def draw_hud(self, img):
        h, w = img.shape[:2]
        now = time.monotonic()
        # top-left: settings
        shade(img, 0, 0, 420, 108)
        text(img, "VIRTUAL TENNIS", (14, 34), 0.95, YELLOW, 2)
        cv2.rectangle(img, (14, 50), (34, 70), self.surface.court_bgr, -1)
        text(img, f"{self.surface.name} court - {self.surface.blurb}", (42, 67), 0.5)
        text(img, f"Level {self.level_id}: {self.level.name} ({self.level.speed:.0f} m/s)", (14, 96), 0.6)
        # top-right: score
        shade(img, w - 270, 0, w, 108)
        text(img, "RALLY", (w - 255, 30), 0.55, (200, 200, 200))
        text(img, str(self.streak), (w - 255, 88), 1.9, WHITE, 3)
        text(img, "RECORD", (w - 120, 30), 0.55, (200, 200, 200))
        text(img, str(self.record), (w - 120, 88), 1.9, GREEN, 3)
        text(img, TOPIC, (w - 262, 104), 0.38, (180, 180, 180))

        # bottom-left: live IMU
        shade(img, 0, h - 112, 400, h)
        title = "IMU (keyboard sim)" if self.sim else "IMU - Double Motor 6065"
        text(img, title, (14, h - 88), 0.55, (200, 200, 200))
        self.meter(img, 14, h - 72, "pitch swing", abs(self.racket.live_rate) / rk.SWING_FULL,
                   rk.SWING_START / rk.SWING_FULL)
        self.meter(img, 14, h - 44, "yaw twist", abs(self.racket.live_twist) / rk.SPIN_FULL,
                   rk.SPIN_DEAD / rk.SPIN_FULL)
        if self.last_swing:
            s = self.last_swing
            text(img, f"last swing: power {s.power * 100:.0f}%  spin {s.spin:+.2f}"
                      + (f"  [{s.label}]" if s.label else ""), (14, h - 12), 0.45)

        # bottom-right: last shot
        if self.last_shot:
            s = self.last_shot
            shade(img, w - 400, h - 112, w, h)
            text(img, "LAST SHOT" + ("  - PERFECT" if s["perfect"] else ""), (w - 386, h - 88), 0.55,
                 GREEN if s["perfect"] else (200, 200, 200))
            text(img, f"{s['stroke']}  {s['label']}  ({s['how']})", (w - 386, h - 60), 0.6)
            text(img, f"{s['speed']:.0f} m/s   power {s['power'] * 100:.0f}%   spin {s['spin']:+.2f}",
                 (w - 386, h - 36), 0.55)
            text(img, f"contact height {s['height'] * 100:.0f} cm", (w - 386, h - 12), 0.55)

        # center: state banners
        if self.state == "MENU":
            shade(img, w // 2 - 330, h // 2 - 120, w // 2 + 330, h // 2 + 95, 0.6)
            text(img, "Show AprilTag 0 to START", (w // 2, h // 2 - 85), 1.0, YELLOW, 2, center=True)
            text(img, "Level:   1 Club     2 Varsity     3 Pro", (w // 2, h // 2 - 30), 0.7, center=True)
            text(img, "Court:   4 Clay     5 Grass     6 Hard", (w // 2, h // 2 + 10), 0.7, center=True)
            text(img, "Swing = pitch.  Twist (yaw) one way = topspin, other way = slice.",
                 (w // 2, h // 2 + 55), 0.55, (200, 200, 200), center=True)
        elif self.state == "COUNTDOWN":
            n = 3 - int(now - self.state_t)
            text(img, str(n) if n > 0 else "PLAY!", (w // 2, h // 2), 3.0, YELLOW, 5, center=True)
        if self.msg and now < self.msg[3]:
            title, detail, color, _ = self.msg
            text(img, title, (w // 2, 160), 1.5, color, 3, center=True)
            if detail:
                text(img, detail, (w // 2, 210), 0.7, WHITE, 1, center=True)

    @staticmethod
    def meter(img, x, y, label, frac, tick):
        bw = 250
        text(img, label, (x, y + 14), 0.5)
        x0 = x + 90
        cv2.rectangle(img, (x0, y), (x0 + bw, y + 16), (90, 90, 90), 1)
        frac = min(max(frac, 0.0), 1.0)
        cv2.rectangle(img, (x0, y), (x0 + int(bw * frac), y + 16), RED if frac >= tick else CYAN, -1)
        tx = x0 + int(bw * tick)
        cv2.line(img, (tx, y - 3), (tx, y + 19), WHITE, 2)


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Virtual Tennis with a LEGO Double Motor racket")
    ap.add_argument("--sim", action="store_true", help="no hub: swing with the keyboard")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="racket hand")
    ap.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--surface", choices=list(SURFACES), default="hard")
    ap.add_argument("--swing-axis", choices=["x", "y", "z"], default=rk.SWING_AXIS,
                    help="gyro axis of the swing (pitch)")
    ap.add_argument("--spin-axis", choices=["x", "y", "z"], default=rk.SPIN_AXIS,
                    help="gyro axis of the spin twist (yaw)")
    ap.add_argument("--model", default="models/pose_landmarker_full.task")
    ap.add_argument("--knn", default="strokes_model.json", help="trained stroke classifier, if present")
    ap.add_argument("--topic", default=TOPIC)
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--view", choices=["web", "cv"], default="web",
                    help="web = Three.js 3D court in the browser (default); cv = old OpenCV window")
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    ap.add_argument("--low-graphics", action="store_true", help="lighter 3D view for slower laptops")
    args = ap.parse_args()

    # 1. The racket: LEGO Education Double Motor on Connection Card 6065
    if args.sim:
        racket = rk.SimRacket()
    else:
        print(f"Connecting to Double Motor, card {DOUBLE_MOTOR_CARD_SERIAL} ...")
        dm = doubleMotor()
        dm.connect(card_serial=DOUBLE_MOTOR_CARD_SERIAL)
        dm.reset_heading()
        dm.movement_set_end_state(le.MOTOR_END_STATE_COAST)
        knn = None
        if os.path.exists(args.knn):
            from strokes import StrokeKNN
            knn = StrokeKNN.load(args.knn)
            print(f"Loaded stroke classifier {args.knn} ({len(knn.y)} swings)")
        racket = rk.Racket(dm, args.swing_axis, args.spin_axis, knn=knn)
        print("Racket connected.")

    # 2. Camera, pose and AprilTags
    from camlib import pick_camera
    cap, _ = pick_camera(1280, 720)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera returned no frames")
    h, w = frame.shape[:2]
    pose = PoseTracker(args.model, hand=args.hand)
    tag_reader = TagReader()
    board = Scoreboard(not args.no_mqtt, args.topic)

    game = Game(racket, board, BodyFrame(w, h), args.hand, args.level, args.surface, args.sim)
    view, win = None, "Virtual Tennis - Anne Jumper"
    if args.view == "web":
        from webview import WebView
        view = WebView(open_browser=not args.no_browser, low=args.low_graphics)
    else:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def handle_key(k):
        """Returns False to quit."""
        if k in (ord("q"), 27):
            return False
        if args.sim and racket.key(k):
            return True
        if k in (ord("1"), ord("2"), ord("3")):
            game.choose(level=k - ord("0"))
        elif k in (ord("c"), ord("g"), ord("h")):
            game.choose(surface={"c": "clay", "g": "grass", "h": "hard"}[chr(k)])
        elif k == ord("s"):
            game.start()
        return True

    last = last_cam = time.monotonic()
    running = True
    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.monotonic()
            dt = min(t - last, 0.1)
            last = t

            body = pose.process(frame, t)
            tags = tag_reader.detect(frame) if game.state in ("MENU", "OVER") else {}
            game.handle_tags(tags)
            game.update(t, dt, body, racket.poll_swings())

            disp = cv2.flip(frame, 1)
            if view:
                # Main view = Three.js court; the webcam (with pose + tags) is the corner inset
                view.send_state(game.snapshot())
                if t - last_cam > 1 / 20:
                    last_cam = t
                    game.draw_body(disp, body)
                    game.draw_tags(disp, tags)
                    small = cv2.resize(disp, (480, int(480 * h / w)))
                    view.send_camera(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes())
                keys = [ord(" ") if k == "space" else ord(k[0]) for k in view.poll_keys()]
                running = all(handle_key(k) for k in keys)
            else:
                game.draw(disp, body, tags)
                cv2.imshow(win, disp)
                running = handle_key(cv2.waitKey(1) & 0xFF)
    finally:
        racket.close()
        board.close()
        cap.release()
        if view:
            view.close()
        cv2.destroyAllWindows()
        print(f"Final record: {game.record} continuous hits")


if __name__ == "__main__":
    main()
