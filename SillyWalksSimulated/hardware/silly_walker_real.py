"""Run a trained Walk-Real-* policy on the real Silly Walker (LEGO Education Double Motor, over Bluetooth).

The laptop runs the policy and talks to the Double Motor with the official `legoeducation` API, like
straight-walk/walk_rl.py. Every 40 ms (the rate the policy was trained at) it reads the IMU and leg encoders, turns them
into the 9 inputs the policy saw in simulation, runs the policy, and sends the two leg speeds back.

    python silly_walker_real.py verify laminate     # no robot: check the exported policy gives the cluster's answers
    python silly_walker_real.py calibrate           # once per robot: sensor axes, units, leg directions, leg home pose
    python silly_walker_real.py testdrive           # scripted straight walk (same as drive.py's), for sim vs real
    python silly_walker_real.py run laminate        # walk with the trained policy (default 10 s; add seconds to change)
    python silly_walker_real.py run laminate --fake # no robot: the same loop against the MuJoCo model on this laptop

walk_default.py, walk_laminate.py and walk_carpet.py next to this file are shortcuts for `run <floor>`.
Every run is logged to real_runs/ (one row per control step) with a one-line summary in real_runs/summary.csv.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CARD_SERIAL = "PUT_YOUR_CARD_NUMBER_HERE"   # <-- the number on your Double Motor's Connection Card
POLICY_DIR = HERE / "policies"              # policy_<floor>.onnx / .json / _check.npz from export_policy.py
CALIBRATION = HERE / "calibration.json"     # written by `calibrate`
RUNS = HERE / "real_runs"
NOTIFY_MS = 20                              # sensor updates from the motor every 20 ms (policy runs every 40 ms)
STILL_S = 1.5                               # seconds the robot sits still before walking (gyro bias is measured here)
SIM_RESULTS = {                             # Walk-Real-* simulation results for the same 10 s, filled in from make_video.py
    "default": None, "laminate": None, "carpet": None,
}
TESTDRIVE = (8.0, 3.0)                      # rad/s, seconds: drive.py's straight phase (sim: about 0.23 m)


# ---- the robot ---------------------------------------------------------------------------------------------------


class RealRobot:
    """The Double Motor through `legoeducation`. Raw readings only; Calibration turns them into robot units."""

    def __init__(self):
        import legoeducation as le
        self.le = le
        self.dev = le.DoubleMotor()

    def connect(self):
        if not CARD_SERIAL.isdigit():
            sys.exit(f"Set CARD_SERIAL at the top of {Path(__file__).name} to the number on your Connection Card.")
        print(f"Connecting to Double Motor {CARD_SERIAL}...")
        self.dev.connect(card_serial=CARD_SERIAL, device_notification_delay=NOTIFY_MS)
        if not self.dev.connected:
            sys.exit("Could not connect. Is the Double Motor on, and is CARD_SERIAL right?")
        self.dev.movement_set_end_state(self.le.MOTOR_END_STATE_BRAKE)
        self.dev.beep()

    def gyro(self):        # raw gyroscope x, y, z
        i = self.dev.imu_device
        return np.array([i.gyroscopeX, i.gyroscopeY, i.gyroscopeZ], dtype=float)

    def yaw(self):         # raw IMU yaw
        return float(self.dev.imu_device.yaw)

    def positions(self):   # raw leg encoder angles in degrees, [motor 0, motor 1]
        return np.array([self.dev.motor[0].position, self.dev.motor[1].position], dtype=float)

    def abs_positions(self):
        return np.array([self.dev.motor[0].absolutePosition, self.dev.motor[1].absolutePosition], dtype=float)

    def set_speeds(self, pct0, pct1):   # % speed for motor 0 (left side) and motor 1 (right side)
        self.dev.movement_move_tank(int(pct0), int(pct1), blocking=False)

    def move_leg_to(self, motor, abs_deg):
        self.dev.motor_run_to_absolute_position(int(abs_deg) % 360, motor=motor, speed=20, blocking=True)

    def stop(self):
        self.dev.movement_stop()

    def disconnect(self):
        try:
            self.stop()
        finally:
            self.dev.disconnect()


class FakeRobot:
    """The MuJoCo model on this laptop pretending to be the Double Motor, so the whole loop can be tested without
    hardware. Its 'raw' readings use made-up LEGO-like conventions that FAKE_CALIBRATION undoes."""

    def __init__(self, floor):
        import mujoco
        self.mj = mujoco
        onshape = HERE.parent / "09_onshape_robot" / "onshape"
        friction, solref = {"default": (1.0, 0.02), "laminate": (0.3, 0.02), "carpet": (0.6, 0.03)}[floor]
        spec = mujoco.MjSpec.from_file(str(onshape / "scene.xml"))
        for a in list(spec.actuators):
            spec.delete(a)
        for j in ("left_wheel", "right_wheel"):   # like mjlab's velocity motor: 0.05 N*m per rad/s, 0.2 N*m cap
            a = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_velocity(kv=0.05)
            a.forcerange, a.forcelimited = [-0.2, 0.2], 1
        spec.body("chassis").add_freejoint()
        spec.option.timestep = 0.005
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        for g in spec.geoms:
            if (g.contype or g.conaffinity) and g.type != mujoco.mjtGeom.mjGEOM_PLANE:
                g.priority, g.friction[0], g.solref = 1, friction, (solref, 1.0)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        low = min((self.m.mesh_vert[self.m.mesh_vertadr[self.m.geom_dataid[g]]:
                                    self.m.mesh_vertadr[self.m.geom_dataid[g]] + self.m.mesh_vertnum[self.m.geom_dataid[g]]]
                   @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g])[:, 2].min()
                  for g in range(self.m.ngeom) if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and self.m.geom_contype[g])
        self.d.qpos[2] -= low
        rng = np.random.default_rng()
        yaw0 = rng.uniform(-0.4, 0.4)              # start a little off-axis, like the simulation
        # The model walks toward -y; "forward" in the fake world is -y, rotated by yaw0.
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, [0, 0, 1], yaw0)
        self.d.qpos[3:7] = q
        mujoco.mj_forward(self.m, self.d)
        self.start_xy = self.d.qpos[:2].copy()
        self.yaw0 = yaw0

    def connect(self):
        print("Fake robot: MuJoCo model on this laptop (no Bluetooth).")

    def _advance(self, seconds):
        end = self.d.time + seconds
        while self.d.time < end:
            self.mj.mj_step(self.m, self.d)

    def _body_rates(self):    # chassis angular velocity in the walking frame (x fwd, y left, z up), rad/s
        w_local = self.d.qvel[3:6]                 # free joint: body-frame angular velocity of the chassis
        # Onshape chassis frame: walks toward -y, left = +x. Walking frame: x_w = -y_c, y_w = +x_c.
        return np.array([-w_local[1], w_local[0], w_local[2]])

    def gyro(self):           # fake raw: deg/s in a scrambled LEGO-like axis order [z, -x, y]
        b = np.degrees(self._body_rates())
        return np.array([b[2], -b[0], b[1]])

    def yaw(self):            # fake raw: tenths of a degree, positive = turned right
        w, x, y, z = self.d.qpos[3:7]
        return -10 * math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def positions(self):      # fake raw: degrees, motor 0 = left leg, positive = forward
        return np.degrees(self.d.qpos[7:9])

    def abs_positions(self):
        return self.positions() % 360

    def set_speeds(self, pct0, pct1):   # fake: 100% = 15 rad/s
        self.d.ctrl[:] = [pct0 * 0.15, pct1 * 0.15]

    def move_leg_to(self, motor, abs_deg):
        pass

    def stop(self):
        self.d.ctrl[:] = 0

    def disconnect(self):
        pass

    def sleep(self, seconds):  # time passes in the simulation instead of on the wall clock
        self._advance(seconds)

    def true_position(self):   # where the fake robot really is, in its walking frame (forward, left), metres
        dx, dy = self.d.qpos[:2] - self.start_xy
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        fwd = np.array([s, -c])                    # -y rotated by yaw0
        left = np.array([c, s])
        return float(np.dot([dx, dy], fwd)), float(np.dot([dx, dy], left))


FAKE_CALIBRATION = {   # what `calibrate` would measure on FakeRobot
    "gyro_matrix": [[0, -1, 0], [0, 0, 1], [1, 0, 0]],     # body = M @ raw  (raw = [z, -x, y])
    "gyro_rad_per_unit": math.radians(1.0),
    "heading_source": "yaw",
    "yaw_rad_per_unit": -math.radians(0.1),                 # tenths of a degree, positive = right
    "yaw_full_turn_units": 3600.0,
    "leg_motor": [0, 1],                                    # policy [left, right] -> motor index
    "leg_sign": [1, 1],                                     # +1: encoder grows when that leg walks forward
    "cmd_sign": [1, 1],                                     # +1: positive % walks forward
    "rad_s_per_pct": [0.15, 0.15],
    "home_abs_deg": None,
}


def tick(robot, seconds):
    if isinstance(robot, FakeRobot):
        robot.sleep(seconds)
    else:
        time.sleep(seconds)


# ---- calibration: raw readings -> the units and axes the policy was trained with ---------------------------------


class Calibration:
    def __init__(self, c):
        self.c = c
        self.M = np.array(c["gyro_matrix"], dtype=float)

    def body_rates(self, raw_gyro, bias):          # rad/s, x forward, y left, z up
        return self.M @ (raw_gyro - bias) * self.c["gyro_rad_per_unit"]

    def heading_delta(self, raw_now, raw_start):   # rad, positive = turned left, wrapped to [-pi, pi)
        full = self.c["yaw_full_turn_units"]
        d = (raw_now - raw_start + full / 2) % full - full / 2
        return d * self.c["yaw_rad_per_unit"]

    def leg_speeds(self, pos_now, pos_before, dt):  # rad/s, [left, right], positive = forward
        out = []
        for side in (0, 1):
            m = self.c["leg_motor"][side]
            out.append(self.c["leg_sign"][side] * math.radians(pos_now[m] - pos_before[m]) / dt)
        return np.array(out)

    def speed_command(self, leg_rad_s):             # policy [left, right] rad/s -> % for [motor 0, motor 1]
        pct = [0, 0]
        for side in (0, 1):
            m = self.c["leg_motor"][side]
            p = self.c["cmd_sign"][side] * leg_rad_s[side] / self.c["rad_s_per_pct"][side]
            pct[m] = int(max(-100, min(100, round(p))))
        return pct


def load_calibration(fake):
    if fake:
        return Calibration(FAKE_CALIBRATION)
    if not CALIBRATION.exists():
        sys.exit("No calibration.json yet: run `python silly_walker_real.py calibrate` first.")
    return Calibration(json.loads(CALIBRATION.read_text()))


def record(robot, seconds, every=0.02):
    """Sample the raw gyro, yaw and leg positions for `seconds`."""
    rows, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        rows.append((time.perf_counter() - t0, robot.gyro(), robot.yaw(), robot.positions()))
        time.sleep(every)
    return rows


def integrate(rows, bias):
    total = np.zeros(3)
    for (t0, g0, _, _), (t1, _, _, _) in zip(rows, rows[1:]):
        total += (g0 - bias) * (t1 - t0)
    return total


def calibrate():
    robot = RealRobot()
    robot.connect()
    c = {}
    try:
        input("\n1/6  Put the robot flat on the floor and don't touch it. Press Enter (measures gyro bias, 2 s)...")
        still = record(robot, 2.0)
        bias = np.mean([g for _, g, _, _ in still], axis=0)
        print(f"     gyro bias {bias}")

        motions = [  # (instruction, body axis index, sign of the rotation about that axis)
            ("Turn it EXACTLY a quarter turn to the LEFT (counterclockwise seen from above) and stop", 2, +1),
            ("Lift its FRONT (the end the legs point to) about 45 degrees, then put it back down", 1, -1),
            ("Lift its LEFT side about 45 degrees, then put it back down", 0, -1),
        ]
        M = np.zeros((3, 3))
        quarter = None
        for k, (text, axis, sign) in enumerate(motions, start=2):
            input(f"\n{k}/6  Set it down pointing the same way. {text} within 3 s after you press Enter...")
            rows = record(robot, 3.0)
            angle = integrate(rows, bias)
            if axis == 2:   # for lift/put-back motions the net angle is ~0, so use the biggest excursion instead
                raw_axis = int(np.argmax(np.abs(angle)))
                value = angle[raw_axis]
                quarter = (rows, value)
            else:
                cum = np.cumsum([(g - bias) * (t1 - t0) for (t0, g, _, _), (t1, _, _, _) in zip(rows, rows[1:])], axis=0)
                peak = cum[np.argmax(np.abs(cum).max(axis=1))]
                raw_axis = int(np.argmax(np.abs(peak)))
                value = peak[raw_axis]
            M[axis, raw_axis] = sign * np.sign(value)
            print(f"     -> body axis {'xyz'[axis]} is raw gyro axis {'XYZ'[raw_axis]} (sign {int(M[axis, raw_axis]):+d})")
        if sorted(np.argmax(np.abs(M), axis=1)) != [0, 1, 2]:
            sys.exit("Two motions used the same gyro axis. Run calibrate again with bigger, cleaner motions.")
        rows, gyro_quarter = quarter
        c["gyro_matrix"] = M.tolist()
        c["gyro_rad_per_unit"] = (math.pi / 2) / abs(gyro_quarter)
        yaw_change = rows[-1][2] - rows[0][2]
        if abs(yaw_change) > 20:            # IMU yaw follows the turn: use it (degrees or tenths of a degree)
            c["heading_source"] = "yaw"
            c["yaw_rad_per_unit"] = (math.pi / 2) / yaw_change
            c["yaw_full_turn_units"] = 3600.0 if abs(yaw_change) > 300 else 360.0
            print(f"     IMU yaw changed by {yaw_change:.0f} for the quarter turn: using IMU yaw for the heading")
        else:                               # yaw stuck (motor mounted on its side): integrate the gyro instead
            c["heading_source"] = "gyro"
            c["yaw_rad_per_unit"], c["yaw_full_turn_units"] = 1.0, 2 * math.pi
            print(f"     IMU yaw barely changed ({yaw_change:.0f}): the heading will come from the gyroscope instead")

        input("\n5/6  Hold the robot in the AIR so the legs spin freely. Press Enter (legs run at 40% for 2 s)...")
        p0 = robot.positions()
        robot.set_speeds(40, 40)
        time.sleep(2.0)
        p1 = robot.positions()
        robot.stop()
        rate = np.radians(p1 - p0) / 2.0 / 40       # rad/s per % (signed, per motor)
        print(f"     motor 0: {abs(rate[0]) * 100:.1f} rad/s at 100%,  motor 1: {abs(rate[1]) * 100:.1f} rad/s at 100%")
        if min(abs(rate)) * 100 < 10:
            print("     NOTE: the policy may ask for up to 10 rad/s; this motor tops out lower, so fast commands are capped.")
        answer = input("\n     Put it on the floor. Press Enter to walk at 40% for 2 s, then say which way it went...")
        robot.set_speeds(40, 40)
        time.sleep(2.0)
        robot.stop()
        forward = input("     Did it walk toward the end the legs point to (its front)? [y/n] ").strip().lower().startswith("y")
        # If positive % walks backward, the robot's front is the other way: forward = negative %, and the policy's
        # left leg is on motor 1's side.
        c["leg_motor"] = [0, 1] if forward else [1, 0]
        c["cmd_sign"] = [1, 1] if forward else [-1, -1]
        c["leg_sign"] = [int(np.sign(rate[m])) * c["cmd_sign"][s] for s, m in enumerate(c["leg_motor"])]
        c["rad_s_per_pct"] = [abs(rate[m]) for m in c["leg_motor"]]

        input("\n6/6  Set both legs by hand to match your Onshape assembly, then press Enter (saves the leg home pose)...")
        c["home_abs_deg"] = robot.abs_positions().tolist()
        CALIBRATION.write_text(json.dumps(c, indent=2))
        print(f"\nSaved {CALIBRATION.name}. Next: `python {Path(__file__).name} testdrive`, then `run <floor>`.")
    finally:
        robot.disconnect()


# ---- running -----------------------------------------------------------------------------------------------------


def load_policy(floor):
    import onnxruntime as ort
    onnx_file, meta_file = POLICY_DIR / f"policy_{floor}.onnx", POLICY_DIR / f"policy_{floor}.json"
    if not onnx_file.exists():
        sys.exit(f"No {onnx_file.relative_to(HERE)}: export it with export_policy.py and copy it into policies/.")
    meta = json.loads(meta_file.read_text())
    names = [t["name"] for t in meta["inputs"]]
    if names != ["ang_vel", "heading", "wheel_vel", "last_action"]:
        sys.exit(f"{meta_file.name} lists inputs {names}; this script builds ang_vel, heading, wheel_vel, last_action.")
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    in_name, out_name = session.get_inputs()[0].name, session.get_outputs()[0].name

    def policy(obs):
        return session.run([out_name], {in_name: obs.astype(np.float32)[None, :]})[0][0]

    return policy, meta


def verify(floor):
    policy, meta = load_policy(floor)
    check = np.load(POLICY_DIR / f"policy_{floor}_check.npz")
    worst = max(float(np.abs(policy(o) - a).max()) for o, a in zip(check["obs"], check["action"]))
    print(f"{meta['task']}: largest difference from the cluster's answers over {len(check['obs'])} inputs: {worst:.2e}")
    print("OK: the laptop runs the same policy." if worst < 1e-4 else "MISMATCH: re-export the policy.")


def home_legs(robot, cal):
    home = cal.c.get("home_abs_deg")
    if home is None or isinstance(robot, FakeRobot):
        return
    print("Moving the legs to the home pose saved by calibrate...")
    for m in (0, 1):
        robot.move_leg_to(m, home[m])


def run(floor, seconds=10.0, fake=False):
    policy, meta = load_policy(floor)
    dt = meta["control_dt_s"]
    max_leg = meta["max_leg_speed_rad_s"]
    cal = load_calibration(fake)
    robot = FakeRobot(floor) if fake else RealRobot()
    robot.connect()
    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = RUNS / f"{'fake' if fake else 'real'}_{floor}_{stamp}.csv"
    try:
        home_legs(robot, cal)
        if not fake:
            input(f"\nPolicy {meta['task']} ({meta['checkpoint']}). Put the robot on the {floor} floor pointing where it "
                  f"should walk, legs in the Onshape pose. Mark its start. Press Enter...")
        # Sit still: measure the gyro bias and the starting heading (the policy's "straight ahead").
        samples = []
        for _ in range(int(STILL_S / 0.02)):
            samples.append(robot.gyro())
            tick(robot, 0.02)
        bias = np.mean(samples, axis=0)
        yaw_start = robot.yaw()
        heading_gyro = 0.0
        last_action = np.zeros(2)
        pos_before = robot.positions()
        rows = []
        print("  second   heading (deg)   leg speeds L/R (rad/s)   commands L/R (%)")
        steps = int(round(seconds / dt))
        t_next = time.perf_counter()
        for step in range(steps):
            rates = cal.body_rates(robot.gyro(), bias)
            if cal.c["heading_source"] == "yaw":
                heading = cal.heading_delta(robot.yaw(), yaw_start)
            else:
                heading_gyro += rates[2] * dt
                heading = heading_gyro
            pos_now = robot.positions()
            legs = cal.leg_speeds(pos_now, pos_before, dt)
            pos_before = pos_now
            obs = np.concatenate([rates, [math.sin(heading), math.cos(heading)], legs, last_action])
            action = np.asarray(policy(obs), dtype=float)
            pct = cal.speed_command(action * max_leg)
            robot.set_speeds(*pct)
            last_action = action
            rows.append([round(step * dt, 3), *np.round(rates, 4), round(math.degrees(heading), 2), *np.round(legs, 3),
                         *np.round(action, 4), *pct])
            if (step + 1) % round(1 / dt) == 0:
                print(f"  {(step + 1) * dt:5.0f}   {math.degrees(heading):11.1f}      {legs[0]:7.2f} {legs[1]:7.2f}"
                      f"            {pct[0]:4d} {pct[1]:4d}")
            if fake:
                robot.sleep(dt)
            else:   # keep a steady 25 Hz; if a step ran long, carry on without trying to catch up
                t_next += dt
                time.sleep(max(0.0, t_next - time.perf_counter()))
                if time.perf_counter() - t_next > dt:
                    t_next = time.perf_counter()
        robot.stop()
    finally:
        robot.disconnect()

    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "rate_x", "rate_y", "rate_z", "heading_deg", "leg_left_rad_s", "leg_right_rad_s",
                    "action_left", "action_right", "cmd_motor0_pct", "cmd_motor1_pct"])
        w.writerows(rows)
    headings = [r[4] for r in rows]
    if fake:
        fwd, side = robot.true_position()
        print(f"\nFake run: walked {fwd:.2f} m forward, {side * 100:+.1f} cm sideways (true position from MuJoCo).")
    else:
        fwd = float(input("\nMeasure from the start mark: how far FORWARD did it go (cm)? ")) / 100
        side = float(input("How far SIDEWAYS (cm, + = to its left, - = to its right)? ")) / 100
    summary = RUNS / "summary.csv"
    new = not summary.exists()
    with summary.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "mode", "floor", "seconds", "forward_m", "sideways_m", "heading_min_deg", "heading_max_deg", "log"])
        w.writerow([stamp, "fake" if fake else "real", floor, seconds, round(fwd, 3), round(side, 3),
                    round(min(headings), 1), round(max(headings), 1), log_path.name])
    print(f"Heading stayed between {min(headings):+.1f} and {max(headings):+.1f} degrees.")
    sim = SIM_RESULTS.get(floor)
    if sim:
        print(f"Simulation (Walk-Real-{floor.capitalize()}, 10 s): {sim}")
    print(f"Logged to {log_path.relative_to(HERE)} and real_runs/summary.csv")


def testdrive():
    """drive.py's straight phase on the real robot: both legs at 8 rad/s for 3 s. Simulation: about 0.23 m."""
    cal = load_calibration(False)
    robot = RealRobot()
    robot.connect()
    try:
        home_legs(robot, cal)
        input("\nPut the robot on the floor, legs in the Onshape pose, and mark its start. Press Enter...")
        speed, secs = TESTDRIVE
        yaw0 = robot.yaw()
        robot.set_speeds(*cal.speed_command(np.array([speed, speed])))
        time.sleep(secs)
        robot.stop()
        turn = math.degrees(cal.heading_delta(robot.yaw(), yaw0)) if cal.c["heading_source"] == "yaw" else float("nan")
    finally:
        robot.disconnect()
    fwd = float(input("How far forward did it go (cm)? "))
    print(f"Real: {fwd:.1f} cm in {secs:.0f} s, heading changed {turn:+.1f} deg.   Simulation (drive.py): about 23 cm.")


def main(argv):
    if not argv or argv[0] not in ("verify", "calibrate", "testdrive", "run"):
        sys.exit(__doc__)
    cmd, rest = argv[0], argv[1:]
    fake = "--fake" in rest
    rest = [a for a in rest if a != "--fake"]
    if cmd in ("verify", "run"):
        if not rest or rest[0] not in ("default", "laminate", "carpet"):
            sys.exit(f"usage: {cmd} default|laminate|carpet" + (" [seconds] [--fake]" if cmd == "run" else ""))
    if cmd == "verify":
        verify(rest[0])
    elif cmd == "calibrate":
        calibrate()
    elif cmd == "testdrive":
        testdrive()
    else:
        run(rest[0], float(rest[1]) if len(rest) > 1 else 10.0, fake)


if __name__ == "__main__":
    main(sys.argv[1:])
