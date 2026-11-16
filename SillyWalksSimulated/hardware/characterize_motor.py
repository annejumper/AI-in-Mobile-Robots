"""Motor characterization for the Silly Walker's Double Motor, and a motor model for mjlab.

Spins the legs through a fixed program of commands, records the leg encoders as fast as the motor reports them, then
fits the MuJoCo/mjlab motor model to the recording. The model is what drive_task.py uses for the Walk-Measured-*
tasks.

    python characterize_motor.py air              # robot propped up, legs spin freely (about 85 s)
    python characterize_motor.py floor laminate   # robot walks on a floor at 20..100% (about 21 s; default/carpet too)
    python characterize_motor.py fit              # fit the model to the newest air run + newest run on each floor
    python characterize_motor.py air --fake       # no robot: a MuJoCo motor with hidden "true" values (FAKE_TRUTH)
    python characterize_motor.py fit --fake       # fit the fake runs; should get back close to FAKE_TRUTH

What it measures (air run):
  speed sweep    +-5..100% for 1.2 s each: steady leg speed per %, top speed, deadband, each motor and direction
  steps          0 -> 20/50/80/100% -> 0, three times each: delay to first movement, how fast speed builds up
  reversals      +60% -> -60%
  random         10 s of random commands (both legs independent), like a policy: the main data for the fit
Floor runs add what walking costs the motor: at the same command the legs turn slower under load.

What it fits (the same parameters mjlab's BuiltinPdActuatorCfg takes):
  damping        velocity gain kd (N*m per rad/s of speed error)
  effort_limit   torque limit (N*m)
  armature       motor and gearbox inertia seen at the leg (kg*m^2)
  frictionloss   dry friction at the leg (N*m), from the speed the motor loses in the air
  delay          command delay in 5 ms physics steps (Bluetooth + motor firmware)
by replaying the recorded commands in the MuJoCo model and matching the leg speeds. It writes motor_model.json (and
motor_fit.png), and prints how far off drive_task.py's current guesses were.

Runs go to motor_runs/ (one JSON each). Fake runs and fake fits are named fake_* and motor_model_fake.json, so they
never reach drive_task.py.
"""
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "motor_runs"
ONSHAPE = HERE.parent / "09_onshape_robot" / "onshape"
CALIBRATION = HERE / "calibration.json"
LEGS = ("left_wheel", "right_wheel")   # joint names in robot.xml
DT = 0.005                             # physics step (same as drive_task.py)
NOTIFY_MS = 20                         # ask the motor for readings every 20 ms (the fastest that is reliable)
SENSOR_LAG_S = 0.015                   # guess: a reading is about this old when it arrives (half a notification
                                       # period + Bluetooth). The measured delay minus this is the command delay.
GRID_S = 0.02                          # leg positions are compared on this time grid...
SPEED_WINDOW = 5                       # ...and leg speeds over 5 grid steps (0.1 s), which hides 1-degree encoder steps
FLOORS = {"default": (1.0, 0.02), "laminate": (0.3, 0.02), "carpet": (0.6, 0.03)}   # friction, solref (as drive_task)
# drive_task.py's motor before measuring: velocity gain and torque limit there, armature and friction from robot.xml
CURRENT_SIM = {"damping": 0.05, "effort_limit": 0.2, "armature": 0.005, "frictionloss": 0.1, "delay_steps": 0}
BOUNDS = {"damping": (1e-3, 1.0), "effort_limit": (5e-3, 1.0), "armature": (1e-6, 0.05), "delay_steps": (0, 40)}
# The fake motor's hidden values. `fit --fake` should come close to these (gain in rad/s per %, top speed in rad/s).
FAKE_TRUTH = {"damping": 0.02, "effort_limit": 0.09, "armature": 0.0012, "frictionloss": 0.004, "delay_steps": 9,
              "gain": [0.11, 0.105], "top": [9.5, 9.0], "encoder_sign": [1, -1]}


# ---- command programs ---------------------------------------------------------------------------------------------


def program_air(seed=193):
    """(seconds, % motor 0, % motor 1) segments. Same every time (fixed seed), so runs can be compared."""
    segs = [(0.5, 0, 0)]
    for p in (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100):        # speed sweep
        for s in (1, -1):
            segs += [(1.2, s * p, s * p), (0.5, 0, 0)]
    for p in (20, 50, 80, 100):                                        # steps from rest
        for _ in range(3):
            segs += [(1.0, p, p), (0.8, 0, 0)]
    for _ in range(3):                                                 # reversals
        segs += [(1.0, 60, 60), (1.0, -60, -60), (0.6, 0, 0)]
    rng = np.random.default_rng(seed)                                  # random, like a policy
    t = 0.0
    while t < 10.0:
        d = float(rng.uniform(0.12, 0.4))
        segs.append((d, int(rng.integers(-100, 101)), int(rng.integers(-100, 101))))
        t += d
    segs.append((1.0, 0, 0))
    return segs


def program_floor():
    """Walking straight at 20..100% ("forward" %, turned into motor % with calibration.json's directions)."""
    segs = [(0.5, 0, 0)]
    for p in (20, 40, 60, 80, 100):
        segs += [(3.0, p, p), (1.0, 0, 0)]
    return segs


def leg_mapping():
    """Which motor drives which leg and which % sign walks forward, from calibration.json (or the usual guess)."""
    if CALIBRATION.exists():
        c = json.loads(CALIBRATION.read_text())
        return {"leg_motor": c["leg_motor"], "cmd_sign": c["cmd_sign"], "from": "calibration.json"}
    return {"leg_motor": [0, 1], "cmd_sign": [1, 1], "from": "guess (no calibration.json)"}


# ---- the simulated motor (used both to fake a robot and to fit the model) -----------------------------------------


class SimMotor:
    """The Silly Walker in MuJoCo with mjlab's velocity motor: torque = damping x (target - speed), clamped to
    effort_limit, plus armature and frictionloss on each leg joint, and the command arriving delay_steps late.

    Targets are given per MOTOR in "motor units": positive = the way a positive % turns that motor. The mapping says
    which leg each motor drives and whether positive % walks forward."""

    _cache = {}

    def __init__(self, mode, floor, mapping):
        import mujoco
        self.mj = mujoco
        key = (mode, floor)
        if key not in SimMotor._cache:
            SimMotor._cache[key] = self._compile(mode, floor)
        self.m = SimMotor._cache[key]
        self.d = mujoco.MjData(self.m)
        self.joint_of_motor = [0, 0]
        self.sign = [1, 1]
        for side, motor in enumerate(mapping["leg_motor"]):
            self.joint_of_motor[motor] = side
            self.sign[motor] = mapping["cmd_sign"][side]
        self.qadr = [self.m.jnt_qposadr[self.m.joint(j).id] for j in LEGS]
        self.vadr = [self.m.jnt_dofadr[self.m.joint(j).id] for j in LEGS]
        if mode == "floor":
            self._put_on_floor()
        mujoco.mj_forward(self.m, self.d)

    @staticmethod
    def _compile(mode, floor):
        import mujoco
        spec = mujoco.MjSpec.from_file(str(ONSHAPE / "scene.xml"))
        for a in list(spec.actuators):
            spec.delete(a)
        for j in LEGS:
            a = spec.add_actuator(name=j, target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.set_to_velocity(kv=0.05)
            jt = spec.joint(j)
            jt.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            jt.actfrcrange = [-0.2, 0.2]
        chassis = spec.body("chassis")
        if mode == "floor":
            chassis.add_freejoint()
            friction, solref = FLOORS[floor]
            for g in spec.geoms:
                if (g.contype or g.conaffinity) and g.type != mujoco.mjtGeom.mjGEOM_PLANE:
                    g.priority, g.friction[0], g.solref = 1, friction, (solref, 1.0)
        else:
            chassis.pos = [0.0, 0.0, 0.15]   # held up in the air, upright
            # A chassis fixed to the world counts as the world, and MuJoCo then stops ignoring contacts between it and
            # its own hubs (the hub hulls overlap the motor housing). Exclude them, as the free-floating robot does.
            for hub in ("left_hub", "right_hub"):
                spec.add_exclude(bodyname1="chassis", bodyname2=hub)
        spec.option.timestep = DT
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        return spec.compile()

    def _put_on_floor(self):
        mj, m, d = self.mj, self.m, self.d
        mj.mj_forward(m, d)
        low = min((m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
                   @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()
                  for g in range(m.ngeom) if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH and m.geom_contype[g])
        d.qpos[2] -= low

    def set_params(self, p):
        m = self.m
        for j in LEGS:
            jid = m.joint(j).id
            m.dof_armature[m.jnt_dofadr[jid]] = p["armature"]
            m.dof_frictionloss[m.jnt_dofadr[jid]] = p["frictionloss"]
            m.jnt_actfrcrange[jid] = [-p["effort_limit"], p["effort_limit"]]
            aid = m.actuator(j).id
            m.actuator_gainprm[aid, 0] = p["damping"]
            m.actuator_biasprm[aid, 2] = -p["damping"]
        self.buf = deque([np.zeros(2)] * (int(p["delay_steps"]) + 1))
        self.target = np.zeros(2)

    def set_targets(self, motor_rad_s):
        self.target = np.asarray(motor_rad_s, dtype=float)

    def step(self):
        self.buf.append(self.target.copy())
        applied = self.buf.popleft()
        for motor in (0, 1):
            self.d.ctrl[self.joint_of_motor[motor]] = self.sign[motor] * applied[motor]
        self.mj.mj_step(self.m, self.d)

    def motor_positions(self):   # rad, motor units
        return np.array([self.sign[mo] * self.d.qpos[self.qadr[self.joint_of_motor[mo]]] for mo in (0, 1)])


def pct_to_target(pct, gain, top):
    """% -> the speed the motor aims for, rad/s (motor units): linear in %, capped at the top speed."""
    return np.clip(np.asarray(gain) * np.asarray(pct, dtype=float), -np.asarray(top), np.asarray(top))


def simulate(params, run, gain, top):
    """Replay a recorded run's commands in MuJoCo. Returns times and motor positions (rad, motor units)."""
    sim = SimMotor(run["mode"], run.get("floor") or "default", run["mapping"])
    sim.set_params(params)
    cmds = run["commands"]
    n = int(run["duration_s"] / DT) + 1
    out = np.zeros((n, 2))
    k = 0
    for i in range(n):
        t = i * DT
        while k < len(cmds) and cmds[k][0] <= t:
            sim.set_targets(pct_to_target(cmds[k][1:3], gain, top))
            k += 1
        sim.step()
        out[i] = sim.motor_positions()
    return np.arange(n) * DT, out


# ---- the two kinds of motor the recorder can talk to --------------------------------------------------------------


class RealRig:
    """The Double Motor over Bluetooth. A background thread polls the encoders every 1 ms and keeps every change, with
    the laptop time it arrived."""

    def __init__(self):
        sys.path.insert(0, str(HERE))
        from silly_walker_real import RealRobot   # connection and CARD_SERIAL live there
        self.robot = RealRobot()
        self.robot.connect()
        self.samples, self._running = [], False

    def now(self):
        return time.perf_counter()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        motors, last = self.robot.dev.motor, None
        while self._running:
            s = (motors[0].position, motors[1].position, motors[0].speed, motors[1].speed)
            if s != last:
                self.samples.append((time.perf_counter(), *s))
                last = s
            time.sleep(0.001)

    def command(self, p0, p1):
        self.robot.set_speeds(p0, p1)

    def wait_until(self, t):
        while (left := t - time.perf_counter()) > 0:
            time.sleep(min(left, 0.005))

    def finish(self):
        self._running = False
        self._thread.join()
        return self.samples

    def home_legs(self):
        home = json.loads(CALIBRATION.read_text()).get("home_abs_deg") if CALIBRATION.exists() else None
        if home:
            print("Moving the legs to the home pose saved by calibrate...")
            for m in (0, 1):
                self.robot.move_leg_to(m, home[m])

    def close(self):
        self.robot.disconnect()


class FakeRig:
    """MuJoCo with FAKE_TRUTH's motor, reporting like the real one: whole degrees, every 20 ms (+-3 ms), each reading
    SENSOR_LAG_S old, motor 1's encoder counting the other way. Time is simulated, so it runs faster than real time."""

    def __init__(self, mode, floor, mapping):
        self.sim = SimMotor(mode, floor or "default", mapping)
        self.sim.set_params(FAKE_TRUTH)
        self.rng = np.random.default_rng()
        self.samples, self.history = [], deque()
        self.next_report = NOTIFY_MS / 1000
        self.pct = np.zeros(2)

    def now(self):
        return self.sim.d.time

    def start(self):
        pass

    def command(self, p0, p1):
        self.pct = np.array([p0, p1], dtype=float)
        self.sim.set_targets(pct_to_target(self.pct, FAKE_TRUTH["gain"], FAKE_TRUTH["top"]))

    def wait_until(self, t):
        while self.sim.d.time < t:
            self.sim.step()
            self.history.append((self.sim.d.time, self.sim.motor_positions()))
            if self.sim.d.time >= self.next_report:
                old = [p for tt, p in self.history if tt <= self.sim.d.time - SENSOR_LAG_S]
                pos = old[-1] if old else self.history[0][1]
                while self.history and self.history[0][0] < self.sim.d.time - 0.1:
                    self.history.popleft()
                deg = np.round(np.degrees(pos) * FAKE_TRUTH["encoder_sign"])
                self.samples.append((self.sim.d.time, *deg, *np.round(self.pct)))
                self.next_report += NOTIFY_MS / 1000 + self.rng.uniform(-0.003, 0.003)

    def finish(self):
        return self.samples

    def home_legs(self):
        pass

    def close(self):
        pass


# ---- recording ---------------------------------------------------------------------------------------------------


def record(mode, floor, fake):
    mapping = leg_mapping()
    if mode == "floor":
        segs = []   # forward % -> motor %
        for secs, p, _ in program_floor():
            pct = [0, 0]
            for side in (0, 1):
                pct[mapping["leg_motor"][side]] = mapping["cmd_sign"][side] * p
            segs.append((secs, *pct))
    else:
        segs = program_air()
    total = sum(s for s, _, _ in segs)
    rig = FakeRig(mode, floor, mapping) if fake else RealRig()
    try:
        rig.home_legs()
        if not fake:
            if mode == "air":
                input(f"\nProp the robot up (on a box or a mug) so BOTH legs spin freely without touching anything.\n"
                      f"The legs will run for about {total:.0f} s, both directions, up to full speed. Press Enter...")
            else:
                input(f"\nPut the robot on the {floor} floor with about 1.5 m clear in front of it, legs in the Onshape "
                      f"pose.\nIt walks for about {total:.0f} s at 20, 40, 60, 80 and 100% with short stops. Press Enter...")
        print(f"Recording {mode} run{' on ' + floor if floor else ''} ({total:.0f} s)...")
        rig.start()
        t0 = rig.now()
        commands, t_end = [], 0.0
        for secs, p0, p1 in segs:
            commands.append([round(rig.now() - t0, 4), p0, p1])
            rig.command(p0, p1)
            t_end += secs
            rig.wait_until(t0 + t_end)
        rig.command(0, 0)
        rig.wait_until(t0 + t_end + 0.3)
        samples = [[round(t - t0, 4), *map(float, rest)] for t, *rest in rig.finish()]
    finally:
        rig.close()

    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{'fake' if fake else 'real'}_{mode}{'_' + floor if floor else ''}_{stamp}.json"
    run = {"mode": mode, "floor": floor, "fake": fake, "when": stamp, "notify_ms": NOTIFY_MS, "mapping": mapping,
           "duration_s": round(t_end + 0.3, 3),
           "columns": ["t_s", "motor0_deg", "motor1_deg", "motor0_speed_raw", "motor1_speed_raw"],
           "commands": commands, "samples": samples}
    (RUNS / name).write_text(json.dumps(run))
    gaps = np.diff([s[0] for s in samples])
    print(f"Saved motor_runs/{name}: {len(samples)} readings, median {1000 * np.median(gaps):.0f} ms apart.")
    if len(samples) < run["duration_s"] / 0.1:
        print("WARNING: very few readings. Is the motor reporting? (legoeducation version, Bluetooth range)")


# ---- analysis ----------------------------------------------------------------------------------------------------


def real_positions(run):
    """Real encoder readings -> motor positions (rad, motor units) on the GRID_S time grid, plus each motor's encoder
    sign (+1 if the encoder counts up for positive %)."""
    s = np.array(run["samples"])
    t = s[:, 0] - SENSOR_LAG_S                      # when the reading was taken, not when it arrived
    deg = np.stack([np.unwrap(s[:, 1], period=360), np.unwrap(s[:, 2], period=360)], axis=1)
    grid = np.arange(0.0, run["duration_s"], GRID_S)
    pos = np.stack([np.interp(grid, t, deg[:, m]) for m in (0, 1)], axis=1)
    speed = speeds(pos)
    pct = command_at(run, grid[: len(speed)])
    sign = [1 if np.sum(pct[:, m] * speed[:, m]) >= 0 else -1 for m in (0, 1)]
    return grid, np.radians(pos) * sign, sign


def speeds(pos):
    return (pos[SPEED_WINDOW:] - pos[:-SPEED_WINDOW]) / (SPEED_WINDOW * GRID_S)


def command_at(run, times):
    cmd_t = np.array([c[0] for c in run["commands"]])
    idx = np.clip(np.searchsorted(cmd_t, times, side="right") - 1, 0, len(cmd_t) - 1)
    return np.array([run["commands"][i][1:3] for i in idx], dtype=float)


def segments(run):
    """[(start_s, end_s, %0, %1, previous %0, previous %1)] from the command list."""
    c = run["commands"]
    out = []
    for i, (t, p0, p1) in enumerate(c):
        end = c[i + 1][0] if i + 1 < len(c) else run["duration_s"]
        prev = c[i - 1][1:3] if i else [0, 0]
        out.append((t, end, p0, p1, *prev))
    return out


def speed_sweep(run):
    """Steady speed for every command held 1.2 s, per motor: [(%, rad/s)]. Uses the last 0.6 s of each hold."""
    grid, pos, _ = real_positions(run)
    points = ([], [])
    for start, end, p0, p1, *_ in segments(run):
        if end - start < 1.15 or p0 == 0:
            continue
        a, b = np.searchsorted(grid, [end - 0.6, end - 0.02])
        for m, p in enumerate((p0, p1)):
            points[m].append((p, (pos[b, m] - pos[a, m]) / (grid[b] - grid[a])))
    return [np.array(p) for p in points]


def fit_gain(points):
    """|speed| = gain x |%| - droop on the part below the top speed. Returns gain, droop, top, deadband %."""
    p, w = np.abs(points[:, 0]), np.abs(points[:, 1])
    top = float(np.mean(w[p == 100])) if np.any(p == 100) else float(w.max())
    use = (p >= 10) & (w < 0.9 * top)
    if use.sum() < 4:
        use = p < 100
    gain, intercept = np.polyfit(p[use], w[use], 1)
    droop = max(0.0, -float(intercept))
    return float(gain), droop, top, droop / gain


def step_metrics(run):
    """For each step from rest: delay until the leg has moved 3 degrees, and time to 63% of the steady speed."""
    s = np.array(run["samples"])
    grid, pos, sign = real_positions(run)
    v = speeds(pos)
    delays, rises = [], []
    for start, end, p0, p1, q0, q1 in segments(run):
        if p0 == 0 or q0 != 0 or q1 != 0 or end - start < 0.9:
            continue
        for m, p in enumerate((p0, p1)):
            col = 1 + m
            before = s[s[:, 0] <= start]
            if not len(before):
                continue
            moved = s[(s[:, 0] > start) & (np.abs(s[:, col] - before[-1, col]) >= 3)]
            if len(moved):
                delays.append(moved[0, 0] - start)
            a, b = np.searchsorted(grid, [start, end - 0.1])
            vv = v[a:min(b, len(v))]
            if len(vv) < 10:
                continue
            steady = np.mean(vv[-10:])
            hit = np.nonzero(np.abs(vv) >= 0.63 * abs(steady))[0]
            if len(hit):   # speed is averaged over 0.1 s, centred 0.05 s later than its index
                rises.append(grid[a + hit[0]] + SPEED_WINDOW * GRID_S / 2 - start)
    return np.array(delays), np.array(rises)


def notify_interval(run):
    t = np.array([x[0] for x in run["samples"]])
    return float(np.median(np.diff(t))) if len(t) > 2 else float("nan")


class Fitter:
    """Error between real and simulated leg speeds for a set of runs."""

    def __init__(self, air, floors, gain, top):
        self.air, self.floors, self.gain, self.top = air, floors, gain, top
        self.real = {id(r): real_positions(r) for r in [air, *floors]}

    def compare(self, params, run):
        grid, pos, _ = self.real[id(run)]
        t, sim = simulate(params, run, self.gain, self.top)
        sim_pos = np.stack([np.interp(grid, t, sim[:, m]) for m in (0, 1)], axis=1)
        return grid[: len(grid) - SPEED_WINDOW], speeds(pos), speeds(sim_pos)

    def air_error(self, params):          # RMS leg-speed error over the whole air run, rad/s
        _, real, sim = self.compare(params, self.air)
        return float(np.sqrt(np.mean((real - sim) ** 2)))

    def floor_stats(self, params, run):   # per command: (mean, spread) of leg speed, real and sim
        g, real, sim = self.compare(params, run)
        out = []
        for start, end, p0, p1, *_ in segments(run):
            if p0 == 0:
                continue
            k = (g >= start + 0.5) & (g <= end - 0.1)
            out.append((abs(p0), real[k].mean(axis=0), real[k].std(axis=0), sim[k].mean(axis=0), sim[k].std(axis=0)))
        return out

    def floor_error(self, params, run):   # RMS over commands and legs of the mean and spread errors, rad/s
        st = self.floor_stats(params, run)
        e = [np.concatenate([rm - sm, rs - ss]) for _, rm, rs, sm, ss in st]
        return float(np.sqrt(np.mean(np.square(e))))

    def cost(self, params):
        errs = [self.air_error(params) ** 2] + [self.floor_error(params, r) ** 2 for r in self.floors]
        return float(np.mean(errs))


def coordinate_search(cost, start, droop, log):
    """Simple and robust: try each parameter a bit up and down, keep what helps, take smaller steps when nothing does.

    It searches the time constant tau = armature / damping instead of the armature: in the air, scaling damping,
    armature and friction together gives exactly the same leg motion, so only their ratio (how fast the speed builds
    up) shows there. With tau held, changing damping only changes how much speed the legs lose under load, which the
    floor runs measure. frictionloss is tied to damping (frictionloss = droop x damping) so the air speeds stay right."""
    def full(p):
        return {"damping": p["damping"], "effort_limit": p["effort_limit"], "armature": p["tau"] * p["damping"],
                "frictionloss": droop * p["damping"], "delay_steps": p["delay_steps"]}
    bounds = {**BOUNDS, "tau": (1e-3, 1.0)}
    best = {"damping": start["damping"], "effort_limit": start["effort_limit"],
            "tau": start["armature"] / start["damping"], "delay_steps": start["delay_steps"]}
    best_c = cost(full(best))
    factor = {"damping": 2.0, "effort_limit": 2.0, "tau": 2.0}
    dstep = 4
    log(f"  start: error {math.sqrt(best_c):.3f} rad/s")
    for round_ in range(1, 40):
        improved = False
        for name in ("delay_steps", "tau", "damping", "effort_limit"):
            if name == "delay_steps":
                cands = [best[name] - dstep, best[name] + dstep]
            else:
                cands = [best[name] * factor[name], best[name] / factor[name]]
            lo, hi = bounds[name]
            for v in cands:
                if not lo <= v <= hi:
                    continue
                trial = {**best, name: v}
                c = cost(full(trial))
                if c < best_c - 1e-9:
                    best, best_c, improved = trial, c, True
        log(f"  round {round_:2d}: error {math.sqrt(best_c):.3f} rad/s   delay {best['delay_steps'] * DT * 1000:.0f} ms  "
            f"tau {best['tau'] * 1000:.0f} ms  damping {best['damping']:.4f}  effort {best['effort_limit']:.3f}")
        if not improved:
            if dstep == 1 and all(f < 1.06 for f in factor.values()):
                break
            dstep = max(1, dstep // 2)
            factor = {k: math.sqrt(f) for k, f in factor.items()}
    return full(best), math.sqrt(best_c)


def newest(pattern):
    files = sorted(RUNS.glob(pattern), key=lambda p: p.name.rsplit("_", 1)[-1])
    return files[-1] if files else None


def fit(fake):
    prefix = "fake" if fake else "real"
    air_file = newest(f"{prefix}_air_*.json")
    if not air_file:
        sys.exit(f"No {prefix} air run in motor_runs/: run `python characterize_motor.py air{' --fake' if fake else ''}` first.")
    air = json.loads(air_file.read_text())
    floor_files = [f for f in (newest(f"{prefix}_floor_{fl}_*.json") for fl in FLOORS) if f]
    floors = [json.loads(f.read_text()) for f in floor_files]
    print(f"Fitting to {air_file.name}" + "".join(f", {f.name}" for f in floor_files))
    if not floors:
        print("  (no floor runs: the torque limit is only loosely pinned down by the air run alone)")

    # 1. Straight from the data, no model.
    sweep = speed_sweep(air)
    per_motor = []
    for m in (0, 1):
        gain, droop, top, deadband = fit_gain(sweep[m])
        fwd = sweep[m][sweep[m][:, 0] == 100, 1]
        back = sweep[m][sweep[m][:, 0] == -100, 1]
        per_motor.append({"gain_rad_s_per_pct": gain, "top_speed_rad_s": top, "droop_rad_s": droop,
                          "deadband_pct": deadband, "top_positive_rad_s": float(np.mean(fwd)) if len(fwd) else None,
                          "top_negative_rad_s": float(-np.mean(back)) if len(back) else None})
    delays, rises = step_metrics(air)
    notify = notify_interval(air)
    gain = [pm["gain_rad_s_per_pct"] for pm in per_motor]
    top = [pm["top_speed_rad_s"] + pm["droop_rad_s"] for pm in per_motor]   # target cap (speed + what friction takes)
    droop = float(np.mean([pm["droop_rad_s"] for pm in per_motor]))
    print("\nMeasured directly:")
    for m, pm in enumerate(per_motor):
        print(f"  motor {m}: {pm['gain_rad_s_per_pct'] * 100:.1f} rad/s per 100%, top speed {pm['top_speed_rad_s']:.1f} "
              f"rad/s (+ {pm['top_positive_rad_s']:.1f} / - {pm['top_negative_rad_s']:.1f}), deadband "
              f"{pm['deadband_pct']:.1f}%")
    if len(delays):
        print(f"  first movement after a command: median {1000 * np.median(delays):.0f} ms "
              f"(10-90%: {1000 * np.percentile(delays, 10):.0f}-{1000 * np.percentile(delays, 90):.0f} ms)")
    if len(rises):
        print(f"  63% of the final speed after: median {1000 * np.median(rises):.0f} ms")
    print(f"  readings arrive every {1000 * notify:.0f} ms (median)")

    # 2. The mjlab model, fitted by replaying the runs in MuJoCo.
    fitter = Fitter(air, floors, gain, top)
    start = dict(CURRENT_SIM)
    if len(delays):
        start["delay_steps"] = int(np.clip(round((np.median(delays) - SENSOR_LAG_S) / DT), *BOUNDS["delay_steps"]))
    print("\nFitting the mjlab motor model (replaying the runs in MuJoCo)...")
    best, err = coordinate_search(fitter.cost, start, droop, print)
    best["delay_steps"] = int(best["delay_steps"])
    delay_range = [max(0, best["delay_steps"] - 2), best["delay_steps"] + 2]   # +-10 ms: readings and commands jitter
    current = {**CURRENT_SIM}
    errors = {"air": {"current_sim": fitter.air_error(current), "fitted": fitter.air_error(best)}}
    for r in floors:
        errors[f"floor_{r['floor']}"] = {"current_sim": fitter.floor_error(current, r), "fitted": fitter.floor_error(best, r)}

    model = {
        "what": "Double Motor model measured by hardware/characterize_motor.py; drive_task.py reads the 'mjlab' part",
        "fake": fake,
        "runs": [air_file.name, *[f.name for f in floor_files]],
        "measured": {"per_motor": per_motor, "notify_interval_s": notify,
                     "first_movement_s": {"median": float(np.median(delays)) if len(delays) else None,
                                          "p10": float(np.percentile(delays, 10)) if len(delays) else None,
                                          "p90": float(np.percentile(delays, 90)) if len(delays) else None},
                     "time_to_63pct_s": float(np.median(rises)) if len(rises) else None,
                     "sensor_lag_assumed_s": SENSOR_LAG_S},
        "mjlab": {
            "damping": best["damping"],
            "effort_limit": best["effort_limit"],
            "armature": best["armature"],
            "frictionloss": best["frictionloss"],
            "delay_steps": delay_range,              # physics steps of 5 ms: [min, max] for delay_min_lag/max_lag
            "physics_dt_s": DT,
            "max_leg_speed_rad_s": float(min(pm["top_speed_rad_s"] for pm in per_motor)),
        },
        "speed_command": {"gain_rad_s_per_pct": gain, "comment": "% = target rad/s / gain (motor units)"},
        "time_constant_s": best["armature"] / best["damping"],
        "fit_error_rad_s": errors,
    }
    if fake:
        model["fake_truth"] = FAKE_TRUTH
    out = HERE / ("motor_model_fake.json" if fake else "motor_model.json")
    out.write_text(json.dumps(model, indent=2))

    print("\nmjlab motor model:" + ("   (fake truth in brackets)" if fake else ""))
    def show(name, value, unit, truth=None):
        t = f"   [{truth}]" if truth is not None else ""
        print(f"  {name:13s} {value}{unit}{t}")
    ft = FAKE_TRUTH if fake else {}
    show("delay (fit)", f"{best['delay_steps'] * DT * 1000:.0f}", " ms", f"{ft['delay_steps'] * DT * 1000:.0f}" if fake else None)
    show("tau", f"{1000 * best['armature'] / best['damping']:.0f}", " ms (armature / damping: how fast speed builds up)",
         f"{1000 * ft['armature'] / ft['damping']:.0f}" if fake else None)
    show("damping", f"{best['damping']:.4f}", " N*m per rad/s", ft.get("damping"))
    show("effort_limit", f"{best['effort_limit']:.3f}", " N*m", ft.get("effort_limit"))
    show("armature", f"{best['armature']:.5f}", " kg*m^2", ft.get("armature"))
    show("frictionloss", f"{best['frictionloss']:.4f}", " N*m", ft.get("frictionloss"))
    show("delay range", f"{delay_range[0] * DT * 1000:.0f}-{delay_range[1] * DT * 1000:.0f}", " ms",
         f"{ft['delay_steps'] * DT * 1000:.0f} ms" if fake else None)
    show("top speed", f"{model['mjlab']['max_leg_speed_rad_s']:.1f}", " rad/s", min(ft["top"]) if fake else None)
    print("\nLeg-speed error, rad/s (lower is better):     drive_task.py now   fitted")
    for k, e in errors.items():
        print(f"  {k:40s} {e['current_sim']:8.2f}   {e['fitted']:8.2f}")
    print(f"\nWrote {out.name}." + ("" if fake else
          " drive_task.py now also registers Walk-Measured-<Floor> (see README.md, 'Motor model')."))
    plot(fitter, air, floors, sweep, per_motor, current, best, HERE / ("motor_fit_fake.png" if fake else "motor_fit.png"))


def plot(fitter, air, floors, sweep, per_motor, current, best, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip install matplotlib for motor_fit.png)")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    for m, pts in enumerate(sweep):
        a.plot(pts[:, 0], pts[:, 1], "o", ms=4, label=f"motor {m} measured")
    p = np.linspace(-100, 100, 201)
    for m, pm in enumerate(per_motor):
        cap = pm["top_speed_rad_s"]
        model = np.sign(p) * np.clip(pm["gain_rad_s_per_pct"] * np.abs(p) - pm["droop_rad_s"], 0, cap)
        a.plot(p, model, "-", lw=1, label=f"motor {m} model")
    a.set(xlabel="command (%)", ylabel="steady leg speed in the air (rad/s)", title="Speed sweep")
    a.grid(alpha=.3)
    a.legend(fontsize=8)

    g, real, sim_fit = fitter.compare(best, air)
    _, _, sim_cur = fitter.compare(current, air)
    steps_start = next(s for s, e, p0, _, q0, _ in segments(air) if p0 == 20 and q0 == 0 and abs(e - s - 1.0) < 0.05)
    for a, (lo, hi), title in ((ax[0, 1], (steps_start - 0.3, steps_start + 11.0), "Steps from rest (air), motor 0"),
                               (ax[1, 0], (air["duration_s"] - 11.5, air["duration_s"]), "Random commands (air), motor 0")):
        k = (g >= lo) & (g <= hi)
        a.plot(g[k], real[k, 0], "k", lw=1.5, label="measured")
        a.plot(g[k], sim_cur[k, 0], "C3", lw=1, alpha=.8, label="drive_task.py now")
        a.plot(g[k], sim_fit[k, 0], "C0", lw=1, label="fitted model")
        cmd = command_at(air, g[k])[:, 0] * fitter.gain[0]
        a.step(g[k], cmd, "C7", lw=.8, where="post", label="command x gain")
        a.set(xlabel="time (s)", ylabel="leg speed (rad/s)", title=title)
        a.grid(alpha=.3)
        a.legend(fontsize=8)

    a = ax[1, 1]
    for i, r in enumerate(floors):
        st_fit = fitter.floor_stats(best, r)
        st_cur = fitter.floor_stats(current, r)
        pct = [s[0] for s in st_fit]
        a.plot(pct, [s[1].mean() for s in st_fit], "o-", color=f"C{i}", label=f"{r['floor']}: measured")
        a.plot(pct, [s[3].mean() for s in st_fit], "s--", color=f"C{i}", mfc="none", label=f"{r['floor']}: fitted")
        a.plot(pct, [s[3].mean() for s in st_cur], "x:", color=f"C{i}", label=f"{r['floor']}: drive_task.py now")
    a.set(xlabel="command (%)", ylabel="mean leg speed walking (rad/s)", title="On the floor")
    a.grid(alpha=.3)
    if floors:
        a.legend(fontsize=7)
    else:
        a.text(0.5, 0.5, "no floor runs yet", ha="center", transform=a.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Wrote {path.name}.")


def main(argv):
    fake = "--fake" in argv
    argv = [a for a in argv if a != "--fake"]
    if argv[:1] == ["air"]:
        record("air", None, fake)
    elif argv[:1] == ["floor"] and len(argv) > 1 and argv[1] in FLOORS:
        record("floor", argv[1], fake)
    elif argv[:1] == ["fit"]:
        fit(fake)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
