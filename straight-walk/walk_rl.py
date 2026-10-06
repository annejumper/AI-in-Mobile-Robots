"""
Q-learning to make a two-legged LEGO CS & AI Double Motor walker go straight.

The Double Motor's built-in IMU yaw is the only feedback: the agent learns
how to bias the left/right leg speeds so the heading stays at 0.

Usage (with the venv active):
    python walk_rl.py check    # connect, print sensor values, test walk
    python walk_rl.py train    # learn / keep learning a Q-table
    python walk_rl.py run      # walk using the learned Q-table (no exploring)
    python walk_rl.py show     # print the Q-table (no robot needed)

The Q-table is saved to q_table.json after every episode, so you can stop
(Ctrl+C) and resume training later, e.g. after charging the battery.
"""

import json
import random
import sys
import time
from pathlib import Path

import legoeducation as le

CARD_SERIAL = "6065"
Q_FILE = Path(__file__).with_name("q_table.json")

# --- Walking parameters -----------------------------------------------------
BASE_SPEED = 50                      # % speed both legs use when walking straight
STEERING = [-20, -10, 0, 10, 20]     # action = speed bias between the legs
STEP_DEGREES = 360                   # each action = one full leg rotation (one stride)
STEPS_PER_EPISODE = 15
YAW_SCALE = 1.0                      # set to 0.1 if `check` shows yaw in tenths of a degree
YAW_FACE = None                      # e.g. le.DEVICE_FACE_RIGHT if `check` shows yaw not changing when the robot turns
YAW_SAMPLES = 5                      # readings averaged per measurement to smooth out leg wobble
FAIL_ANGLE = 45                      # episode ends if heading drifts past this many degrees

# --- State discretisation ---------------------------------------------------
# Heading error bins (degrees) and turn-rate bins (degrees per step).
ERROR_EDGES = [-30, -15, -5, 5, 15, 30]          # -> 7 bins
TURN_EDGES = [-3, 3]                             # -> 3 bins: turning one way / steady / other way
N_STATES = (len(ERROR_EDGES) + 1) * (len(TURN_EDGES) + 1)
N_ACTIONS = len(STEERING)

# --- Learning parameters ----------------------------------------------------
ALPHA = 0.2          # learning rate
GAMMA = 0.9          # discount factor
EPS_START = 0.5      # initial exploration rate
EPS_MIN = 0.05
EPS_DECAY = 0.93     # multiplied in after each episode


def bin_index(value, edges):
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def to_state(error, turn):
    return bin_index(error, ERROR_EDGES) * (len(TURN_EDGES) + 1) + bin_index(turn, TURN_EDGES)


def wrap180(angle):
    return (angle + 180) % 360 - 180


def reward_for(error, turn, action):
    # Stay pointed straight, don't swing, and prefer gentle corrections.
    return 1.0 - abs(error) / 10 - abs(turn) / 10 - abs(STEERING[action]) / 100


def load_q():
    if Q_FILE.exists():
        data = json.loads(Q_FILE.read_text())
        return data["q"], data["episodes"]
    return [[0.0] * N_ACTIONS for _ in range(N_STATES)], 0


def save_q(q, episodes):
    Q_FILE.write_text(json.dumps({"q": q, "episodes": episodes}, indent=1))


def choose_action(q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    best = max(q[state])
    return random.choice([a for a, v in enumerate(q[state]) if v == best])


class Walker:
    def __init__(self):
        self.motor = le.DoubleMotor()

    def connect(self):
        print(f"Connecting to Double Motor with card {CARD_SERIAL}...")
        self.motor.connect(card_serial=CARD_SERIAL, device_notification_delay=50)
        if not self.motor.connected:
            sys.exit("Could not connect. Is the Double Motor on and the card correct?")
        self.motor.movement_set_end_state(le.MOTOR_END_STATE_BRAKE)
        if YAW_FACE is not None:
            self.motor.imu_set_yaw_face(YAW_FACE)
        self.motor.beep()
        print("Connected.")
        self.start_yaw = 0.0

    def raw_yaw(self):
        # Average a few readings so a single wobble mid-stride doesn't dominate.
        total = 0.0
        for _ in range(YAW_SAMPLES):
            total += self.motor.imu_device.yaw * YAW_SCALE
            time.sleep(0.05)
        return total / YAW_SAMPLES

    def yaw(self):
        return wrap180(self.raw_yaw() - self.start_yaw)

    def reset_heading(self):
        # Measure relative to wherever the robot points now, so a slow or
        # missed yaw reset can't cause an instant failure.
        self.motor.imu_reset_yaw_axis(0)
        time.sleep(0.3)
        self.start_yaw = self.raw_yaw()

    def act(self, action):
        # Run one full stride so every heading reading is taken at the same
        # point in the gait.
        bias = STEERING[action]
        self.motor.movement_move_tank_for_degrees(
            STEP_DEGREES, speed_left=BASE_SPEED - bias, speed_right=BASE_SPEED + bias)

    def stop(self):
        self.motor.movement_stop()

    def disconnect(self):
        try:
            self.stop()
        finally:
            self.motor.disconnect()


def run_episode(walker, q, epsilon, learn):
    walker.reset_heading()
    error, turn = 0.0, 0.0
    state = to_state(error, turn)
    total = 0.0

    for step in range(STEPS_PER_EPISODE):
        action = choose_action(q, state, epsilon)
        walker.act(action)

        new_error = wrap180(walker.yaw())
        turn = wrap180(new_error - error)
        error = new_error
        failed = abs(error) > FAIL_ANGLE
        reward = -10.0 if failed else reward_for(error, turn, action)
        next_state = to_state(error, turn)

        if learn:
            target = reward if failed else reward + GAMMA * max(q[next_state])
            q[state][action] += ALPHA * (target - q[state][action])

        total += reward
        print(f"  step {step:2d}  steer {STEERING[action]:+3d}  yaw {error:+6.1f}  "
              f"turn {turn:+5.1f}  reward {reward:+5.2f}")
        state = next_state
        if failed:
            print("  Drifted too far, ending episode.")
            break

    walker.stop()
    return total, error


def check(walker):
    print("\nTurn the robot by hand. Yaw should change by about 90 for a quarter turn.")
    print("If it changes by about 900, set YAW_SCALE = 0.1.  (Ctrl+C to skip)\n")
    walker.reset_heading()
    try:
        for _ in range(100):
            imu = walker.motor.imu_device
            left, right = walker.motor.motor
            print(f"\r  yaw {imu.yaw:6}  pitch {imu.pitch:6}  roll {imu.roll:6}  "
                  f"leg positions L {left.position:7} R {right.position:7}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    input("\n\nPut the robot on the floor, then press Enter to walk 8 strides with equal leg speeds...")
    print("Yaw should creep slowly. If it jumps by tens of degrees while the robot")
    print("walks straight, the wrong face is set as yaw: try a YAW_FACE setting.\n")
    walker.reset_heading()
    for stride in range(8):
        walker.act(STEERING.index(0))
        imu = walker.motor.imu_device
        print(f"  stride {stride + 1}  heading {walker.yaw():+6.1f}   "
              f"raw yaw {imu.yaw:6}  pitch {imu.pitch:6}  roll {imu.roll:6}")
    walker.stop()


def train(walker):
    q, episodes = load_q()
    epsilon = max(EPS_MIN, EPS_START * EPS_DECAY ** episodes)
    print(f"Starting at episode {episodes}, epsilon {epsilon:.2f}. Ctrl+C to stop and save.")
    try:
        while True:
            input(f"\nPoint the robot straight on the floor and press Enter for episode {episodes + 1}...")
            total, final = run_episode(walker, q, epsilon, learn=True)
            episodes += 1
            save_q(q, episodes)
            print(f"Episode {episodes}: total reward {total:+.1f}, final heading {final:+.1f}, "
                  f"epsilon {epsilon:.2f} (saved)")
            epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
    except KeyboardInterrupt:
        save_q(q, episodes)
        print(f"\nStopped. Q-table saved after {episodes} episodes.")


def run(walker):
    q, episodes = load_q()
    if episodes == 0:
        sys.exit("No trained Q-table yet. Run `python walk_rl.py train` first.")
    input(f"Using Q-table from {episodes} episodes. Point the robot straight and press Enter...")
    total, final = run_episode(walker, q, epsilon=0.0, learn=False)
    print(f"Total reward {total:+.1f}, final heading {final:+.1f}")


def range_label(i, edges):
    if i == 0:
        return f"< {edges[0]}"
    if i == len(edges):
        return f">= {edges[-1]}"
    return f"{edges[i - 1]} to {edges[i]}"


def show():
    q, episodes = load_q()
    turn_names = ["turning -", "steady", "turning +"]
    print(f"Q-table after {episodes} episodes "
          f"({N_STATES} states x {N_ACTIONS} actions; * = action the robot would pick)\n")
    print(f"  {'heading':>12}  {'turn':>10}   " + "".join(f"steer {s:+3d}  " for s in STEERING))
    for state, row in enumerate(q):
        e, t = divmod(state, len(TURN_EDGES) + 1)
        best = row.index(max(row)) if any(row) else None
        cells = "".join(f"{v:+8.2f}{'*' if a == best else ' '}   " for a, v in enumerate(row))
        note = "" if any(row) else "  (never visited)"
        print(f"  {range_label(e, ERROR_EDGES):>12}  {turn_names[t]:>10}   {cells}{note}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "show":
        return show()
    if mode not in ("check", "train", "run"):
        sys.exit(__doc__)
    walker = Walker()
    walker.connect()
    try:
        {"check": check, "train": train, "run": run}[mode](walker)
    except KeyboardInterrupt:
        pass
    finally:
        walker.disconnect()


if __name__ == "__main__":
    main()
