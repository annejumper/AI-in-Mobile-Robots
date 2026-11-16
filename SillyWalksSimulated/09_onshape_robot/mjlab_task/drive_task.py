"""mjlab task: teach the Silly Walker (LEGO double motor with two beam legs) to walk in a straight line down +x.

The robot starts at the origin pointing a little off to one side (random yaw) and must drive forward
fast while staying on the line y = 0 and keeping its heading along +x. The policy sees its own
velocities, its heading error and its sideways offset, and outputs one wheel speed for each wheel.

Importing this file registers two tasks per floor in SURFACES:
  Drive-Straight-<Floor>  simulation-only: the policy sees perfect velocity and line offset (50 Hz).
  Walk-Real-<Floor>       hardware-ready: the policy sees only what the real Double Motor can measure (gyro, IMU heading,
                          leg speeds, its last command), with sensor noise, Bluetooth delay and randomized physics (25 Hz).
                          Export with export_policy.py and run on the robot with ../../hardware/.
(see train.py / play.py next to it).
Everything you will want to tune is in the constants block and the reward weights.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.actuator import BuiltinPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  dr,
  reset_joints_by_offset,
  action_rate_l2,
  base_ang_vel,
  base_lin_vel,
  bad_orientation,
  joint_vel_rel,
  last_action,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# ---- constants you may want to change ----------------------------------------------------------------
# the Onshape export: next to this file when run by runIt.py, one folder up in the repo
_HERE = Path(__file__).resolve().parent
ROBOT_XML = next(p for p in (_HERE / "onshape" / "robot.xml", _HERE.parent / "onshape" / "robot.xml") if p.exists())
LEFT, RIGHT = "left_wheel", "right_wheel"       # joint names in robot.xml
TARGET_SPEED = 0.07         # m/s the robot is rewarded for going forward (Silly Walker tops out near 0.075 m/s)
MAX_WHEEL_SPEED = 10.0      # rad/s; a policy output of 1.0 means this wheel speed
WHEEL_KD = 0.05             # how hard the motor pushes toward the wheel speed it was told (N*m per rad/s)
WHEEL_TORQUE_LIMIT = 0.2    # N*m (gentler than 0.3; joints already lose 0.1 to frictionloss)
START_YAW = 0.4             # rad; the robot starts pointing up to this far off +x
MAX_OFF_LINE = 1.5          # m; the episode ends if the robot wanders this far from y = 0
EPISODE_S = 10.0
FACE_YAW = math.pi / 2      # rad; turns the Onshape model (axles along x, walks toward -y) so it walks toward +x
START_HEIGHT = 0.0032       # m; chassis origin height at reset: the motor bottom (3.2 mm below origin) rests on the floor
# Floors to train on: ABS plastic (motor, hubs, beams) against each surface. Typical textbook values, not measured.
#   friction: sliding friction coefficient.  solref: (contact time constant s, damping ratio); larger = softer floor.
SURFACES = {
  "default": {"friction": 1.0, "solref": (0.02, 1.0)},    # MuJoCo's default floor (what the first experiment used)
  "laminate": {"friction": 0.3, "solref": (0.02, 1.0)},   # linoleum / laminate: slippery and hard
  "carpet": {"friction": 0.6, "solref": (0.03, 1.0)},     # dense low-pile commercial loop carpet: grippier, gives ~4 mm
}
# -------------------------------------------------------------------------------------------------------

# Hardware-ready tasks (Walk-Real-*): what the real robot's sensors and Bluetooth link look like.
REAL_DECIMATION = 8          # 8 x 0.005 s = 0.04 s per policy step (25 Hz); Bluetooth notifications come every 20 ms
REAL_NOISE = {"gyro": 0.15, "heading": 0.03, "leg_speed": 0.5}   # +- uniform noise: rad/s, (sin/cos units), rad/s
REAL_MAX_DELAY_STEPS = 1     # each reading may be up to one policy step (40 ms) old
REAL_LEG_START = 0.3         # rad; legs are set by hand to the Onshape pose, so allow this much error

_ROBOT = SceneEntityCfg("robot")


def _turn_contents(body, yaw: float) -> None:
  """Rotate everything inside `body` (shapes, child bodies, mass) by `yaw` radians about z, leaving the body's own
  frame alone. Used so the body's +x axis (what the rewards call "forward") points the way the robot walks."""
  c, s = math.cos(yaw), math.sin(yaw)
  R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  q = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  def turn(el):
    el.pos = R @ np.asarray(el.pos)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, np.asarray(el.quat, dtype=float))
    el.quat = out
  for el in [*body.geoms, *body.bodies, *body.sites]:
    turn(el)
  body.ipos = R @ np.asarray(body.ipos)
  if body.fullinertia[0] > 0:               # Onshape writes a full 3x3 inertia (ixx iyy izz ixy ixz iyz)
    ixx, iyy, izz, ixy, ixz, iyz = body.fullinertia
    I = R @ np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]) @ R.T
    body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
  else:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, np.asarray(body.iquat, dtype=float))
    body.iquat = out


def _get_spec(surface: str) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  for actuator in list(spec.actuators):     # mjlab adds its own wheel motors (below)
    spec.delete(actuator)
  chassis = spec.body("chassis")
  _turn_contents(chassis, FACE_YAW)         # the Silly Walker walks toward -y in Onshape; turn it to face +x
  chassis.add_freejoint()                   # let the robot move around (same as drive.py)
  # The floor. MuJoCo normally mixes two touching geoms by taking the LARGER friction, so the floor's default 1.0 would
  # always win. Giving every robot collision geom priority 1 makes MuJoCo use the robot's friction and solref for each
  # robot-floor contact instead, so these numbers ARE the floor. (The whole robot is ABS, so one value for all parts.)
  for i, g in enumerate(spec.geoms):
    if g.contype or g.conaffinity:          # collision shapes only, not the visual copies
      g.name = f"collision_{i}"             # named so Walk-Real-* can randomize their friction
      g.priority = 1
      g.friction[0] = SURFACES[surface]["friction"]
      g.solref = SURFACES[surface]["solref"]
  return spec


def _robot_cfg(surface: str) -> EntityCfg:
  return EntityCfg(
    spec_fn=functools.partial(_get_spec, surface),
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, START_HEIGHT), joint_vel={".*": 0.0}),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        # stiffness 0 = no position target, so the motor only chases the velocity we ask for
        BuiltinPdActuatorCfg(
          target_names_expr=(LEFT, RIGHT), stiffness=0.0, damping=WHEEL_KD, effort_limit=WHEEL_TORQUE_LIMIT
        ),
      ),
    ),
  )


# ---- observations and rewards: how the robot is doing relative to "straight down +x" -------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Sideways distance from this env's start line (its y axis), in metres."""
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]


def heading(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw)]: what the real robot's IMU yaw gives (relative to where it pointed at the start)."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def heading_and_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[sin(yaw), cos(yaw), off-line distance]. Shape [num_envs, 3]."""
  yaw = _yaw(env)
  return torch.stack([torch.sin(yaw), torch.cos(yaw), _off_line(env)], dim=-1)


def forward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for going along +x at TARGET_SPEED: 0 when standing still, 1 at the target, less beyond it, negative backwards.

  It slopes everywhere (a narrow bump around the target would give a robot that starts out still nothing to learn from).
  """
  vx = env.scene["robot"].data.root_link_lin_vel_w[:, 0]
  return torch.clamp(1.0 - torch.square((vx - TARGET_SPEED) / TARGET_SPEED), min=-1.0)


def stay_on_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_off_line(env))


def face_forward(env: ManagerBasedRlEnv) -> torch.Tensor:
  return -torch.square(_yaw(env))


def too_far_off_line(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _off_line(env).abs() > MAX_OFF_LINE


# ---- environment ----------------------------------------------------------------------------------------


def drive_straight_env_cfg(surface: str, play: bool = False, real: bool = False) -> ManagerBasedRlEnvCfg:
  # Everything the simulator knows. The sim-only policy sees all of it; for Walk-Real-* only the critic does (the critic
  # only scores actions during training and is not used on the robot, so it may use information the robot lacks).
  full_terms = {
    "lin_vel": ObservationTermCfg(func=base_lin_vel),
    "ang_vel": ObservationTermCfg(func=base_ang_vel),
    "heading_offset": ObservationTermCfg(func=heading_and_offset),
    "wheel_vel": ObservationTermCfg(func=joint_vel_rel),
    "last_action": ObservationTermCfg(func=last_action),
  }
  if real:
    def sensed(func, noise):   # a real sensor reading: noisy and up to REAL_MAX_DELAY_STEPS old
      return ObservationTermCfg(func=func, noise=Unoise(n_min=-noise, n_max=noise), delay_max_lag=REAL_MAX_DELAY_STEPS)
    # ORDER MATTERS: hardware/silly_walker_real.py builds its 9 inputs in exactly this order.
    actor_terms = {
      "ang_vel": sensed(base_ang_vel, REAL_NOISE["gyro"]),          # IMU gyroscope, body frame (x fwd, y left, z up)
      "heading": sensed(heading, REAL_NOISE["heading"]),            # IMU yaw
      "wheel_vel": sensed(joint_vel_rel, REAL_NOISE["leg_speed"]),  # motor encoders (left, right)
      "last_action": ObservationTermCfg(func=last_action),          # what the policy sent last step
    }
  else:
    actor_terms = full_terms
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    "critic": ObservationGroupCfg({**full_terms}),
  }

  actions: dict[str, ActionTermCfg] = {
    "wheels": JointVelocityActionCfg(
      entity_name="robot", actuator_names=(LEFT, RIGHT), scale=MAX_WHEEL_SPEED, use_default_offset=True
    ),
  }

  events = {
    "reset_pose": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-START_YAW, START_YAW)}, "velocity_range": {}},
    ),
  }
  if real:   # the real robot is never exactly the model: randomize what we measured or guessed
    events.update({
      "leg_start": EventTermCfg(   # legs set by hand to the Onshape pose, never exactly
        func=reset_joints_by_offset, mode="reset",
        params={"position_range": (-REAL_LEG_START, REAL_LEG_START), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(LEFT, RIGHT))},
      ),
      "floor_friction": EventTermCfg(   # textbook friction values, not measured: +-30%
        func=dr.geom_friction, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=("collision_.*",)), "operation": "scale",
                "ranges": (0.7, 1.3), "shared_random": True},
      ),
      "motor_mass": EventTermCfg(   # scale reading and battery charge: +-10%
        func=dr.body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("chassis",)), "operation": "scale", "ranges": (0.9, 1.1)},
      ),
      "motor_com": EventTermCfg(   # the motor's mass is not really spread evenly (motors, gears, battery inside): +-5 mm
        func=dr.body_com_offset, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("chassis",)), "operation": "add",
                "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.005, 0.005)}},
      ),
      "motor_strength": EventTermCfg(   # real LEGO motor torque is not known exactly: +-30%
        func=dr.effort_limits, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "effort_limit_range": (0.7, 1.3), "operation": "scale"},
      ),
    })

  rewards = {
    "forward_speed": RewardTermCfg(func=forward_speed, weight=1.0),
    "stay_on_line": RewardTermCfg(func=stay_on_line, weight=1.0),
    "face_forward": RewardTermCfg(func=face_forward, weight=0.5),
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.05),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "tipped_over": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(60.0)}),
    "off_line": TerminationTermCfg(func=too_far_off_line),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg(surface)},
      num_envs=1,
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.45,                  # close in: the Silly Walker is only ~8 cm long
      elevation=-25.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005),
      njmax=400,       # contact-constraint slots per robot; the default ran out ("nefc overflow") and dropped contacts
    ),
    decimation=REAL_DECIMATION if real else 4,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def drive_straight_ppo_runner_cfg(surface: str, real: bool = False) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=f"{'walk_real' if real else 'drive_straight'}_{surface}",   # logs/rsl_rl/<this name>/
    logger="tensorboard",           # the default, wandb, needs an account and a login
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=500,
  )


for _surface in SURFACES:     # registers Drive-Straight-<Floor> (sim-only) and Walk-Real-<Floor> (hardware-ready)
  for _real, _prefix in ((False, "Drive-Straight"), (True, "Walk-Real")):
    register_mjlab_task(
      task_id=f"{_prefix}-{_surface.capitalize()}",
      env_cfg=drive_straight_env_cfg(_surface, real=_real),
      play_env_cfg=drive_straight_env_cfg(_surface, play=True, real=_real),
      rl_cfg=drive_straight_ppo_runner_cfg(_surface, real=_real),
    )
