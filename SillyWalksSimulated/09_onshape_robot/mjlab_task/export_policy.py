"""Export a trained Walk-Real-* policy so the real robot can run it (on the cluster, like make_video.py).

    python3 runIt.py --gpu --time 00:10:00 --mem 16G --cpus 4 \
      --with 09_onshape_robot/mjlab_task/drive_task.py --with 09_onshape_robot/onshape \
      --with runs/<your run>/logs/rsl_rl/walk_real_laminate/<date folder>/model_1999.pt \
      09_onshape_robot/mjlab_task/export_policy.py model_1999.pt Walk-Real-Laminate

Writes, next to this script (runIt.py brings them home to runs/<name>/):
  policy_<floor>.onnx       (policy_measured_<floor>.* for Walk-Measured-*) the policy network, including its input normalization (run it with onnxruntime)
  policy_<floor>.json       what the inputs and outputs mean, the control rate, and the speed scale
  policy_<floor>_check.npz  500 inputs recorded from a simulated run and the policy's outputs for them, so the laptop
                            can prove its ONNX copy gives the same answers (hardware/silly_walker_real.py verify)
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import drive_task  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

if len(sys.argv) != 3 or not sys.argv[2].startswith(("Walk-Real-", "Walk-Measured-")):
  sys.exit("usage: export_policy.py model_1999.pt Walk-Real-Laminate   (or -Default / -Carpet, or Walk-Measured-*)")
checkpoint, TASK = Path(sys.argv[1]), sys.argv[2]
measured = TASK.startswith("Walk-Measured-")
floor = TASK.split("-")[-1].lower()
stem = f"policy_{'measured_' if measured else ''}{floor}"   # policy_laminate / policy_measured_laminate

device = "cuda:0" if torch.cuda.is_available() else "cpu"
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
agent_cfg = load_rl_cfg(TASK)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=device)
runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

# 1. The network, with its observation normalizer built in.
runner.export_policy_to_onnx(".", f"{stem}.onnx")

# 2. Reference pairs from a real simulated rollout (realistic inputs, not just random numbers).
obs_log, act_log = [], []
obs = env.get_observations()
with torch.no_grad():
  for _ in range(500):
    action = policy(obs)
    obs_log.append(obs["actor"][0].cpu().numpy().copy())
    act_log.append(action[0].cpu().numpy().copy())
    obs = env.step(action)[0]
np.savez(f"{stem}_check.npz", obs=np.array(obs_log, dtype=np.float32), action=np.array(act_log, dtype=np.float32))

# 3. What the numbers mean. The input order is the order of the actor terms in drive_task.py.
groups = env.unwrapped.observation_manager.active_terms["actor"]
dims = env.unwrapped.observation_manager.group_obs_term_dim["actor"]
meta = {
  "task": TASK,
  "floor": floor,
  "floor_friction": drive_task.SURFACES[floor]["friction"],
  "control_dt_s": env.unwrapped.step_dt,
  "inputs": [{"name": n, "size": int(np.prod(d))} for n, d in zip(groups, dims)],
  "input_units": {
    "ang_vel": "rad/s, body frame: x forward, y left, z up (gyroscope)",
    "heading": "[sin(yaw), cos(yaw)], yaw in rad from the start direction, positive = turned left (counterclockwise from above)",
    "wheel_vel": "rad/s, [left leg, right leg], positive = the direction that walks the robot forward",
    "last_action": "the policy's own previous output, [left, right]",
  },
  "outputs": "[left, right] in about -1..1; target leg speed = output x max_leg_speed_rad_s, positive = forward",
  "max_leg_speed_rad_s": float(env_cfg.actions["wheels"].scale),
  "motor_model": drive_task.MOTOR_MODEL if measured else None,
  "target_speed_m_s": drive_task.TARGET_SPEED,
  "checkpoint": checkpoint.name,
}
Path(f"{stem}.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
print(f"wrote {stem}.onnx, {stem}.json, {stem}_check.npz")
env.close()
