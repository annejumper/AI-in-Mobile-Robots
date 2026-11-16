"""Drive a two-wheeled robot that was built in Onshape, and film it. Run it with runIt.py (see README.md).

Everything you may want to change is in the first block of constants.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering; must be set before importing mujoco

import imageio
import mujoco

MODEL = sys.argv[1] if len(sys.argv) > 1 else "onshape/scene.xml"   # your Onshape robot (or pass sample/scene.xml)
SURFACE = sys.argv[2] if len(sys.argv) > 2 else "laminate"         # floor: default, laminate or carpet (same values as drive_task.py)
SURFACES = {"default": (1.0, (0.02, 1.0)), "laminate": (0.3, (0.02, 1.0)), "carpet": (0.6, (0.03, 1.0))}   # (friction, solref) for ABS on that floor
LEFT, RIGHT = "left_wheel", "right_wheel"   # the wheel actuator names (printed below if they are wrong)
OUT = f"drive_{SURFACE}.mp4"

# What the wheels do, in order: (until this second, left wheel speed, right wheel speed), in radians/second.
PLAN = [
    (1.0, 0, 0),      # sit still and settle
    (4.0, 8, 8),      # both wheels forward: drive straight
    (7.0, 4, -4),     # wheels opposite: spin on the spot
    (10.0, 8, 4),     # left wheel faster: curve to the right
]

# ---- load the robot -------------------------------------------------------------------------------
spec = mujoco.MjSpec.from_file(MODEL)
chassis = spec.worldbody.first_body()
if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
    chassis.add_freejoint()           # let the robot move around (an Onshape export may pin it in place)
model = spec.compile()
data = mujoco.MjData(model)

print("joints:   ", [model.joint(i).name for i in range(model.njnt) if model.joint(i).name])
print("actuators:", [model.actuator(i).name for i in range(model.nu)])
left, right = model.actuator(LEFT).id, model.actuator(RIGHT).id   # stops with a clear error if a name is wrong

# Put the robot ON the floor (as when you set the real one down): find its exact lowest point and move it to z = 0.
# Mesh shapes use their real vertices; other shapes (the sample robot's boxes and spheres) use their bounding radius.
def lowest_point():
    low = float("inf")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or not model.geom_contype[g]:
            continue                                     # skip the floor and the visual-only copies
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            verts = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
            low = min(low, (verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
        else:
            low = min(low, data.geom_xpos[g, 2] - model.geom_rbound[g])
    return low


mujoco.mj_forward(model, data)
data.qpos[2] -= lowest_point()

# The floor. MuJoCo takes the LARGER friction of two touching geoms, so the floor's default 1.0 would always win.
# Priority 1 on the robot's collision geoms makes every robot-floor contact use the robot's values instead.
# (The whole Silly Walker is ABS, so the motor body and the beams get the same friction.)
friction, solref = SURFACES[SURFACE]
for g in range(model.ngeom):
    if model.geom_bodyid[g] != 0 and model.geom_contype[g]:
        model.geom_priority[g] = 1
        model.geom_friction[g, 0] = friction
        model.geom_solref[g] = solref
print(f"floor: {SURFACE} (friction {friction}, solref {solref})")


def drive(actuator, speed):
    """Tell one wheel to turn at `speed` rad/s for one physics step."""
    if model.actuator_biasprm[actuator, 1] != 0:           # a position motor: keep moving its target angle
        data.ctrl[actuator] += speed * model.opt.timestep
    else:                                                    # a velocity motor: give it the speed
        data.ctrl[actuator] = speed


# ---- simulate and film ----------------------------------------------------------------------------
renderer = mujoco.Renderer(model, 480, 640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING     # follow the robot
camera.trackbodyid = model.body(chassis.name).id
camera.distance, camera.azimuth, camera.elevation = 0.35, 140, -25   # close in: the Silly Walker is only ~8 cm long

FPS = 30
frames = []
for end, left_speed, right_speed in PLAN:
    while data.time < end:
        while data.time < len(frames) / FPS:
            drive(left, left_speed)
            drive(right, right_speed)
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    x, y = data.qpos[0], data.qpos[1]
    print(f"t={data.time:4.1f}s  robot at x={x:+.2f} m, y={y:+.2f} m")

renderer.close()
imageio.mimwrite(OUT, frames, fps=FPS)
print(f"wrote {OUT}: {len(frames)} frames")
