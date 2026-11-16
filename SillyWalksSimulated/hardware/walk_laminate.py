"""Walk the real Silly Walker with the policy trained on the laminate floor (Walk-Real-Laminate).

    python walk_laminate.py            # 10 s on the robot
    python walk_laminate.py 20         # 20 s
    python walk_laminate.py --fake     # no robot: the MuJoCo model on this laptop stands in for it

Run `python silly_walker_real.py calibrate` once first. See README.md.
"""
import sys

from silly_walker_real import main

main(["run", "laminate", *sys.argv[1:]])
