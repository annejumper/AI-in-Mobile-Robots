"""Walk the real Silly Walker with the policy trained on the carpet floor (Walk-Real-Carpet).

    python walk_carpet.py            # 10 s on the robot
    python walk_carpet.py 20         # 20 s
    python walk_carpet.py --fake     # no robot: the MuJoCo model on this laptop stands in for it

Run `python silly_walker_real.py calibrate` once first. See README.md.
"""
import sys

from silly_walker_real import main

main(["run", "carpet", *sys.argv[1:]])
