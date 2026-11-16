"""Train the Drive-Straight task. Same as mjlab's own `train`, but it registers our task first.

    python train.py Drive-Straight-Laminate --env.scene.num-envs 4096
"""
import sys

import drive_task  # noqa: F401  (registers "Drive-Straight-Laminate" and "-Carpet")
from mjlab.scripts.train import main

if __name__ == "__main__":
  sys.exit(main())
