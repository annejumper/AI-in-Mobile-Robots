"""Settings shared by every lock script, so the computer and the UNO Q see images the same way."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CAM_INDEX = 0            # OpenCV index of the icspring webcam (press n in collect.py to cycle)
CAP_W, CAP_H = 1280, 720 # capture resolution: keep the same on the UNO Q (it changes the field of view)

IMG_SIZE = 64            # the network sees IMG_SIZE x IMG_SIZE colour crops of the red square
CLASSES = [0, 1, 2, 3, 4, 5]   # number of fingers; 0 = fist or empty square

DATA_DIR = ROOT / "data"                   # data/<fingers>/<image>.jpg
MODEL_PATH = ROOT / "models" / "finger_cnn.pt"
RESULTS_DIR = ROOT / "results"
ROI_PATH = ROOT / "roi.json"               # where the red square is in the camera image
CODE_PATH = ROOT / "lock_code.json"        # the secret code (not committed to git)


def load_roi():
    """(x, y, size) of the square crop in the full camera frame."""
    if ROI_PATH.exists():
        r = json.loads(ROI_PATH.read_text())
        return r["x"], r["y"], r["size"]
    s = min(CAP_W, CAP_H)
    return (CAP_W - s) // 2, (CAP_H - s) // 2, s


def save_roi(x, y, size):
    ROI_PATH.write_text(json.dumps({"x": int(x), "y": int(y), "size": int(size)}, indent=2))


def crop(frame):
    x, y, s = load_roi()
    return frame[y:y + s, x:x + s]
