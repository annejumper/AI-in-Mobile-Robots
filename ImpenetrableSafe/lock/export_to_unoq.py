"""
Get the trained model ready for the UNO Q. Run after every train.py, then deploy:

    python lock/export_to_unoq.py
    python3 tools/deploy.py unoq/finger-lock

What it does:
  1. converts models/finger_cnn.pt to ONNX (a small file onnxruntime can run on the board;
     full PyTorch is too heavy there)
  2. checks the ONNX model gives the same answers as PyTorch on the training images
  3. copies the model, the crop square (roi.json), the camera settings and lock_policy.py
     into the board app, so both sides crop, scale and decide exactly the same way
"""
import json
import random
import shutil

import cv2
import numpy as np
import onnxruntime as ort
import torch

import config
from model import load_model, preprocess

APP_PY = config.ROOT / "unoq" / "finger-lock" / "python"
ONNX_PATH = APP_PY / "finger_cnn.onnx"


def main():
    APP_PY.mkdir(parents=True, exist_ok=True)
    model = load_model(config.MODEL_PATH)
    dummy = torch.zeros(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    torch.onnx.export(model, dummy, str(ONNX_PATH), input_names=["image"], output_names=["scores"],
                      dynamo=False)
    print(f"wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1024:.0f} KB)")

    # Same answers?
    paths = [p for c in config.CLASSES for p in (config.DATA_DIR / str(c)).glob("*.jpg")]
    random.Random(0).shuffle(paths)
    sess = ort.InferenceSession(str(ONNX_PATH))
    agree, max_diff = 0, 0.0
    sample = paths[:300]
    with torch.no_grad():
        for p in sample:
            x = preprocess(cv2.imread(str(p))).unsqueeze(0)
            a = model(x).numpy()
            b = sess.run(None, {"image": x.numpy()})[0]
            agree += int(a.argmax() == b.argmax())
            max_diff = max(max_diff, float(np.abs(a - b).max()))
    print(f"ONNX vs PyTorch on {len(sample)} images: same answer {agree}/{len(sample)}, "
          f"largest score difference {max_diff:.2e}")

    x, y, s = config.load_roi()
    settings = {"cap_w": config.CAP_W, "cap_h": config.CAP_H, "img_size": config.IMG_SIZE,
                "classes": config.CLASSES, "roi": {"x": x, "y": y, "size": s}}
    (APP_PY / "settings.json").write_text(json.dumps(settings, indent=2))
    shutil.copy(config.ROOT / "lock" / "lock_policy.py", APP_PY / "lock_policy.py")
    if config.CODE_PATH.exists():
        shutil.copy(config.CODE_PATH, APP_PY / "lock_code.json")
        print("copied lock_code.json (kept out of git)")
    print(f"copied settings.json and lock_policy.py into {APP_PY}")


if __name__ == "__main__":
    main()
