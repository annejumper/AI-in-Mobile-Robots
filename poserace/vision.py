"""Webcam capture + MediaPipe Pose landmark detection."""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_lite.task")


def ensure_model_downloaded():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading pose landmarker model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker() -> vision.PoseLandmarker:
    ensure_model_downloaded()
    # macOS builds of mediapipe>=1.0 crash trying to open a Metal GPU
    # service for the pose detector's NMS calculator; the CPU delegate flag
    # alone doesn't prevent it. This project pins mediapipe==0.10.35 (see
    # requirements.txt), which runs the same Tasks API fully on CPU.
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


class PoseCamera:
    """Opens a webcam and yields (frame, landmarks) pairs.

    Frames are mirrored (flipped horizontally) before detection so the
    on-screen view feels natural, like looking in a mirror.
    """

    def __init__(self, camera_index: int = 0, mirrored: bool = True):
        self.camera_index = camera_index
        self.mirrored = mirrored
        self._cap = None
        self._landmarker = None
        self._start_time = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self.camera_index}")
        self._landmarker = create_landmarker()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cap is not None:
            self._cap.release()
        if self._landmarker is not None:
            self._landmarker.close()

    def read(self):
        """Read one frame. Returns (frame_bgr, landmarks_or_None)."""
        ok, frame = self._cap.read()
        if not ok:
            return None, None

        if self.mirrored:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        return frame, landmarks
