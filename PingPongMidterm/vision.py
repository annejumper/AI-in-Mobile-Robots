"""
Camera-side sensing for T-anne-is.

  * PoseTracker - MediaPipe Pose Landmarker; finds the player's racket hand
                  (and the free hand, which tosses the ball on a serve).
  * TagReader   - AprilTag (36h11) detection with OpenCV's aruco module.
  * BodyFrame   - maps world meters (ball position) onto the screen using the
                  player's own body as a ruler, so it works at any distance.

Everything returned here is in MIRRORED display pixels (selfie view), so the
player's right hand is on the right side of the screen.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from court import HIP_HEIGHT

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
ARM = {"right": (R_SHOULDER, R_ELBOW, R_WRIST), "left": (L_SHOULDER, L_ELBOW, L_WRIST)}
SKELETON = [(L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
            (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST), (L_SHOULDER, L_HIP),
            (R_SHOULDER, R_HIP), (L_HIP, R_HIP)]

TORSO_M = 0.50      # shoulder-to-hip length of a typical adult, m
SHOULDER_M = 0.38   # shoulder width, m (used if the hips are out of frame)
VISIBLE = 0.5


@dataclass
class Body:
    points: dict            # landmark index -> (x, y, visibility) in display px
    racket: tuple           # (x, y) of the racket hand, or None
    hip: tuple              # hip center (x, y)
    torso_px: float         # pixels per TORSO_M
    free: tuple = None      # (x, y) of the free (tossing) hand, or None


class PoseTracker:
    def __init__(self, model_path, hand="right", infer_width=640):
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        self.arm = ARM[hand]
        self.free_arm = ARM["left" if hand == "right" else "right"]
        self.infer_width = infer_width
        self._t0 = None
        self._last_ms = -1

    def process(self, frame_bgr, t):
        """Run pose on the UN-mirrored camera frame (so MediaPipe's left/right
        labels are anatomically correct) and return a mirrored Body or None."""
        h, w = frame_bgr.shape[:2]
        small = frame_bgr
        if w > self.infer_width:
            small = cv2.resize(frame_bgr, (self.infer_width, int(h * self.infer_width / w)))
        if self._t0 is None:
            self._t0 = t
        ms = max(int((t - self._t0) * 1000), self._last_ms + 1)
        self._last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, ms)
        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        pts = {}
        for i in (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP):
            lm = lms[i]
            vis = lm.visibility if lm.visibility is not None else 1.0
            pts[i] = ((1.0 - lm.x) * w, lm.y * h, vis)

        def seen(*ids):
            return all(pts[i][2] > VISIBLE for i in ids)

        if not seen(L_SHOULDER, R_SHOULDER):
            return None
        sh = np.mean([pts[L_SHOULDER][:2], pts[R_SHOULDER][:2]], axis=0)
        if seen(L_HIP, R_HIP):
            hip = np.mean([pts[L_HIP][:2], pts[R_HIP][:2]], axis=0)
            torso_px = float(np.linalg.norm(hip - sh))
        else:  # hips out of frame: estimate them from shoulder width
            sw = float(np.linalg.norm(np.subtract(pts[L_SHOULDER][:2], pts[R_SHOULDER][:2])))
            torso_px = sw * TORSO_M / SHOULDER_M
            hip = sh + np.array([0.0, torso_px])

        _, elbow, wrist = self.arm
        racket = None
        if seen(wrist):
            wx, wy = pts[wrist][:2]
            if seen(elbow):  # the hub sits in the palm, a bit past the wrist
                ex, ey = pts[elbow][:2]
                wx, wy = wx + 0.35 * (wx - ex), wy + 0.35 * (wy - ey)
            racket = (wx, wy)
        free_wrist = self.free_arm[2]
        free = pts[free_wrist][:2] if seen(free_wrist) else None
        return Body(pts, racket, (float(hip[0]), float(hip[1])), torso_px, free)


class BodyFrame:
    """World (x, y, z) meters -> screen pixels, anchored on the player's hips
    and scaled by their torso. A one-point perspective makes far-away
    things (the opponent, the net) shrink toward a vanishing point."""

    DEPTH = 4.0  # perspective strength: things DEPTH meters away appear half size

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.ox, self.oy = width / 2, height * 0.62
        self.ppm = height / 2.4
        self.tracking = False

    def update(self, body, alpha=0.15):
        if body is None:
            self.tracking = False
            return
        self.tracking = True
        ppm = body.torso_px / TORSO_M
        self.ox += alpha * (body.hip[0] - self.ox)
        self.oy += alpha * (body.hip[1] - self.oy)
        self.ppm += alpha * (ppm - self.ppm)

    def to_screen(self, x, y):
        return self.ox + x * self.ppm, self.oy - (y - HIP_HEIGHT) * self.ppm

    def to_world(self, sx, sy):
        """Screen pixel on the player's body plane -> world (x, y) meters."""
        return (sx - self.ox) / self.ppm, HIP_HEIGHT + (self.oy - sy) / self.ppm

    def project(self, x, y, z):
        sx, sy = self.to_screen(x, y)
        s = self.DEPTH / (self.DEPTH + max(z, -0.7 * self.DEPTH))
        vx, vy = self.ox, self.h * 0.08
        return (vx + (sx - vx) * s, vy + (sy - vy) * s), s


class TagReader:
    def __init__(self):
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(TAG_DICT), cv2.aruco.DetectorParameters())

    def detect(self, frame_bgr):
        """Detect on the un-mirrored frame (a mirrored tag can't be decoded),
        then mirror the corners for drawing. Returns {id: corners (4x2)}."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        out = {}
        if ids is not None:
            w = frame_bgr.shape[1]
            for c, i in zip(corners, ids.flatten()):
                pts = c.reshape(4, 2).copy()
                pts[:, 0] = w - 1 - pts[:, 0]
                out[int(i)] = pts
        return out
