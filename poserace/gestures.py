"""Convert MediaPipe pose landmarks into per-wheel motor speeds.

Gesture scheme: each wrist's height relative to its own shoulder controls
that side's wheel speed, normalized by torso length so it works regardless
of how far the user stands from the camera.
"""

from dataclasses import dataclass

from mediapipe.tasks.python import vision

PoseLandmark = vision.PoseLandmark

# Wrist at shoulder height stays inside this band and produces zero speed,
# so a resting arm doesn't cause drift.
DEAD_ZONE = 0.15

# Wrist this many torso-heights above/below the shoulder maps to full speed.
FULL_SPEED_RANGE = 1.0

MAX_SPEED_PERCENT = 100

# Exponential moving average factor for smoothing landmark jitter frame to frame.
# Lower = smoother but laggier; higher = snappier but noisier.
SMOOTHING_ALPHA = 0.4


@dataclass
class WheelSpeeds:
    left: float
    right: float


def _arm_speed(shoulder_y: float, wrist_y: float, hip_y: float) -> float:
    """Map one arm's wrist height to a speed percentage in [-100, 100].

    Image y-coordinates increase downward, so a raised wrist has a smaller
    y than the shoulder.
    """
    torso_height = hip_y - shoulder_y
    if torso_height <= 1e-6:
        return 0.0

    raise_amount = (shoulder_y - wrist_y) / torso_height

    magnitude = (abs(raise_amount) - DEAD_ZONE) / (FULL_SPEED_RANGE - DEAD_ZONE)
    magnitude = max(0.0, min(1.0, magnitude))

    sign = 1.0 if raise_amount > 0 else -1.0
    return sign * magnitude * MAX_SPEED_PERCENT


class GestureController:
    """Tracks smoothed left/right wheel speeds across frames."""

    def __init__(self):
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0

    def update(self, landmarks, mirrored: bool) -> WheelSpeeds:
        """Compute new wheel speeds from one frame's pose landmarks.

        Parameters:
            landmarks: List of normalized landmarks from PoseLandmarker,
                indexed by PoseLandmark enum values.
            mirrored: Whether the source frame was flipped horizontally for
                a natural "mirror" display. MediaPipe's LEFT_*/RIGHT_*
                labels describe the image as captured, so when the frame is
                mirrored, the person's actual right arm is reported as
                MediaPipe's "left" side. Swap here so a raised right arm
                always drives the right wheel.
        """
        left_landmarks = (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_HIP)
        right_landmarks = (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_HIP)
        if mirrored:
            left_landmarks, right_landmarks = right_landmarks, left_landmarks

        left_shoulder, left_wrist, left_hip = (landmarks[i] for i in left_landmarks)
        right_shoulder, right_wrist, right_hip = (landmarks[i] for i in right_landmarks)

        target_left = _arm_speed(left_shoulder.y, left_wrist.y, left_hip.y)
        target_right = _arm_speed(right_shoulder.y, right_wrist.y, right_hip.y)

        self._smoothed_left += SMOOTHING_ALPHA * (target_left - self._smoothed_left)
        self._smoothed_right += SMOOTHING_ALPHA * (target_right - self._smoothed_right)

        return WheelSpeeds(left=self._smoothed_left, right=self._smoothed_right)

    def reset(self):
        self._smoothed_left = 0.0
        self._smoothed_right = 0.0
