"""
Pose analysis for the currently tracked target:
  - limb keypoint visibility (camera visibility, NOT a medical diagnosis)
  - whether an elongated object (stick, pole, flag) is present near either
    hand, via a geometric contour heuristic
  - sitting vs standing posture, from hip-knee-ankle angle
  - per-body-part bounding boxes (head / torso / each arm / each leg) --
    computed for internal use (e.g. cropping the head for emotion analysis)
    but NOT necessarily drawn on screen; that's the caller's choice

IMPORTANT SCOPE NOTE: "limb not visible" and posture/object signals are
statements about what the camera/model can currently see -- not verified
medical or threat assessments. Report them as visibility/geometry signals
for a human operator to interpret.

PERFORMANCE NOTE: this used to also run a full COCO-class object detector
on a crop around each wrist (to catch things like phones/bottles). That
added a whole extra YOLO inference call per hand per target per analysis
interval, which adds up fast with multiple targets. It's been removed --
the elongated-shape heuristic below (pure OpenCV contour analysis, no
model inference) covers the actual use case (detecting a carried stick/
pole/flag) at a fraction of the cost.
"""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    BODY_PART_BOX_PADDING_PX,
    HAND_OBJECT_FOREARM_EXCLUSION_MARGIN_PX,
    HAND_OBJECT_ROI_RADIUS_PX,
    POSE_KEYPOINT_CONF_THRESHOLD,
    POSE_LIMB_KEYPOINTS,
    POSE_MODEL,
    POSE_WRIST_KEYPOINTS,
    SITTING_KNEE_ANGLE_THRESHOLD_DEG,
    STICK_ASPECT_RATIO_THRESHOLD,
    STICK_MIN_LENGTH_PX,
)
from src.device import get_inference_device

# COCO-17 keypoint index groups used to build per-body-part boxes.
_BODY_PART_GROUPS = {
    "head": [0, 1, 2, 3, 4],          # nose, eyes, ears
    "torso": [5, 6, 11, 12],          # shoulders + hips
    "left_arm": [5, 7, 9],            # left shoulder, elbow, wrist
    "right_arm": [6, 8, 10],          # right shoulder, elbow, wrist
    "left_leg": [11, 13, 15],         # left hip, knee, ankle
    "right_leg": [12, 14, 16],        # right hip, knee, ankle
}

# (hip, knee, ankle) index triples for sitting-angle estimation.
_LEG_TRIPLES = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}

# COCO-17 elbow index for each wrist, used to compute forearm direction so
# the arm itself can be excluded from the hand-object check (see
# HAND_OBJECT_FOREARM_EXCLUSION_MARGIN_PX in config.py).
_ELBOW_KEYPOINT_FOR_WRIST = {"left_wrist": 7, "right_wrist": 8}


@dataclass
class PoseResult:
    keypoints_visible: dict = field(default_factory=dict)
    all_core_limbs_visible: bool = False
    objects_near_hands: list = field(default_factory=list)   # (hand_name, label, score)
    is_sitting: bool | None = None                           # None = couldn't estimate
    body_part_boxes: dict = field(default_factory=dict)       # name -> (x1, y1, x2, y2)


def _angle_at_vertex(a, b, c) -> float:
    """Angle at point b, formed by segments b->a and b->c, in degrees."""
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


class PoseAnalyzer:
    def __init__(self):
        self.pose_model = YOLO(POSE_MODEL)
        self.device, self.half = get_inference_device()
        self.pose_model.to(self.device)

    def analyze(self, frame_bgr: np.ndarray, target_box) -> PoseResult | None:
        x1, y1, x2, y2 = target_box
        pad = 10
        h, w = frame_bgr.shape[:2]
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        results = self.pose_model.predict(
            crop, device=self.device, half=self.half, verbose=False
        )[0]
        if results.keypoints is None or len(results.keypoints) == 0:
            return None

        kpts_xy = results.keypoints.xy[0].cpu().numpy()      # (17, 2), crop-local coords
        kpts_conf = results.keypoints.conf[0].cpu().numpy()  # (17,)
        offset = (cx1, cy1)

        result = PoseResult()

        for name, idx in POSE_LIMB_KEYPOINTS.items():
            result.keypoints_visible[name] = bool(kpts_conf[idx] >= POSE_KEYPOINT_CONF_THRESHOLD)
        result.all_core_limbs_visible = all(result.keypoints_visible.values())

        result.objects_near_hands = self._check_hands_for_elongated_objects(
            frame_bgr, kpts_xy, kpts_conf, offset
        )
        result.is_sitting = self._estimate_sitting(kpts_xy, kpts_conf)
        result.body_part_boxes = self._compute_body_part_boxes(
            frame_bgr.shape, kpts_xy, kpts_conf, offset
        )
        return result

    # ------------------------------------------------------------------
    # Hand-held elongated object detection (stick / pole / flag)
    # ------------------------------------------------------------------
    def _check_hands_for_elongated_objects(self, frame_bgr, kpts_xy, kpts_conf, offset):
        h, w = frame_bgr.shape[:2]
        ox, oy = offset
        findings = []

        for hand_name, wrist_idx in POSE_WRIST_KEYPOINTS.items():
            if kpts_conf[wrist_idx] < POSE_KEYPOINT_CONF_THRESHOLD:
                continue

            elbow_idx = _ELBOW_KEYPOINT_FOR_WRIST[hand_name]
            if kpts_conf[elbow_idx] < POSE_KEYPOINT_CONF_THRESHOLD:
                # Can't tell which direction the forearm is coming from, so
                # we can't safely exclude it -- skip rather than risk
                # flagging the person's own arm as a held object.
                continue

            wx, wy = kpts_xy[wrist_idx]
            ex, ey = kpts_xy[elbow_idx]
            forearm_dir = np.array([wx - ex, wy - ey], dtype=np.float64)
            norm = np.linalg.norm(forearm_dir)
            if norm < 1e-3:
                continue
            forearm_dir /= norm

            wx, wy = wx + ox, wy + oy

            r = HAND_OBJECT_ROI_RADIUS_PX
            rx1, ry1 = int(max(0, wx - r)), int(max(0, wy - r))
            rx2, ry2 = int(min(w, wx + r)), int(min(h, wy + r))
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            wrist_local = (wx - rx1, wy - ry1)
            score = self._detect_elongated_shape(roi, wrist_local, forearm_dir)
            if score is not None:
                findings.append((hand_name, "elongated object (stick/pole/flag)", score))

        return findings

    @staticmethod
    def _detect_elongated_shape(roi_bgr, wrist_local=None, forearm_dir=None) -> float | None:
        """
        Returns a pseudo-confidence (0-1) if an elongated contour is found
        in the ROI, else None. Pure OpenCV contour geometry, no model
        inference -- flags SHAPE only, not object identity.

        wrist_local/forearm_dir (optional): when given, a candidate shape
        is only accepted if it sits on the far side of the wrist, AWAY from
        the elbow -- this is what a genuinely held object (extending past
        the hand) looks like. A shape on the near side (toward the elbow)
        is almost always the person's own forearm/sleeve edge, which is
        elongated by definition and would otherwise false-positive on every
        single frame regardless of whether anything is actually held.
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best_score = None
        for c in contours:
            if cv2.contourArea(c) < 15:
                continue
            (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
            if rw == 0 or rh == 0:
                continue
            long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-3)
            aspect_ratio = long_side / short_side

            if long_side < STICK_MIN_LENGTH_PX or aspect_ratio < STICK_ASPECT_RATIO_THRESHOLD:
                continue

            if wrist_local is not None and forearm_dir is not None:
                centroid_vec = np.array([cx - wrist_local[0], cy - wrist_local[1]])
                if np.dot(centroid_vec, forearm_dir) < HAND_OBJECT_FOREARM_EXCLUSION_MARGIN_PX:
                    continue  # sits toward the elbow -- this is the arm, not an object

            score = min(1.0, aspect_ratio / (STICK_ASPECT_RATIO_THRESHOLD * 2))
            if best_score is None or score > best_score:
                best_score = score

        return best_score

    # ------------------------------------------------------------------
    # Sitting / standing estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_sitting(kpts_xy, kpts_conf) -> bool | None:
        angles = []
        for hip_i, knee_i, ankle_i in _LEG_TRIPLES.values():
            if (kpts_conf[hip_i] < POSE_KEYPOINT_CONF_THRESHOLD or
                    kpts_conf[knee_i] < POSE_KEYPOINT_CONF_THRESHOLD or
                    kpts_conf[ankle_i] < POSE_KEYPOINT_CONF_THRESHOLD):
                continue
            angle = _angle_at_vertex(kpts_xy[hip_i], kpts_xy[knee_i], kpts_xy[ankle_i])
            angles.append(angle)

        if not angles:
            return None  # not enough visible leg keypoints to judge

        avg_angle = sum(angles) / len(angles)
        return avg_angle < SITTING_KNEE_ANGLE_THRESHOLD_DEG

    # ------------------------------------------------------------------
    # Per-body-part bounding boxes (used internally, e.g. head crop for
    # emotion analysis -- not necessarily drawn on screen)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_body_part_boxes(frame_shape, kpts_xy, kpts_conf, offset):
        h, w = frame_shape[:2]
        ox, oy = offset
        pad = BODY_PART_BOX_PADDING_PX
        boxes = {}

        for part_name, indices in _BODY_PART_GROUPS.items():
            pts = [
                (kpts_xy[i][0] + ox, kpts_xy[i][1] + oy)
                for i in indices
                if kpts_conf[i] >= POSE_KEYPOINT_CONF_THRESHOLD
            ]
            if not pts:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = int(max(0, min(xs) - pad))
            y1 = int(max(0, min(ys) - pad))
            x2 = int(min(w, max(xs) + pad))
            y2 = int(min(h, max(ys) + pad))
            if x2 > x1 and y2 > y1:
                boxes[part_name] = (x1, y1, x2, y2)

        return boxes
