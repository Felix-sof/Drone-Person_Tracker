"""
Tests for the elongated-shape hand-object heuristic in
src/pose_analysis.py -- specifically the forearm-exclusion fix.

Regression context: before this fix, the heuristic false-positived on the
person's OWN forearm/sleeve silhouette (an elbow-to-wrist edge is itself
long and high-aspect-ratio) even with an empty hand. The fix uses the
elbow->wrist direction to reject any elongated shape sitting on the
elbow side of the wrist, and only accepts one extending past the wrist,
away from the body.

These call the staticmethod directly (no PoseAnalyzer() instantiation
needed), so no YOLO model gets loaded/downloaded to run these tests.
"""

import numpy as np
import cv2

from src.pose_analysis import PoseAnalyzer


def _roi_with_rect(rect_xyxy):
    roi = np.zeros((90, 90, 3), dtype=np.uint8)
    x1, y1, x2, y2 = rect_xyxy
    cv2.rectangle(roi, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return roi


def test_forearm_silhouette_is_not_flagged_as_object():
    """An elongated shape sitting on the ELBOW side of the wrist (i.e. the
    person's own forearm/sleeve edge) must NOT be reported as a held
    object -- this was the actual false-positive bug."""
    wrist_local = (45, 45)
    forearm_dir = np.array([1.0, 0.0])  # elbow -> wrist points rightward,
    # so the elbow itself sits to the LEFT of the wrist.

    roi = _roi_with_rect((0, 40, 44, 50))  # shape to the LEFT of the wrist
    score = PoseAnalyzer._detect_elongated_shape(roi, wrist_local, forearm_dir)
    assert score is None


def test_object_extending_past_hand_is_still_flagged():
    """An elongated shape extending AWAY from the elbow (past the hand --
    where a genuinely held stick/pole would be) must still be detected;
    the forearm-exclusion fix must not blind the check entirely."""
    wrist_local = (45, 45)
    forearm_dir = np.array([1.0, 0.0])

    roi = _roi_with_rect((46, 40, 89, 50))  # shape to the RIGHT of the wrist
    score = PoseAnalyzer._detect_elongated_shape(roi, wrist_local, forearm_dir)
    assert score is not None
    assert 0.0 < score <= 1.0


def test_no_wrist_info_falls_back_to_unfiltered_behavior():
    """Without wrist_local/forearm_dir (e.g. elbow keypoint wasn't
    confidently visible), the check still works -- just without the
    directional filter -- rather than crashing."""
    roi = _roi_with_rect((10, 40, 80, 50))
    score = PoseAnalyzer._detect_elongated_shape(roi)
    assert score is not None


def test_short_or_blob_shapes_are_never_flagged_either_way():
    """A roughly-square/short shape shouldn't be flagged regardless of
    which side of the wrist it's on -- sanity check that the aspect-ratio/
    min-length gates from before this change still apply first."""
    wrist_local = (45, 45)
    forearm_dir = np.array([1.0, 0.0])
    roi = _roi_with_rect((40, 40, 50, 50))  # small square blob
    score = PoseAnalyzer._detect_elongated_shape(roi, wrist_local, forearm_dir)
    assert score is None
