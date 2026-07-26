"""
Lightweight single-target tracker for frame-to-frame continuity.

Once Re-ID has confirmed WHICH detection is our target in a given frame,
we don't want to run the (relatively expensive) embedding comparison every
single frame. Instead we track the confirmed box forward using simple IOU
association against the new frame's detections. If tracking is lost
(no overlapping detection for too many frames), the pipeline falls back to
a full Re-ID rescan to recover the target.
"""

from config import TRACKER_MAX_MISSED_FRAMES


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


class SingleTargetTracker:
    """
    State machine with two modes:
      - LOCKED: we have a confirmed box, look for the best-overlapping
        detection each frame to carry it forward.
      - LOST: no reliable match recently; pipeline should trigger a
        full Re-ID rescan.
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.current_box = None
        self.missed_frames = 0
        self.is_locked = False

    def initialize(self, box):
        self.current_box = box
        self.missed_frames = 0
        self.is_locked = True

    def update(self, detections: list):
        """
        Args:
            detections: list of Detection objects for the current frame.

        Returns:
            matched_box or None
        """
        if not self.is_locked or self.current_box is None:
            return None

        best_iou, best_box = 0.0, None
        for det in detections:
            score = iou(self.current_box, det.box)
            if score > best_iou:
                best_iou, best_box = score, det.box

        if best_iou >= self.iou_threshold:
            self.current_box = best_box
            self.missed_frames = 0
            return best_box

        self.missed_frames += 1
        if self.missed_frames > TRACKER_MAX_MISSED_FRAMES:
            self.is_locked = False
            self.current_box = None
        return None

    def lost(self) -> bool:
        return not self.is_locked
