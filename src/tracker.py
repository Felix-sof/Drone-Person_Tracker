"""
Lightweight single-target tracker for frame-to-frame continuity.

Once Re-ID has confirmed WHICH detection is our target in a given frame,
we don't want to run the (relatively expensive) embedding comparison every
single frame. Instead we track the confirmed box forward using IOU
association against the new frame's detections. If tracking is lost
(no overlapping detection for too many frames), the pipeline falls back to
a full Re-ID rescan to recover the target.

Motion prediction: matching against the LAST seen box breaks down as soon
as the target moves more than roughly its own width between two processed
frames -- a running person, a drone pass, or simply VIDEO_PROCESS_EVERY_N_FRAMES
skipping frames. A constant-velocity Kalman filter (the same idea SORT /
DeepSORT / ByteTrack use) predicts where the box SHOULD be this frame, and
IOU is computed against that prediction instead. It also lets a target
"coast" through a few frames of missed detections (brief occlusion, motion
blur) along its predicted path instead of being dropped immediately.
"""

import numpy as np

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


def iou_matrix(boxes_a: list, boxes_b: list) -> np.ndarray:
    """Pairwise IOU, shape (len(boxes_a), len(boxes_b)). Vectorized so the
    global assignment step stays cheap even with a crowd in frame."""
    if not boxes_a or not boxes_b:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]
    inter_w = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    inter_h = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = inter_w * inter_h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def affine_scale(affine) -> float:
    """Uniform scale of a rotation+scale+translation (partial) affine."""
    return float(np.sqrt(abs(np.linalg.det(np.asarray(affine, dtype=np.float64)[:, :2]))))


def transform_point(point, affine) -> tuple:
    a = np.asarray(affine, dtype=np.float64)
    x, y = point
    return (float(a[0, 0] * x + a[0, 1] * y + a[0, 2]),
            float(a[1, 0] * x + a[1, 1] * y + a[1, 2]))


def transform_box(box, affine) -> tuple:
    """Moves a box with the camera: its center follows the affine, its size
    follows the affine's scale (an axis-aligned box can't rotate, and for
    the few degrees a drone rotates between frames that's negligible)."""
    x1, y1, x2, y2 = box
    cx, cy = transform_point(((x1 + x2) / 2.0, (y1 + y2) / 2.0), affine)
    s = affine_scale(affine)
    hw, hh = (x2 - x1) * s / 2.0, (y2 - y1) * s / 2.0
    return (int(round(cx - hw)), int(round(cy - hh)), int(round(cx + hw)), int(round(cy + hh)))


class KalmanBoxFilter:
    """
    Constant-velocity Kalman filter over a box's (cx, cy, w, h).

    State: [cx, cy, w, h, vcx, vcy, vw, vh], one time step = one PROCESSED
    frame. Process/measurement noise are scaled by the box's own height
    (the DeepSORT convention): a 300px-tall close-up person and a 20px-tall
    person seen from altitude jitter by very different pixel amounts, and a
    fixed pixel noise would be wrong for at least one of them.
    """

    _STD_WEIGHT_POSITION = 1.0 / 20
    _STD_WEIGHT_VELOCITY = 1.0 / 160
    # Initial velocity uncertainty, in box-heights per frame. DeepSORT's
    # default (10 * 1/160 ~= 0.06) assumes a near-static camera at 30fps;
    # drone footage with frame skipping routinely sees a box move half its
    # own height (or more) between processed frames, and with a too-tight
    # prior the filter needs many frames to believe that velocity.
    _INITIAL_STD_VELOCITY = 0.5

    def __init__(self, box):
        self._F = np.eye(8)
        self._F[:4, 4:] = np.eye(4)
        self._H = np.eye(4, 8)

        self.x = np.zeros(8)
        self.x[:4] = self._box_to_z(box)
        h = max(self.x[3], 1.0)
        std = np.array([2 * self._STD_WEIGHT_POSITION * h] * 4
                       + [self._INITIAL_STD_VELOCITY * h] * 2
                       + [10 * self._STD_WEIGHT_VELOCITY * h] * 2)
        self.P = np.diag(std ** 2)

    @staticmethod
    def _box_to_z(box) -> np.ndarray:
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0,
                         max(x2 - x1, 1.0), max(y2 - y1, 1.0)])

    def box(self) -> tuple:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return (int(round(cx - w / 2)), int(round(cy - h / 2)),
                int(round(cx + w / 2)), int(round(cy + h / 2)))

    def predict(self) -> tuple:
        h = max(self.x[3], 1.0)
        std = np.array([self._STD_WEIGHT_POSITION * h] * 4
                       + [self._STD_WEIGHT_VELOCITY * h] * 4)
        self.x = self._F @ self.x
        # A shrinking box can be extrapolated through zero size during a
        # long coast -- clamp so the box stays a valid rectangle.
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.P = self._F @ self.P @ self._F.T + np.diag(std ** 2)
        return self.box()

    def update(self, box) -> None:
        z = self._box_to_z(box)
        h = max(z[3], 1.0)
        R = np.diag((np.array([self._STD_WEIGHT_POSITION * h] * 4)) ** 2)
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self._H @ self.x)
        self.P = (np.eye(8) - K @ self._H) @ self.P

    def apply_affine(self, affine) -> None:
        """
        Camera-motion compensation (BoT-SORT-style GMC): re-express the
        state in the NEW frame's coordinates. Position gets the full affine,
        velocity only its rotation+scale (a velocity is a direction, not a
        location), and box size the uniform scale. Without this, the
        drone's own motion shows up as target velocity -- the filter then
        "predicts" people sliding across the frame with the camera.
        """
        a = np.asarray(affine, dtype=np.float64)
        R = a[:, :2]
        t = a[:, 2]
        s = affine_scale(a)
        M = np.zeros((8, 8))
        M[0:2, 0:2] = R
        M[2:4, 2:4] = np.eye(2) * s
        M[4:6, 4:6] = R
        M[6:8, 6:8] = np.eye(2) * s
        self.x = M @ self.x
        self.x[0:2] += t
        self.P = M @ self.P @ M.T

    @property
    def velocity(self) -> tuple:
        """(vcx, vcy) in pixels per processed frame."""
        return float(self.x[4]), float(self.x[5])


class SingleTargetTracker:
    """
    State machine with three effective modes:
      - LOCKED + matched: we have a confirmed box this frame.
      - LOCKED + coasting: missed a few frames, still following the Kalman
        prediction (missed_frames > 0 but <= max_missed_frames).
      - LOST: no reliable match for too long; pipeline should trigger a
        full Re-ID rescan.

    Per-frame protocol (the pipeline drives this so it can do a GLOBAL
    assignment across all targets in between):
        predict() -> [assignment] -> mark_matched(box) or mark_missed()
    """

    def __init__(self, iou_threshold: float = 0.3,
                 max_missed_frames: int = TRACKER_MAX_MISSED_FRAMES):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.current_box = None     # last CONFIRMED (measured) box
        self.predicted_box = None   # Kalman prediction for the current frame
        self.missed_frames = 0
        self.hits = 0               # consecutive confirmed frames since (re)initialization
        self.is_locked = False
        self._kf: KalmanBoxFilter | None = None

    def initialize(self, box):
        self._kf = KalmanBoxFilter(box)
        self.current_box = box
        self.predicted_box = box
        self.missed_frames = 0
        self.hits = 1
        self.is_locked = True

    def predict(self):
        """Advance the motion model one step; call once per processed frame
        BEFORE association. Returns the predicted box (or None if lost)."""
        if not self.is_locked or self._kf is None:
            self.predicted_box = None
            return None
        self.predicted_box = self._kf.predict()
        return self.predicted_box

    def apply_camera_motion(self, affine) -> None:
        """Shift this track's state into the current frame's coordinates
        (call BEFORE predict())."""
        if self._kf is not None:
            self._kf.apply_affine(affine)
        if self.current_box is not None:
            self.current_box = transform_box(self.current_box, affine)
        if self.predicted_box is not None:
            self.predicted_box = transform_box(self.predicted_box, affine)

    @property
    def search_box(self):
        """The box new detections should be compared against this frame."""
        return self.predicted_box if self.predicted_box is not None else self.current_box

    def mark_matched(self, box):
        if self._kf is None:
            self.initialize(box)
            return
        self._kf.update(box)
        self.current_box = box
        self.missed_frames = 0
        self.hits += 1

    def mark_missed(self):
        self.missed_frames += 1
        self.hits = 0
        if self.missed_frames > self.max_missed_frames:
            self.lose()

    def lose(self):
        self.is_locked = False
        self.current_box = None
        self.predicted_box = None
        self.missed_frames = 0
        self.hits = 0
        self._kf = None

    @property
    def velocity(self) -> tuple | None:
        return self._kf.velocity if self._kf is not None else None

    def update(self, detections: list):
        """
        Standalone single-target convenience wrapper (predict + greedy IOU
        match + bookkeeping). The multi-target pipeline does NOT use this --
        it calls predict()/mark_matched()/mark_missed() directly so it can
        run a global assignment across all targets in between.

        Args:
            detections: list of Detection objects for the current frame.

        Returns:
            matched_box or None
        """
        if not self.is_locked or self.current_box is None:
            return None

        reference = self.predict()
        best_iou, best_box = 0.0, None
        for det in detections:
            score = iou(reference, det.box)
            if score > best_iou:
                best_iou, best_box = score, det.box

        if best_iou >= self.iou_threshold:
            self.mark_matched(best_box)
            return best_box

        self.mark_missed()
        return None

    def lost(self) -> bool:
        return not self.is_locked
