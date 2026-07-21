"""
Per-target state for multi-target tracking.

Each tracked person gets their OWN identity: their own gallery of reference
embeddings, their own IOU tracker for frame-to-frame continuity, and their
own activity/posture state -- so multiple people can be tracked at once
without being confused for one another, and the same physical person can
never end up double-counted as two different targets (see
DUPLICATE_TARGET_SIMILARITY_THRESHOLD in config.py, checked when a target
is first created).
"""

from config import TRACKER_IOU_THRESHOLD
from src.activity import ActivityClassifier
from src.posture import PostureTracker
from src.tracker import SingleTargetTracker

# BGR colors, cycled through as new targets are created so each one is
# visually distinguishable on screen.
_COLOR_PALETTE = [
    (0, 255, 0),      # green
    (0, 165, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 0, 255),      # red
    (255, 128, 0),    # blue
]


class Target:
    def __init__(self, target_id: int, initial_embedding):
        self.id = target_id
        self.gallery = [initial_embedding]
        self.color = _COLOR_PALETTE[target_id % len(_COLOR_PALETTE)]

        self.tracker = SingleTargetTracker(iou_threshold=TRACKER_IOU_THRESHOLD)
        self.activity_classifier = ActivityClassifier()
        self.posture_tracker = PostureTracker()

        self.last_pose_result = None
        self.last_emotion = None
        self.locked_score = None
