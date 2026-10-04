"""
Per-target state for multi-target tracking.

Each tracked person gets their OWN identity: their own gallery of reference
embeddings, their own IOU tracker for frame-to-frame continuity, and their
own activity/posture state -- so multiple people can be tracked at once
without being confused for one another, and the same physical person can
never end up double-counted as two different targets (see
DUPLICATE_TARGET_SIMILARITY_THRESHOLD in config.py, checked when a target
is first created).

Lifecycle:
  - MANUAL targets (reference photo / 'n' key) are confirmed immediately
    and never expire -- in a search-and-rescue run, the person you were
    asked to find must not silently disappear from the list just because
    they've been out of view for a while.
  - AUTO targets (ENABLE_AUTO_TRACK_ALL) start out TENTATIVE with a
    temporary negative ID and are only promoted (given a real, visible ID)
    after AUTO_TARGET_MIN_HITS consecutive matches. A one-frame false
    positive (a bush, a shadow) therefore never gets an ID, a HUD panel,
    or a DB row. Confirmed auto targets that stay lost for
    LOST_TARGET_EXPIRY_FRAMES are dropped so they stop occupying a slot
    under MAX_CONCURRENT_TARGETS.
"""

import colorsys
from collections import deque

from config import TRACKER_IOU_THRESHOLD, TRAIL_LENGTH
from src.activity import ActivityClassifier
from src.posture import PostureTracker
from src.tracker import SingleTargetTracker, transform_point

# BGR colors for the first few targets -- each one visually distinct.
_COLOR_PALETTE = [
    (0, 255, 0),      # green
    (0, 165, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 0, 255),      # red
    (255, 128, 0),    # blue
]
_TENTATIVE_COLOR = (140, 140, 140)
_TRAIL_BREAK_HEIGHTS = 1.5


def color_for_id(target_id: int) -> tuple:
    """Stable, distinct BGR color per target ID. Beyond the hand-picked
    palette, hues are spaced by the golden ratio so consecutive IDs never
    land on near-identical colors (with MAX_CONCURRENT_TARGETS=20, a plain
    modulo over 6 colors would give several targets the exact same color)."""
    if target_id <= 0:
        return _TENTATIVE_COLOR
    if target_id <= len(_COLOR_PALETTE):
        return _COLOR_PALETTE[target_id - 1]
    hue = (target_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class Target:
    def __init__(self, target_id: int, initial_embedding, manual: bool = True):
        self.id = target_id
        self.gallery = [initial_embedding]
        self.manual = manual
        self.confirmed = manual
        self.color = color_for_id(target_id)

        self.tracker = SingleTargetTracker(iou_threshold=TRACKER_IOU_THRESHOLD)
        self.activity_classifier = ActivityClassifier()
        self.posture_tracker = PostureTracker()

        self.last_pose_result = None
        self.last_emotion = None
        self.locked_score = None
        self.last_distance_m = None

        # Processed frames since this target was last matched to a
        # detection (0 = matched this frame). Separate from the tracker's
        # own missed_frames, which resets once the tracker gives up.
        self.frames_since_seen = 0
        self.total_frames_seen = 0
        # Recent ground-contact points (bottom-center of the box), drawn
        # as a fading trajectory trail.
        self.trail = deque(maxlen=max(TRAIL_LENGTH, 1))

    def confirm(self, new_id: int) -> None:
        """Promote a tentative auto target to a real, visible identity."""
        self.id = new_id
        self.confirmed = True
        self.color = color_for_id(new_id)

    def record_seen(self, box) -> None:
        self.frames_since_seen = 0
        self.total_frames_seen += 1
        x1, y1, x2, y2 = box
        point = ((x1 + x2) // 2, y2)
        # A jump much larger than the person's own height between two
        # consecutive trail points isn't walking -- it's a Re-ID re-lock or
        # a motion-compensation re-anchor. Start a fresh trail instead of
        # drawing a line straight across the frame.
        if self.trail:
            lx, ly = self.trail[-1]
            if (point[0] - lx) ** 2 + (point[1] - ly) ** 2 > (_TRAIL_BREAK_HEIGHTS * max(y2 - y1, 1)) ** 2:
                self.trail.clear()
        self.trail.append(point)

    def apply_camera_motion(self, affine) -> None:
        """Move everything this target remembers in pixel coordinates
        (tracker state, trail, speed history) along with the camera."""
        self.tracker.apply_camera_motion(affine)
        self.activity_classifier.apply_camera_motion(affine)
        if self.trail:
            self.trail = deque(
                (tuple(int(round(v)) for v in transform_point(p, affine)) for p in self.trail),
                maxlen=self.trail.maxlen,
            )

    def record_missed(self) -> None:
        self.frames_since_seen += 1

    def reset_analysis_state(self) -> None:
        self.last_pose_result = None
        self.last_emotion = None
        self.locked_score = None
        self.last_distance_m = None
        self.activity_classifier.reset()
        self.posture_tracker.reset()
        self.trail.clear()
