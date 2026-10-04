"""
Movement state classification: "still" / "walking" / "running", based on
how fast the tracked box moves, expressed as a RATIO of the target's own
current box height (per second) rather than raw pixels per second.

Why ratio and not raw pixels: a monocular camera has no depth/scale
reference, so the same physical running speed produces very different raw
pixel-speeds depending on how far the subject is from the camera. Box
height itself shrinks/grows with distance in roughly the same proportion,
so normalizing by it cancels most of that distance dependence.

TWO kinds of motion are combined, because a 2D bounding box only directly
sees ONE of them:
  - LATERAL motion: the box's center moving across the frame (someone
    running left-to-right, or diagonally). This is what a naive "center
    displacement" metric alone captures.
  - RADIAL motion: someone running STRAIGHT TOWARD or AWAY FROM the
    camera. Their box center barely moves in x/y at all -- only the box's
    SIZE changes (it grows as they approach, shrinks as they recede). A
    lateral-only metric would misread a person sprinting directly at the
    camera as "still", which is exactly backwards. We use the box height's
    rate of change as a proxy for this and combine it with lateral speed
    (take whichever signal is stronger) so radial motion isn't invisible.

Still a heuristic on 2D pixel geometry, not a calibrated real-world speed
measurement -- but this covers the two motion directions that actually
occur in typical footage (someone moving purely perpendicular to both the
lateral and depth axis simultaneously with zero net effect on either
signal is a measure-zero edge case in practice).
"""

import time
from collections import deque

from config import (
    ACTIVITY_HISTORY_SECONDS,
    ACTIVITY_RADIAL_SPEED_SCALE,
    ACTIVITY_RUN_THRESHOLD_HEIGHTS_S,
    ACTIVITY_WALK_THRESHOLD_HEIGHTS_S,
)


class ActivityClassifier:
    def __init__(self):
        # Each entry: (timestamp, center_x, center_y, box_height)
        self._history = deque()

    def reset(self):
        self._history.clear()

    def apply_camera_motion(self, affine) -> None:
        """Re-express the stored positions in the current frame's
        coordinates, so a moving drone's own motion isn't measured as the
        target walking/running (a person standing still under a drone that
        flies past would otherwise read as "running")."""
        if not self._history:
            return
        a = affine
        s = float(abs(a[0][0] * a[1][1] - a[0][1] * a[1][0])) ** 0.5
        self._history = deque(
            (t, a[0][0] * x + a[0][1] * y + a[0][2], a[1][0] * x + a[1][1] * y + a[1][2], h * s)
            for t, x, y, h in self._history
        )

    def update(self, box, timestamp: float | None = None) -> str:
        """
        Args:
            box: (x1, y1, x2, y2) of the currently tracked target, or None
                 if the target isn't locked this frame.
            timestamp: when this frame was captured, in seconds. For video
                 files pass the VIDEO's own timestamp, not wall-clock time:
                 if analysis runs slower than real time, wall-clock seconds
                 stretch while the target's on-screen displacement doesn't,
                 so a runner would read as "walking". None -> wall clock
                 (correct for live sources).

        Returns:
            One of "unknown" (not enough data), "still", "walking", "running".
        """
        now = time.monotonic() if timestamp is None else float(timestamp)

        if box is None:
            self.reset()
            return "unknown"

        # Time went backwards (looping video, seek, switching clock source):
        # old history is meaningless relative to the new timeline.
        if self._history and now < self._history[-1][0]:
            self.reset()

        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        height = max(y2 - y1, 1.0)
        self._history.append((now, cx, cy, height))

        cutoff = now - ACTIVITY_HISTORY_SECONDS
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 2:
            return "unknown"

        t0, x0, y0, h0 = self._history[0]
        t1, x1_, y1_, h1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return "unknown"

        # Lateral: center displacement, normalized by current box height.
        lateral_dist_px = ((x1_ - x0) ** 2 + (y1_ - y0) ** 2) ** 0.5
        lateral_speed = (lateral_dist_px / h1) / dt

        # Radial: fractional change in box height -- catches someone
        # running straight toward/away from the camera, which lateral
        # motion alone is blind to.
        height_change_ratio = abs(h1 - h0) / h1
        radial_speed = (height_change_ratio / dt) * ACTIVITY_RADIAL_SPEED_SCALE

        speed_heights_s = max(lateral_speed, radial_speed)

        if speed_heights_s >= ACTIVITY_RUN_THRESHOLD_HEIGHTS_S:
            return "running"
        if speed_heights_s >= ACTIVITY_WALK_THRESHOLD_HEIGHTS_S:
            return "walking"
        return "still"