"""
Tracks sitting/standing transitions over time so a brief "rising" label
can be shown right after someone goes from sitting to standing, instead
of the state just silently flipping in one frame.
"""

from config import POSTURE_TRANSITION_DISPLAY_FRAMES


class PostureTracker:
    def __init__(self):
        self._prev_sitting = None
        self._transition_frames_left = 0

    def reset(self):
        self._prev_sitting = None
        self._transition_frames_left = 0

    def update(self, is_sitting: bool | None) -> str | None:
        """
        Args:
            is_sitting: True/False from pose analysis, or None if unknown
                        this frame (e.g. pose analysis didn't run / no legs
                        visible).

        Returns:
            "sitting", "rising", or None (meaning: defer to the speed-based
            activity classifier for a walking/running/still label instead).
        """
        if is_sitting is None:
            # Don't reset state on a single missed frame -- pose analysis
            # only runs every few frames (see POSE_ANALYSIS_INTERVAL).
            if self._transition_frames_left > 0:
                self._transition_frames_left -= 1
                return "rising"
            return None

        if self._prev_sitting is True and is_sitting is False:
            self._transition_frames_left = POSTURE_TRANSITION_DISPLAY_FRAMES

        self._prev_sitting = is_sitting

        if self._transition_frames_left > 0:
            self._transition_frames_left -= 1
            return "rising"

        return "sitting" if is_sitting else None
