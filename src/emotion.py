"""
Facial expression (emotion) analysis on the tracked target's head crop.

Uses DeepFace's pretrained emotion model (7 categories: angry, disgust,
fear, happy, sad, surprise, neutral). This is a well-known-imperfect
classifier trained on FER2013 -- report it as a rough, informational signal
for a human operator to interpret, not a diagnostic or certified reading of
someone's mental/emotional state.

If `deepface` isn't installed (or fails to import for any reason, e.g. a
tensorflow install issue), this module disables itself gracefully instead
of crashing the rest of the pipeline.
"""

import numpy as np

from config import EMOTION_ANALYSIS_INTERVAL, EMOTION_MIN_FACE_SIZE_PX

_ = EMOTION_ANALYSIS_INTERVAL  # used by pipeline.py directly; imported here for discoverability


class EmotionAnalyzer:
    def __init__(self):
        self.available = True
        try:
            from deepface import DeepFace
            self._DeepFace = DeepFace
        except Exception as exc:
            print(
                f"[emotion] DeepFace not available ({exc}); emotion analysis "
                "disabled. Run `pip install deepface` to enable it."
            )
            self.available = False

    def analyze(self, head_crop_bgr: np.ndarray) -> str | None:
        """Returns the dominant emotion label (English), or None if
        unavailable / face too small / detection failed."""
        if not self.available or head_crop_bgr is None or head_crop_bgr.size == 0:
            return None

        h, w = head_crop_bgr.shape[:2]
        if min(h, w) < EMOTION_MIN_FACE_SIZE_PX:
            return None

        try:
            result = self._DeepFace.analyze(
                head_crop_bgr,
                actions=["emotion"],
                detector_backend="skip",  # we already cropped the face via pose
                                           # keypoints -- skip DeepFace's own face
                                           # detector entirely (avoids depending on
                                           # its OpenCV Haar-cascade backend, which
                                           # can be broken/missing on some installs)
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            return result.get("dominant_emotion")
        except Exception as exc:
            # Print (don't swallow silently) so failures are visible instead
            # of just showing up as "emotion never appears" with no clue why.
            print(f"[emotion] analyze() failed: {exc}")
            return None
