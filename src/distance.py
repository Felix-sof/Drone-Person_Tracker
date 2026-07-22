"""
Rough monocular distance estimation from a target's apparent pixel height.

Pinhole camera approximation: an object's apparent height in pixels is
inversely proportional to its distance from the camera. Given the frame's
vertical field of view and an assumed real-world height for the target, we
can invert that relationship to estimate distance:

    distance = (real_height * frame_height_px) / (2 * box_height_px * tan(VFOV/2))

Accuracy depends entirely on two assumptions holding:
  1. CAMERA_VERTICAL_FOV_DEG actually matches the camera in use.
  2. The target is standing upright and fully visible in the box -- someone
     crouching, sitting, or partially occluded has a SMALLER apparent
     height for reasons unrelated to distance, which this method has no
     way to distinguish from "farther away".

Treat the output as a rough estimate for situational awareness (e.g.
"roughly 15m out"), not a survey-grade measurement.
"""

import math

from config import ASSUMED_PERSON_HEIGHT_M, CAMERA_VERTICAL_FOV_DEG


def estimate_distance_m(box_height_px: float, frame_height_px: int) -> float | None:
    """
    Args:
        box_height_px: the tracked target's bounding box height, in pixels.
        frame_height_px: the full frame's height, in pixels.

    Returns:
        Estimated distance in meters, or None if the inputs don't allow a
        sane estimate (e.g. a degenerate zero-height box).
    """
    if box_height_px <= 0 or frame_height_px <= 0:
        return None

    vfov_rad = math.radians(CAMERA_VERTICAL_FOV_DEG)
    denom = 2 * box_height_px * math.tan(vfov_rad / 2)
    if denom <= 0:
        return None

    return (ASSUMED_PERSON_HEIGHT_M * frame_height_px) / denom