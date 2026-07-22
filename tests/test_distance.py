import math
import pytest

from src.distance import estimate_distance_m


def test_estimate_distance_returns_none_for_zero_box_height():
    assert estimate_distance_m(box_height_px=0, frame_height_px=1080) is None


def test_estimate_distance_returns_none_for_negative_box_height():
    assert estimate_distance_m(box_height_px=-10, frame_height_px=1080) is None


def test_estimate_distance_returns_none_for_zero_frame_height():
    assert estimate_distance_m(box_height_px=100, frame_height_px=0) is None


def test_estimate_distance_known_value():
    # Manually derived reference case:
    # ASSUMED_PERSON_HEIGHT_M=1.7, CAMERA_VERTICAL_FOV_DEG=55.0 (config defaults)
    # frame_height_px=1080, box_height_px=200
    frame_height_px = 1080
    box_height_px = 200
    vfov_rad = math.radians(55.0)
    expected = (1.7 * frame_height_px) / (2 * box_height_px * math.tan(vfov_rad / 2))

    result = estimate_distance_m(box_height_px, frame_height_px)
    assert result == pytest.approx(expected, rel=1e-6)


def test_estimate_distance_decreases_as_box_grows():
    # A larger box height means the person appears bigger -> closer.
    frame_height_px = 1080
    far = estimate_distance_m(box_height_px=50, frame_height_px=frame_height_px)
    near = estimate_distance_m(box_height_px=400, frame_height_px=frame_height_px)
    assert near < far


def test_estimate_distance_scales_with_frame_height():
    # Same box height, but a taller frame (more pixels per real-world meter)
    # should produce a larger reported distance -- the box occupies a
    # smaller vertical FRACTION of a taller frame.
    box_height_px = 200
    small_frame = estimate_distance_m(box_height_px, frame_height_px=720)
    large_frame = estimate_distance_m(box_height_px, frame_height_px=1440)
    assert large_frame > small_frame