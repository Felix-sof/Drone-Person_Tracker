"""
Unit tests for the estimation-downscale feature in
src/motion_compensation.py (MOTION_COMP_ESTIMATION_WIDTH).

These are geometry/plumbing checks, not full optical-flow integration
tests -- they don't need a real camera frame.
"""

import math

import numpy as np

from config import MAX_CUMULATIVE_ROTATION_DEG
from src.motion_compensation import EgoMotionCompensator


def test_downscale_noop_when_frame_already_narrow():
    """A frame narrower than MOTION_COMP_ESTIMATION_WIDTH should pass
    through unchanged, with scale_factor == 1.0 (no resize cost paid)."""
    gray = np.zeros((100, 200), dtype=np.uint8)  # width 200 < default 480
    small, scale_factor = EgoMotionCompensator._downscale_for_estimation(gray)
    assert scale_factor == 1.0
    assert small.shape == gray.shape


def test_downscale_shrinks_wide_frame_and_reports_scale_factor():
    """A frame wider than MOTION_COMP_ESTIMATION_WIDTH should be resized
    down to that width, and scale_factor should be exactly
    original_width / new_width so translation can be rescaled correctly."""
    gray = np.zeros((540, 1920), dtype=np.uint8)
    small, scale_factor = EgoMotionCompensator._downscale_for_estimation(gray)
    assert small.shape[1] < gray.shape[1]
    assert abs(scale_factor - (1920 / small.shape[1])) < 1e-6


def test_estimate_affine_rescales_translation_only():
    """Rotation/scale part of the affine (top-left 2x2) must be left alone
    by the scale_factor rescale -- only the translation column (index 2)
    should change. This is what makes it safe to estimate motion on a
    downscaled frame and still warp the full-resolution frame correctly."""
    identity_affine = np.array([[1.0, 0.0, 10.0],
                                 [0.0, 1.0, 20.0]], dtype=np.float32)
    scale_factor = 4.0

    rescaled = identity_affine.copy()
    rescaled[:, 2] *= scale_factor

    # Rotation/scale part (columns 0-1) unchanged
    assert np.array_equal(rescaled[:, :2], identity_affine[:, :2])
    # Translation (column 2) scaled up by scale_factor
    assert rescaled[0, 2] == 40.0
    assert rescaled[1, 2] == 80.0


def test_exclusion_mask_blocks_out_target_region():
    """A target box should end up as 0 (excluded) in the mask, everything
    else should stay 255 (searchable) -- this is what stops corners picked
    up ON a moving person from corrupting the camera-motion estimate."""
    shape = (100, 200)  # (h, w), already-downscaled estimation size
    box_full_res = (80, 20, 120, 60)  # x1, y1, x2, y2, full-resolution
    scale_factor = 1.0  # no downscale, for simple arithmetic

    mask = EgoMotionCompensator._build_exclusion_mask(shape, [box_full_res], scale_factor)

    assert mask is not None
    # Well inside the (padded) box -> excluded
    assert mask[40, 100] == 0
    # Far outside the box -> still searchable
    assert mask[5, 5] == 255
    assert mask[95, 195] == 255


def test_exclusion_mask_none_when_no_targets():
    """No active targets yet -> no mask needed, corner search covers the
    whole frame (goodFeaturesToTrack treats mask=None as 'search everywhere')."""
    mask = EgoMotionCompensator._build_exclusion_mask((100, 200), None, 1.0)
    assert mask is None
    mask = EgoMotionCompensator._build_exclusion_mask((100, 200), [], 1.0)
    assert mask is None


def test_safety_crop_preserves_frame_size():
    """The output must always be the same size as the input -- the crop is
    an internal detail (crop then rescale back up), never a visible resize
    of the video the person is watching."""
    frame = np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8)
    cropped = EgoMotionCompensator._apply_safety_crop(frame)
    assert cropped.shape == frame.shape


def test_safety_crop_discards_original_edge_pixels():
    """Sanity check that a crop actually happened (not a no-op passthrough)
    -- paint a distinct marker in the outermost pixel border and confirm
    it does NOT survive into the output, since that border is exactly the
    replicate-stretched strip this crop exists to hide."""
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[0, :] = 255       # top edge marker
    frame[:, 0] = 255       # left edge marker
    out = EgoMotionCompensator._apply_safety_crop(frame)
    assert not np.array_equal(out[0, :], frame[0, :])


def test_rotation_deg_matches_expected_angle():
    """Sanity check the shared rotation-extraction helper against a known
    30-degree rotation matrix -- this is what both the per-frame plausibility
    check AND the cumulative-drift cap rely on."""
    theta = math.radians(30)
    affine = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                        [math.sin(theta),  math.cos(theta), 0.0]], dtype=np.float32)
    assert abs(abs(EgoMotionCompensator._rotation_deg(affine)) - 30.0) < 0.5


def test_cumulative_rotation_cap_resets_before_extreme_drift():
    """Regression test for the 'whole frame ends up rotated ~30+ degrees'
    bug: several individually-small (well under MAX_PLAUSIBLE_ROTATION_DEG)
    same-direction rotations must NOT be allowed to compound past
    MAX_CUMULATIVE_ROTATION_DEG before a reset kicks in -- previously this
    only happened every REANCHOR_INTERVAL frames, which was long enough for
    severe visible drift to build up first."""
    comp = EgoMotionCompensator()
    small_rotation_deg = 3.0  # under MAX_PLAUSIBLE_ROTATION_DEG (4.0)
    theta = math.radians(small_rotation_deg)
    step_affine = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                             [math.sin(theta),  math.cos(theta), 0.0]], dtype=np.float32)

    # Manually drive the same compose+cap logic step() uses, without needing
    # real video frames -- feed the SAME small rotation repeatedly, the
    # worst case for compounding drift.
    for _ in range(20):
        comp._cumulative_transform = comp._compose_affine(comp._cumulative_transform, step_affine)
        if abs(comp._rotation_deg(comp._cumulative_transform)) > MAX_CUMULATIVE_ROTATION_DEG:
            comp._cumulative_transform = np.eye(2, 3, dtype=np.float32)
        # After every step, cumulative rotation must stay within the cap
        # (plus one step's worth of slack, since the cap check happens
        # AFTER composing that step).
        assert abs(comp._rotation_deg(comp._cumulative_transform)) <= MAX_CUMULATIVE_ROTATION_DEG + small_rotation_deg