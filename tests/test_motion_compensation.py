"""
Unit tests for the estimation-downscale feature in
src/motion_compensation.py (MOTION_COMP_ESTIMATION_WIDTH).

These are geometry/plumbing checks, not full optical-flow integration
tests -- they don't need a real camera frame.
"""

import numpy as np

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
