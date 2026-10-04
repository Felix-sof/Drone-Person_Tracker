"""
Tests for the Kalman-backed single-target tracker and the vectorized IOU
helpers in src/tracker.py.
"""

import numpy as np
import pytest

from src.tracker import KalmanBoxFilter, SingleTargetTracker, iou, iou_matrix


def _moving_box(step: int, vx: int = 25, vy: int = 0, x0: int = 100, y0: int = 100,
                w: int = 40, h: int = 100):
    return (x0 + vx * step, y0 + vy * step, x0 + vx * step + w, y0 + vy * step + h)


def test_iou_matrix_matches_scalar_iou():
    a = [(0, 0, 10, 10), (5, 5, 20, 20), (100, 100, 110, 120)]
    b = [(0, 0, 10, 10), (8, 8, 18, 18)]
    m = iou_matrix(a, b)
    assert m.shape == (3, 2)
    for i, ba in enumerate(a):
        for j, bb in enumerate(b):
            assert m[i, j] == pytest.approx(iou(ba, bb), abs=1e-6)


def test_iou_matrix_empty_inputs():
    assert iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1)
    assert iou_matrix([(0, 0, 1, 1)], []).shape == (1, 0)


def test_kalman_learns_constant_velocity():
    """After a few measurements of a box moving 25px/frame, the prediction
    for the next frame should land close to where the box actually goes --
    much closer than the last observed box (which is what the old
    IOU-only tracker compared against)."""
    kf = KalmanBoxFilter(_moving_box(0))
    for step in range(1, 8):
        kf.predict()
        kf.update(_moving_box(step))

    predicted = kf.predict()
    actual = _moving_box(8)
    last_seen = _moving_box(7)

    assert iou(predicted, actual) > 0.7
    assert iou(predicted, actual) > iou(last_seen, actual)
    vx, vy = kf.velocity
    assert vx == pytest.approx(25, abs=3)
    assert vy == pytest.approx(0, abs=3)


def test_kalman_box_never_degenerates_while_coasting():
    """A shrinking box extrapolated for a long time must stay a valid
    rectangle (positive width/height) rather than inverting."""
    kf = KalmanBoxFilter((100, 100, 140, 200))
    for step in range(1, 6):
        kf.predict()
        size = 40 - 7 * step
        kf.update((100, 100, 100 + max(size, 2), 100 + max(100 - 18 * step, 2)))
    for _ in range(50):
        x1, y1, x2, y2 = kf.predict()
        assert x2 > x1 and y2 > y1


def test_tracker_follows_fast_mover_with_prediction():
    """A target moving more than its own width per frame breaks plain
    last-box IOU, but the predict/match protocol keeps it locked once the
    filter has a velocity."""
    tracker = SingleTargetTracker(iou_threshold=0.3, max_missed_frames=3)
    tracker.initialize(_moving_box(0, vx=30))
    tracker.predict()
    tracker.mark_matched(_moving_box(1, vx=30))   # e.g. via gated Re-ID continuation

    for step in range(2, 12):
        pred = tracker.predict()
        actual = _moving_box(step, vx=30)
        # Plain last-box IOU would fail here: the box moves 30px with a 40px width.
        assert iou(tracker.current_box, actual) < 0.3
        assert iou(pred, actual) >= 0.3
        tracker.mark_matched(actual)

    assert tracker.is_locked
    assert tracker.hits == 12


def test_tracker_coasts_then_loses_after_max_missed():
    tracker = SingleTargetTracker(iou_threshold=0.3, max_missed_frames=3)
    tracker.initialize((0, 0, 10, 30))
    for _ in range(3):
        tracker.predict()
        tracker.mark_missed()
        assert tracker.is_locked          # still coasting
        assert tracker.predicted_box is not None
    tracker.predict()
    tracker.mark_missed()
    assert tracker.lost()
    assert tracker.current_box is None
    assert tracker.predict() is None


def test_standalone_update_wrapper_still_works():
    """SingleTargetTracker.update() (greedy single-target use) keeps its
    old contract: returns the matched box or None."""

    class _Det:
        def __init__(self, box):
            self.box = box

    tracker = SingleTargetTracker(iou_threshold=0.3)
    tracker.initialize((0, 0, 50, 100))
    assert tracker.update([_Det((2, 0, 52, 100)), _Det((300, 300, 350, 400))]) == (2, 0, 52, 100)
    assert tracker.update([_Det((300, 300, 350, 400))]) is None
    assert tracker.missed_frames == 1


def test_kalman_state_is_finite_after_many_steps():
    kf = KalmanBoxFilter((10, 10, 30, 60))
    for step in range(200):
        kf.predict()
        if step % 3 == 0:
            kf.update((10 + step, 10, 30 + step, 60))
    assert np.all(np.isfinite(kf.x))
    assert np.all(np.isfinite(kf.P))


def test_kalman_apply_affine_translation_moves_position_not_velocity():
    from src.tracker import KalmanBoxFilter
    kf = KalmanBoxFilter((100, 100, 140, 200))
    for step in range(1, 6):
        kf.predict()
        kf.update((100 + 10 * step, 100, 140 + 10 * step, 200))
    vx_before, vy_before = kf.velocity
    cx_before = kf.x[0]
    kf.apply_affine(np.array([[1.0, 0.0, 50.0], [0.0, 1.0, -30.0]]))
    assert kf.x[0] == pytest.approx(cx_before + 50)
    assert kf.velocity == pytest.approx((vx_before, vy_before))


def test_kalman_apply_affine_rotation_rotates_velocity():
    from src.tracker import KalmanBoxFilter
    kf = KalmanBoxFilter((0, 0, 20, 40))
    kf.x[4:6] = [10.0, 0.0]
    rot90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    kf.apply_affine(rot90)
    assert kf.velocity == pytest.approx((0.0, 10.0), abs=1e-9)


def test_tracker_camera_motion_keeps_lock_on_panning_drone():
    """Static person, camera pans 60px/frame: in image coordinates the
    person jumps 60px each frame (more than their 40px width). With the
    camera motion applied to the track, prediction lands right on them."""
    from src.tracker import SingleTargetTracker, iou
    tracker = SingleTargetTracker(iou_threshold=0.3, max_missed_frames=2)
    box = (300, 300, 340, 400)
    tracker.initialize(box)
    pan = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, -60.0]])   # scene moves up 60px/frame
    for step in range(1, 8):
        box = (box[0], box[1] - 60, box[2], box[3] - 60)
        tracker.apply_camera_motion(pan)
        pred = tracker.predict()
        assert iou(pred, box) > 0.8
        tracker.mark_matched(box)
