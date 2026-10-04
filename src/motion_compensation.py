"""
Ego-motion estimation and compensation.

Idea: most of the pixels in a drone frame belong to the static background.
If we track a large set of sparse points frame-to-frame and fit a single
global transform (affine: translation + rotation + scale) to the MAJORITY
of them via RANSAC, that transform represents the camera's own motion.
Warping the next frame by the inverse of that transform "cancels" camera
shake/drift, so whatever motion remains after warping is real object motion.

This directly reuses the block-matching / motion-estimation intuition from
video coding (cost measures, sparse correspondences) but applies it as a
global alignment step instead of per-block prediction.
"""

import math

import cv2
import numpy as np

from config import (
    MAX_CORNERS,
    MAX_CUMULATIVE_ROTATION_DEG,
    MAX_PLAUSIBLE_ROTATION_DEG,
    MAX_PLAUSIBLE_SCALE_DELTA,
    MIN_CORNERS_REQUIRED,
    MOTION_COMP_ESTIMATION_WIDTH,
    RANSAC_REPROJ_THRESHOLD,
    REANCHOR_INTERVAL,
    SAFETY_CROP_PX,
    TARGET_EXCLUSION_PADDING_PX,
)


class EgoMotionCompensator:
    """
    Stateful compensator: call `step(frame)` once per frame, in order.
    Returns the frame warped into the coordinate system of the FIRST frame
    it ever saw, plus the estimated affine transform for this step.
    """

    def __init__(self):
        self._prev_gray = None
        self._prev_gray_small = None  # downscaled copy, used only for estimation
        # Cumulative transform mapping "current frame" -> "reference frame"
        self._cumulative_transform = np.eye(2, 3, dtype=np.float32)
        self._frames_since_reanchor = 0

    def reset(self):
        self._prev_gray = None
        self._prev_gray_small = None
        self._cumulative_transform = np.eye(2, 3, dtype=np.float32)
        self._frames_since_reanchor = 0

    @staticmethod
    def _downscale_for_estimation(gray: np.ndarray):
        """
        Returns (small_gray, scale_factor) where scale_factor converts a
        distance measured in small_gray's pixels back to the ORIGINAL
        frame's pixels (scale_factor = original_width / small_width).
        No-op (scale_factor=1.0) if the frame is already narrower than the
        configured estimation width.
        """
        h, w = gray.shape[:2]
        if w <= MOTION_COMP_ESTIMATION_WIDTH:
            return gray, 1.0
        scale_factor = w / MOTION_COMP_ESTIMATION_WIDTH
        new_w = MOTION_COMP_ESTIMATION_WIDTH
        new_h = max(1, int(round(h / scale_factor)))
        small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return small, scale_factor

    def estimate(self, frame_bgr: np.ndarray, target_boxes: list | None = None):
        """
        Camera-motion ESTIMATION only, no warping ("tracks" mode, see
        MOTION_COMP_MODE in config.py).

        Returns the affine (2x3, full-resolution pixels) that maps a point
        in the PREVIOUS frame to where that same static-scene point appears
        in THIS frame -- or None on the first frame / when the estimate is
        unreliable. The pipeline applies it to every track's state instead
        of warping the image, so a drone that is deliberately flying/panning
        doesn't produce stretched border artifacts or a drifting coordinate
        frame (the failure mode of step() on non-hovering footage).

        target_boxes: previous frame's tracked boxes (excluded from corner
        search, same as step()).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small, scale_factor = self._downscale_for_estimation(gray)
        prev_small = self._prev_gray_small
        self._prev_gray_small = gray_small
        self._prev_gray = None   # full-res copy is only needed by step()

        if prev_small is None or prev_small.shape != gray_small.shape:
            return None
        mask = self._build_exclusion_mask(gray_small.shape[:2], target_boxes, scale_factor)
        affine = self._estimate_affine(prev_small, gray_small, scale_factor, mask)
        if affine is None or not self._is_plausible(affine):
            return None
        return affine

    def step(self, frame_bgr: np.ndarray, target_boxes: list | None = None):
        """
        Args:
            frame_bgr: current frame, BGR uint8.
            target_boxes: full-resolution (x1, y1, x2, y2) boxes of currently
                tracked targets (from the PREVIOUS frame's tracking result --
                this frame's detections don't exist yet at this point in the
                pipeline). Corner features are not picked from inside these
                boxes, so a moving person can't corrupt the background-motion
                estimate. None/empty -> search the whole frame, as before.

        Returns:
            warped_frame: frame aligned to the reference coordinate frame.
            affine_2x3: the incremental (prev -> current) affine transform,
                        or None if it could not be estimated this frame.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small, scale_factor = self._downscale_for_estimation(gray)

        if self._prev_gray is None:
            # First frame: nothing to compare against yet.
            self._prev_gray = gray
            self._prev_gray_small = gray_small
            return frame_bgr.copy(), None

        mask = self._build_exclusion_mask(gray_small.shape[:2], target_boxes, scale_factor)
        affine = self._estimate_affine(self._prev_gray_small, gray_small, scale_factor, mask)
        self._prev_gray = gray
        self._prev_gray_small = gray_small

        if affine is None or not self._is_plausible(affine):
            # Could not estimate reliably this frame (e.g. too few features,
            # heavy blur, or a degenerate/implausible jump). Falling back to
            # "no motion" for this single frame is much safer than composing
            # a bad transform -- a bad transform doesn't just ruin one frame,
            # it corrupts every frame after it too (see re-anchoring note
            # below for why that compounding is the real danger).
            warped = frame_bgr.copy()
            return warped, None

        # Compose with cumulative transform so nearby frames stay aligned.
        self._cumulative_transform = self._compose_affine(
            self._cumulative_transform, affine
        )
        self._frames_since_reanchor += 1

        # Cumulative-drift cap: several individually-small, same-direction
        # rotations can each pass _is_plausible() on their own yet still
        # compound into a frame rotated by tens of degrees before the next
        # periodic re-anchor. Check the CUMULATIVE rotation every frame (not
        # just at REANCHOR_INTERVAL) and reset the moment it exceeds this cap.
        if abs(self._rotation_deg(self._cumulative_transform)) > MAX_CUMULATIVE_ROTATION_DEG:
            self._cumulative_transform = np.eye(2, 3, dtype=np.float32)
            self._frames_since_reanchor = 0

        h, w = frame_bgr.shape[:2]
        warped = cv2.warpAffine(
            frame_bgr, self._cumulative_transform, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        warped = self._apply_safety_crop(warped)

        # Re-anchor periodically: treat the current (warped) frame as the new
        # zero-point. Without this, tiny per-frame estimation noise compounds
        # multiplicatively over hundreds of frames into severe warping/
        # streaking artifacts -- exactly the smeared look you get if you
        # never reset. This is standard practice in video stabilization
        # (sliding reference window instead of a single global anchor).
        if self._frames_since_reanchor >= REANCHOR_INTERVAL:
            self._cumulative_transform = np.eye(2, 3, dtype=np.float32)
            self._frames_since_reanchor = 0

        return warped, affine

    @staticmethod
    def _is_plausible(affine: np.ndarray) -> bool:
        """
        Sanity-check a single-frame affine estimate. A real camera cannot
        rotate or rescale drastically within one frame interval; if the
        estimate implies that, the underlying correspondences were noise
        (e.g. a flat, texture-less wall gave goodFeaturesToTrack nothing
        reliable to lock onto) and should be rejected rather than applied.
        """
        a, b = affine[0, 0], affine[0, 1]
        scale = math.hypot(a, b)
        rotation_deg = EgoMotionCompensator._rotation_deg(affine)

        if abs(rotation_deg) > MAX_PLAUSIBLE_ROTATION_DEG:
            return False
        if abs(scale - 1.0) > MAX_PLAUSIBLE_SCALE_DELTA:
            return False
        return True

    @staticmethod
    def _rotation_deg(affine: np.ndarray) -> float:
        """Shared rotation-extraction helper: both the per-frame plausibility
        check and the cumulative-drift cap rely on this same math, so it's
        centralized here rather than duplicated."""
        a, b = affine[0, 0], affine[0, 1]
        return math.degrees(math.atan2(b, a))

    @staticmethod
    def _build_exclusion_mask(shape: tuple, target_boxes: list | None, scale_factor: float):
        """
        Builds a goodFeaturesToTrack mask (uint8, same `shape` as the
        estimation-resolution gray image -- possibly downscaled) that
        excludes currently-tracked targets' regions, so a moving person's
        silhouette can't be picked up as a "background" corner and corrupt
        the camera-motion estimate.

        `target_boxes` are (x1, y1, x2, y2) in FULL-RESOLUTION pixel coords;
        `scale_factor` (original_width / estimation_width) converts them
        down into the estimation resolution described by `shape`.

        Returns None when there's nothing to exclude -- goodFeaturesToTrack
        treats mask=None as "search the whole frame", so this is also the
        zero-cost path when there are no active targets yet.
        """
        if not target_boxes:
            return None
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        pad = TARGET_EXCLUSION_PADDING_PX
        for box in target_boxes:
            x1, y1, x2, y2 = box
            sx1 = max(0, int((x1 - pad) / scale_factor))
            sy1 = max(0, int((y1 - pad) / scale_factor))
            sx2 = min(w, int((x2 + pad) / scale_factor))
            sy2 = min(h, int((y2 + pad) / scale_factor))
            mask[sy1:sy2, sx1:sx2] = 0
        return mask

    @staticmethod
    def _apply_safety_crop(frame: np.ndarray) -> np.ndarray:
        """
        warpAffine's replicate border mode stretches edge pixels into thin,
        visible streaks near the frame boundary. Cropping a small margin off
        each side and rescaling back up to the original size discards that
        artifact without changing the output resolution the rest of the
        pipeline (detection, display) expects.
        """
        h, w = frame.shape[:2]
        c = SAFETY_CROP_PX
        if h <= 2 * c or w <= 2 * c:
            return frame
        cropped = frame[c:h - c, c:w - c]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _estimate_affine(prev_gray: np.ndarray, curr_gray: np.ndarray, scale_factor: float = 1.0,
                          mask: np.ndarray | None = None):
        """
        1) Pick sparse "good" corner points in the previous frame
           (these are likely to be trackable background texture).
        2) Track them into the current frame with Lucas-Kanade optical flow.
        3) Fit ONE affine transform to the majority of correspondences using
           RANSAC -- outliers here are typically points that sit ON a moving
           object (the person), not on the static background.

        prev_gray/curr_gray may be a DOWNSCALED copy of the real frame (see
        MOTION_COMP_ESTIMATION_WIDTH) -- scale_factor converts the resulting
        transform's translation back into full-resolution pixel units.
        Rotation/scale don't need adjusting: they're scale-invariant, only
        translation (measured in pixels) does.
        """
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=MAX_CORNERS, qualityLevel=0.01,
            minDistance=8, blockSize=7, mask=mask
        )
        if prev_pts is None or len(prev_pts) < MIN_CORNERS_REQUIRED:
            return None

        curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None
        )
        if curr_pts is None:
            return None

        status = status.reshape(-1).astype(bool)
        prev_valid = prev_pts[status]
        curr_valid = curr_pts[status]

        if len(prev_valid) < MIN_CORNERS_REQUIRED:
            return None

        # estimateAffinePartial2D restricts the model to rotation+scale+
        # translation (4 DOF) which matches rigid camera motion well and is
        # more stable than a full 6-DOF affine fit on noisy points.
        affine, inliers = cv2.estimateAffinePartial2D(
            prev_valid, curr_valid,
            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
        )
        if affine is None:
            return None

        if scale_factor != 1.0:
            affine = affine.copy()
            affine[:, 2] *= scale_factor  # translation column only -> full-res units

        return affine

    @staticmethod
    def _compose_affine(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        """Compose two 2x3 affine transforms: result = outer . inner."""
        outer_3x3 = np.vstack([outer, [0, 0, 1]])
        inner_3x3 = np.vstack([inner, [0, 0, 1]])
        composed = outer_3x3 @ inner_3x3
        return composed[:2, :]