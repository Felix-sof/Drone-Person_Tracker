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
    MOTION_COMP_CROP_MARGIN_PCT,
    MOTION_COMP_ESTIMATION_WIDTH,
    MOTION_COMP_EXCLUSION_PAD_RATIO,
    RANSAC_REPROJ_THRESHOLD,
    REANCHOR_INTERVAL,
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

    @staticmethod
    def _build_exclusion_mask(shape, exclude_boxes, scale_factor):
        """
        Builds a goodFeaturesToTrack mask (uint8; 255 = ok to search here,
        0 = skip) that blocks out the given boxes so camera-motion
        estimation never picks corners that sit ON a tracked person.

        exclude_boxes are in FULL-RESOLUTION (x1, y1, x2, y2) coordinates
        (they come from each target's tracker, one frame stale); shape/
        scale_factor describe the downscaled estimation image these corners
        will actually be searched in, so boxes are converted down to match.
        Returns None (meaning "search everywhere") if there's nothing to
        exclude, so the common case (no active targets yet) pays zero cost.
        """
        if not exclude_boxes:
            return None

        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        for (x1, y1, x2, y2) in exclude_boxes:
            pad_x = (x2 - x1) * MOTION_COMP_EXCLUSION_PAD_RATIO
            pad_y = (y2 - y1) * MOTION_COMP_EXCLUSION_PAD_RATIO
            sx1 = int((x1 - pad_x) / scale_factor)
            sy1 = int((y1 - pad_y) / scale_factor)
            sx2 = int((x2 + pad_x) / scale_factor)
            sy2 = int((y2 + pad_y) / scale_factor)
            sx1, sy1 = max(sx1, 0), max(sy1, 0)
            sx2, sy2 = min(sx2, w), min(sy2, h)
            if sx2 > sx1 and sy2 > sy1:
                mask[sy1:sy2, sx1:sx2] = 0
        return mask

    @staticmethod
    def _apply_safety_crop(frame: np.ndarray) -> np.ndarray:
        """
        Crops MOTION_COMP_CROP_MARGIN_PCT off every edge and rescales back
        to the original size. Applied unconditionally (every frame, whether
        or not compensation actually ran this frame) so the field of view
        never visibly zooms in/out -- only WHERE the crop hides the
        replicate-border smear changes frame to frame, not its size.
        """
        h, w = frame.shape[:2]
        mx = int(w * MOTION_COMP_CROP_MARGIN_PCT)
        my = int(h * MOTION_COMP_CROP_MARGIN_PCT)
        if mx <= 0 or my <= 0:
            return frame
        cropped = frame[my:h - my, mx:w - mx]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def step(self, frame_bgr: np.ndarray, exclude_boxes: list | None = None):
        """
        Args:
            frame_bgr: current frame, BGR uint8.
            exclude_boxes: optional list of (x1, y1, x2, y2) in frame_bgr's
                coordinates -- typically each active target's last-known
                box -- so estimation ignores corners sitting on a moving
                subject instead of the static background.

        Returns:
            warped_frame: frame aligned to the reference coordinate frame.
            affine_2x3: the incremental (prev -> current) affine transform,
                        or None if it could not be estimated this frame.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small, scale_factor = self._downscale_for_estimation(gray)
        mask = self._build_exclusion_mask(gray_small.shape, exclude_boxes, scale_factor)

        if self._prev_gray is None:
            # First frame: nothing to compare against yet.
            self._prev_gray = gray
            self._prev_gray_small = gray_small
            return self._apply_safety_crop(frame_bgr.copy()), None

        affine = self._estimate_affine(self._prev_gray_small, gray_small, scale_factor, mask=mask)
        self._prev_gray = gray
        self._prev_gray_small = gray_small

        if affine is None or not self._is_plausible(affine):
            # Could not estimate reliably this frame (e.g. too few features,
            # heavy blur, or a degenerate/implausible jump). Falling back to
            # "no motion" for this single frame is much safer than composing
            # a bad transform -- a bad transform doesn't just ruin one frame,
            # it corrupts every frame after it too (see re-anchoring note
            # below for why that compounding is the real danger).
            return self._apply_safety_crop(frame_bgr.copy()), None

        # Compose with cumulative transform so nearby frames stay aligned.
        self._cumulative_transform = self._compose_affine(
            self._cumulative_transform, affine
        )
        self._frames_since_reanchor += 1

        # Safety net on top of the per-frame plausibility check: several
        # individually-"plausible" small errors can still compound in the
        # same direction faster than REANCHOR_INTERVAL resets them (this is
        # exactly how a frame ends up looking wildly rotated/smeared even
        # though no single step looked unreasonable). If the CUMULATIVE
        # rotation has drifted past a sane bound, reset right now instead of
        # riding out the rest of the interval.
        if abs(self._rotation_deg(self._cumulative_transform)) > MAX_CUMULATIVE_ROTATION_DEG:
            self._cumulative_transform = np.eye(2, 3, dtype=np.float32)
            self._frames_since_reanchor = 0

        h, w = frame_bgr.shape[:2]
        warped = cv2.warpAffine(
            frame_bgr, self._cumulative_transform, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Re-anchor periodically: treat the current (warped) frame as the new
        # zero-point. Without this, tiny per-frame estimation noise compounds
        # multiplicatively over hundreds of frames into severe warping/
        # streaking artifacts -- exactly the smeared look you get if you
        # never reset. This is standard practice in video stabilization
        # (sliding reference window instead of a single global anchor).
        if self._frames_since_reanchor >= REANCHOR_INTERVAL:
            self._cumulative_transform = np.eye(2, 3, dtype=np.float32)
            self._frames_since_reanchor = 0

        return self._apply_safety_crop(warped), affine

    @staticmethod
    def _rotation_deg(affine: np.ndarray) -> float:
        """Rotation implied by an affine's linear part, in degrees."""
        a, b = affine[0, 0], affine[0, 1]
        return math.degrees(math.atan2(b, a))

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
    def _estimate_affine(prev_gray: np.ndarray, curr_gray: np.ndarray,
                          scale_factor: float = 1.0, mask: np.ndarray | None = None):
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

        mask (optional): passed straight to goodFeaturesToTrack to keep
        corner search OUT of known target regions (see
        _build_exclusion_mask) -- this is what stops a moving subject from
        corrupting the camera-motion estimate.
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
        """Compose two 2x3 affine transforms: result = outer ∘ inner."""
        outer_3x3 = np.vstack([outer, [0, 0, 1]])
        inner_3x3 = np.vstack([inner, [0, 0, 1]])
        composed = outer_3x3 @ inner_3x3
        return composed[:2, :]