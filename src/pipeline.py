"""
Orchestrates the full flow, frame by frame, for MULTIPLE simultaneous
targets:

  raw frame
    -> ego-motion compensation   (cancel camera shake/drift, shared)
    -> person detection          (YOLO, ONE shared pass for all targets)
    -> Kalman prediction         (where should each locked target be now?)
    -> GLOBAL IOU assignment     (all locked targets vs all detections at
                                  once, Hungarian algorithm)
    -> gated Re-ID continuation  (IOU failed, but a similar-looking person
                                  is near the predicted position -> same
                                  track; lets the motion model learn fast
                                  movers' velocity)
    -> GLOBAL Re-ID assignment   (lost targets + periodic confirmations vs
                                  still-unclaimed detections, with switch-
                                  margin hysteresis)
    -> per matched target: periodic pose analysis, emotion analysis,
       activity/posture, distance, automatic gallery growth
    -> auto-track-all: unclaimed people become TENTATIVE targets, promoted
       to a real ID only after AUTO_TARGET_MIN_HITS consecutive matches
    -> annotated output frame (each target in its own color, dashboard-
       style HUD panel, corner-bracket box, trajectory trail)

Why GLOBAL assignment: the previous version let each target, in list order,
grab its best-overlapping detection. When two people walk close together,
target #1 can greedily take the box that really belongs to #2 (slightly
higher IOU with #1's stale box), leaving #2 with the wrong one or none --
the classic ID-swap. Solving one assignment over the whole IOU (or
similarity) matrix picks the combination that's best OVERALL instead.

The "claim" set (detection boxes already assigned this frame) still
guarantees no two targets lock onto the same physical person.
"""

import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    AUTO_GALLERY_NOVELTY_MAX_SIMILARITY,
    AUTO_GALLERY_UPDATE_INTERVAL,
    AUTO_TARGET_MIN_HITS,
    DB_LOG_INTERVAL,
    DISTANCE_ESTIMATION_INTERVAL,
    DUPLICATE_TARGET_SIMILARITY_THRESHOLD,
    EMOTION_ANALYSIS_INTERVAL,
    ENABLE_AUTO_GALLERY_UPDATE,
    ENABLE_AUTO_TRACK_ALL,
    ENABLE_DISTANCE_ESTIMATION,
    ENABLE_EMOTION_ANALYSIS,
    ENABLE_MOTION_COMPENSATION,
    ENABLE_TILED_DETECTION,
    ENABLE_TRAILS,
    MOTION_COMP_MODE,
    HUD_MAX_PANELS,
    LOST_TARGET_EXPIRY_FRAMES,
    MAX_CONCURRENT_TARGETS,
    POSE_ANALYSIS_INTERVAL,
    REID_CONTINUATION_GATE_HEIGHTS,
    REID_GALLERY_MAX_SIZE,
    REID_LOST_RESCAN_INTERVAL,
    REID_MATCH_THRESHOLD,
    REID_RESCAN_INTERVAL,
    REID_SWITCH_MARGIN,
    TRACKER_IOU_THRESHOLD,
)
from src.db import log_target_event
from src.distance import estimate_distance_m
from src.target import Target
from src.tracker import iou, iou_matrix

# A target must have been followed by IOU continuity for at least this many
# consecutive frames before its current appearance is trusted enough to be
# added to its gallery automatically -- a fresh Re-ID (re)lock alone is not
# enough evidence that we're on the right person.
_GALLERY_UPDATE_MIN_HITS = 5
_GATE_COST = 1e6


class AddTargetResult:
    def __init__(self, ok: bool, target_id: int | None, message: str):
        self.ok = ok
        self.target_id = target_id
        self.message = message


def _clip_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return (int(max(0, min(w, x1))), int(max(0, min(h, y1))),
            int(max(0, min(w, x2))), int(max(0, min(h, y2))))


def _crop(frame, box):
    """Crop with clamping -- a negative coordinate would otherwise silently
    wrap around in numpy slicing and return a wrong region."""
    x1, y1, x2, y2 = _clip_box(box, frame.shape)
    return frame[y1:y2, x1:x2]


def _in_motion_gate(predicted_box, det_box) -> bool:
    """Is `det_box` a physically plausible next position for a target
    predicted at `predicted_box`? Center within REID_CONTINUATION_GATE_HEIGHTS
    box-heights, and box height within a factor of 2 (a person can't
    double or halve in apparent size between two processed frames)."""
    px1, py1, px2, py2 = predicted_box
    dx1, dy1, dx2, dy2 = det_box
    ph = max(py2 - py1, 1)
    dh = max(dy2 - dy1, 1)
    if not (0.5 <= dh / ph <= 2.0):
        return False
    dist = np.hypot((dx1 + dx2 - px1 - px2) / 2.0, (dy1 + dy2 - py1 - py2) / 2.0)
    return dist <= REID_CONTINUATION_GATE_HEIGHTS * ph


def _blend_rect(frame, x1, y1, x2, y2, color, alpha: float) -> None:
    """Semi-transparent filled rectangle, blending ONLY that region in place.
    (cv2.addWeighted over a full-frame copy per panel cost ~5ms each on a
    1080x1920 frame -- for a few hundred pixels of actual panel.)"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    fill[:] = color
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _gated_assignment(score_matrix: np.ndarray, min_score: float) -> list[tuple[int, int]]:
    """Maximum-total-score one-to-one assignment, ignoring any pair whose
    score is below `min_score` (gated out before solving, so the solver
    never trades a good pair for a below-threshold one)."""
    if score_matrix.size == 0:
        return []
    valid = score_matrix >= min_score
    if not valid.any():
        return []
    cost = np.where(valid, -score_matrix, _GATE_COST)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if valid[r, c]]


class _EmbeddingCache:
    """Per-frame Re-ID embedding cache keyed by box. Several stages (Re-ID
    rescan for every lost target, auto-track, gallery growth) need
    embeddings of the same detections; this computes each one at most once
    per frame, with all misses batched into a single forward pass."""

    def __init__(self, embedder, frame):
        self._embedder = embedder
        self._frame = frame
        self._cache: dict = {}
        self._invalid: set = set()

    def embed_boxes(self, boxes: list):
        """Returns (valid_boxes, embeddings[N, D]) -- boxes whose crop is
        empty are dropped."""
        missing = [b for b in dict.fromkeys(boxes)
                   if b not in self._cache and b not in self._invalid]
        crops = []
        to_embed = []
        for b in missing:
            crop = _crop(self._frame, b)
            if crop.size == 0:
                self._invalid.add(b)
            else:
                crops.append(crop)
                to_embed.append(b)
        if crops:
            for b, emb in zip(to_embed, self._embedder.embed_batch(crops)):
                self._cache[b] = emb

        valid = [b for b in boxes if b in self._cache]
        if not valid:
            return [], np.empty((0, 0), dtype=np.float32)
        return valid, np.stack([self._cache[b] for b in valid])


class DronePersonTrackingPipeline:
    def __init__(self, detector=None, embedder=None, motion_compensator=None,
                 pose_analyzer=None, emotion_analyzer=None):
        """All components are injectable (tests pass lightweight fakes);
        omitted ones are built with their real, model-backed defaults.
        Imports are deferred so a fully-injected pipeline loads no models."""
        if detector is None:
            from src.detection import PersonDetector
            detector = PersonDetector()
        if embedder is None:
            from src.reid import ReIDEmbedder
            embedder = ReIDEmbedder()
        if motion_compensator is None:
            from src.motion_compensation import EgoMotionCompensator
            motion_compensator = EgoMotionCompensator()
        if pose_analyzer is None:
            from src.pose_analysis import PoseAnalyzer
            pose_analyzer = PoseAnalyzer()
        if emotion_analyzer is None:
            from src.emotion import EmotionAnalyzer
            emotion_analyzer = EmotionAnalyzer()

        self.detector = detector
        self.embedder = embedder
        self.motion_compensator = motion_compensator
        self.pose_analyzer = pose_analyzer
        self.emotion_analyzer = emotion_analyzer

        self.targets: list[Target] = []
        self._next_target_id = 1
        self._next_tentative_id = -1
        self._frame_count = 0
        self.last_detection_count = 0
        self.last_camera_affine = None

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------
    @property
    def confirmed_targets(self) -> list[Target]:
        return [t for t in self.targets if t.confirmed]

    def add_target(self, frame_bgr: np.ndarray) -> AddTargetResult:
        """Captures the largest person in frame_bgr as a brand-new target,
        UNLESS they already match an existing target's gallery closely
        enough to be considered the same physical person (in which case
        creation is refused -- use add_angle() for that existing target
        instead)."""
        if len(self.confirmed_targets) >= MAX_CONCURRENT_TARGETS:
            return AddTargetResult(False, None,
                                    f"Already tracking the max of {MAX_CONCURRENT_TARGETS} targets.")

        detections = self.detector.detect(frame_bgr)
        if not detections:
            return AddTargetResult(False, None, "No person detected in this frame.")

        # Ignore detections that ARE one of the targets we're already
        # tracking. Without this, "largest person in the whole frame" tends
        # to just re-pick an existing (usually closer/bigger) target instead
        # of the new person you're pointing at -- which then looks like a
        # false "already tracked" refusal on every subsequent 'n' press,
        # even though the person you actually want to add was never
        # embedded or compared at all.
        candidates = self._exclude_tracked_detections(detections)
        if not candidates:
            return AddTargetResult(
                False, None,
                "Only already-tracked target(s) are visible in this frame -- "
                "get the new person clearly in view (ideally alone, or bigger "
                "than the existing targets) before pressing 'n' again."
            )

        largest = max(candidates, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        crop = _crop(frame_bgr, largest.box)
        if crop.size == 0:
            return AddTargetResult(False, None, "No person detected in this frame.")

        embedding = self.embedder.embed(crop)

        duplicate_id = self._find_duplicate_target(embedding)
        if duplicate_id is not None:
            return AddTargetResult(
                False, duplicate_id,
                f"This looks like target #{duplicate_id}, already being tracked -- "
                f"use add_angle (key 'a') if you meant to add another angle for them."
            )

        # A tentative auto-track identity for this same person would become
        # a second ID once it confirms -- drop it in favor of the manual one.
        self.targets = [
            t for t in self.targets
            if t.confirmed or t.tracker.current_box is None
            or iou(t.tracker.current_box, largest.box) < 0.3
        ]

        new_id = self._next_target_id
        self._next_target_id += 1
        self.targets.append(Target(new_id, embedding, manual=True))
        return AddTargetResult(True, new_id, f"New target #{new_id} added.")

    def add_angle(self, target_id: int, frame_bgr: np.ndarray) -> AddTargetResult:
        """Adds an additional reference angle to an EXISTING target
        (e.g. their back view after they turn around) without creating a
        new identity."""
        target = self._find_target(target_id)
        if target is None:
            return AddTargetResult(False, None, f"No target #{target_id}.")

        detections = self.detector.detect(frame_bgr)
        if not detections:
            return AddTargetResult(False, None, "No person detected in this frame.")

        # Match against THIS target's own last-known box, not "whoever is
        # biggest in the frame" -- with multiple targets on screen, the
        # biggest person is often a DIFFERENT target, and blindly taking
        # them here would silently add a wrong-person embedding to this
        # target's gallery (a much worse failure than add_target's, since
        # nothing refuses it -- it just quietly corrupts future Re-ID
        # matches for this target).
        if target.tracker.current_box is not None:
            best_det = max(detections, key=lambda d: iou(d.box, target.tracker.current_box))
            if iou(best_det.box, target.tracker.current_box) < 0.1:
                return AddTargetResult(
                    False, None,
                    f"Could not find target #{target_id} in this frame -- "
                    f"make sure they're clearly in view before pressing 'a'."
                )
            chosen = best_det
        else:
            chosen = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        crop = _crop(frame_bgr, chosen.box)
        if crop.size == 0:
            return AddTargetResult(False, None, "No person detected in this frame.")

        self._append_to_gallery(target, self.embedder.embed(crop))
        return AddTargetResult(True, target_id, f"Angle added to target #{target_id}.")

    def remove_target(self, target_id: int) -> bool:
        before = len(self.targets)
        self.targets = [t for t in self.targets if t.id != target_id]
        return len(self.targets) < before

    def get_target(self, target_id: int) -> Target | None:
        return self._find_target(target_id)

    def _find_target(self, target_id: int) -> Target | None:
        for t in self.targets:
            if t.confirmed and t.id == target_id:
                return t
        return None

    @staticmethod
    def _append_to_gallery(target: Target, embedding: np.ndarray) -> None:
        target.gallery.append(embedding)
        if len(target.gallery) > REID_GALLERY_MAX_SIZE:
            # Index 0 is the original (manually captured / first-seen)
            # reference shot -- never evict it.
            target.gallery.pop(1)

    def _exclude_tracked_detections(self, detections: list, iou_threshold: float = 0.3) -> list:
        """Drops detections that overlap an existing target's current
        tracked box -- i.e. detections that ARE one of our targets, not a
        new person standing near/behind them."""
        tracked_boxes = [t.tracker.current_box for t in self.confirmed_targets
                          if t.tracker.current_box is not None]
        if not tracked_boxes:
            return detections
        return [
            det for det in detections
            if all(iou(det.box, tb) < iou_threshold for tb in tracked_boxes)
        ]

    def _find_duplicate_target(self, embedding: np.ndarray,
                               include_tentative: bool = False) -> int | None:
        for target in self.targets:
            if not target.confirmed and not include_tentative:
                continue
            gallery_array = np.stack(target.gallery)
            max_sim = float(np.max(gallery_array @ embedding))
            if max_sim >= DUPLICATE_TARGET_SIMILARITY_THRESHOLD:
                return target.id
        return None

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, timestamp: float | None = None,
                      selected_id: int | None = None):
        """
        Args:
            frame_bgr: the raw frame.
            timestamp: capture time in seconds (video PTS for files);
                None -> wall clock. Drives activity/speed classification.
            selected_id: the operator's currently selected target, which is
                highlighted and gets the first HUD panel.

        Returns:
            (annotated_frame, statuses) -- one status dict per CONFIRMED
            target (tentative auto-track candidates are not reported).
        """
        self._frame_count += 1
        ts = time.monotonic() if timestamp is None else float(timestamp)

        tracked_boxes = [t.tracker.current_box for t in self.targets
                         if t.tracker.current_box is not None]
        warped_frame = frame_bgr
        self.last_camera_affine = None
        if ENABLE_MOTION_COMPENSATION and MOTION_COMP_MODE == "warp":
            warped_frame, _affine = self.motion_compensator.step(frame_bgr, target_boxes=tracked_boxes)
        elif ENABLE_MOTION_COMPENSATION:
            # "tracks" mode: leave the image alone, move the TRACKS with the
            # camera instead (works for a flying/panning drone, not just a
            # hovering one -- see MOTION_COMP_MODE in config.py).
            affine = self.motion_compensator.estimate(frame_bgr, target_boxes=tracked_boxes)
            if affine is not None:
                self.last_camera_affine = affine
                for target in self.targets:
                    target.apply_camera_motion(affine)
        detections = (self.detector.detect_tiled(warped_frame) if ENABLE_TILED_DETECTION
                      else self.detector.detect(warped_frame))
        self.last_detection_count = len(detections)

        cache = _EmbeddingCache(self.embedder, warped_frame)
        claimed: set = set()
        matched: dict = {}   # Target -> box
        scores: dict = {}    # Target -> best Re-ID score this frame

        for target in self.targets:
            target.tracker.predict()

        self._associate_by_iou(detections, matched, claimed)
        self._associate_by_gated_reid(detections, matched, claimed, scores, cache)
        for target in self.targets:
            if target.tracker.is_locked and target not in matched:
                target.tracker.mark_missed()
        self._associate_by_reid(detections, matched, claimed, scores, cache)

        self._run_due_pose_analysis(warped_frame, matched)

        statuses = []
        survivors = []
        for target in self.targets:
            box = matched.get(target)
            if box is None:
                target.record_missed()
                if not target.confirmed:
                    continue   # tentative + missed once -> discard silently
                if (not target.manual and target.tracker.lost()
                        and target.frames_since_seen > LOST_TARGET_EXPIRY_FRAMES):
                    continue   # long-gone auto target -> free its slot
            else:
                target.record_seen(box)
                if not target.confirmed and target.tracker.hits >= AUTO_TARGET_MIN_HITS:
                    target.confirm(self._next_target_id)
                    self._next_target_id += 1
            survivors.append(target)
            if target.confirmed:
                statuses.append(self._analyze_target(
                    target, warped_frame, box, scores.get(target), ts, cache))
        self.targets = survivors

        if ENABLE_AUTO_TRACK_ALL:
            statuses.extend(self._auto_track_new_people(warped_frame, detections, claimed, ts, cache))

        annotated = self._annotate(warped_frame, statuses, selected_id)
        return annotated, statuses

    def _run_due_pose_analysis(self, frame, matched: dict) -> None:
        """Pose analysis for every matched, confirmed target whose staggered
        turn it is this frame -- in ONE batched model call."""
        due = [t for t in self.targets
               if t.confirmed and matched.get(t) is not None
               and (self._frame_count + t.id) % POSE_ANALYSIS_INTERVAL == 0]
        if not due:
            return
        if hasattr(self.pose_analyzer, "analyze_batch"):
            results = self.pose_analyzer.analyze_batch(frame, [matched[t] for t in due])
        else:
            results = [self.pose_analyzer.analyze(frame, matched[t]) for t in due]
        for target, result in zip(due, results):
            target.last_pose_result = result

    def _associate_by_iou(self, detections, matched: dict, claimed: set) -> None:
        """Stage 1: every locked target's PREDICTED box vs every detection,
        solved as one assignment problem."""
        locked = [t for t in self.targets if t.tracker.is_locked]
        if locked and detections:
            ious = iou_matrix([t.tracker.search_box for t in locked], [d.box for d in detections])
            for r, c in _gated_assignment(ious, TRACKER_IOU_THRESHOLD):
                target, box = locked[r], detections[c].box
                target.tracker.mark_matched(box)
                matched[target] = box
                claimed.add(box)

    def _associate_by_gated_reid(self, detections, matched: dict, claimed: set,
                                 scores: dict, cache: _EmbeddingCache) -> None:
        """
        Stage 1b: locked targets that IOU couldn't place, vs unclaimed
        detections NEAR their predicted position, matched by appearance.

        Why: the Kalman filter needs two measurements to learn a velocity.
        A target moving more than ~its own width per processed frame fails
        IOU on the very next frame (prediction == last box until velocity
        is known), so without this stage it would coast, get lost, get
        re-initialized by Re-ID (velocity reset to zero) -- and repeat,
        never actually learning how fast it's going. A tentative auto-track
        candidate would simply be discarded every time. Matching by
        appearance inside a motion gate (DeepSORT-style) lets the filter
        absorb that jump as a measurement and lock on.
        """
        pending = [t for t in self.targets if t.tracker.is_locked and t not in matched]
        if not pending:
            return
        unclaimed = [d.box for d in detections if d.box not in claimed]
        if not unclaimed:
            return

        gate = np.array([[_in_motion_gate(t.tracker.search_box, b) for b in unclaimed]
                         for t in pending])
        if not gate.any():
            return
        in_any_gate = [b for b, g in zip(unclaimed, gate.any(axis=0)) if g]
        valid_boxes, embs = cache.embed_boxes(in_any_gate)
        if not valid_boxes:
            return
        col_of = {b: i for i, b in enumerate(unclaimed)}
        cols = [col_of[b] for b in valid_boxes]

        sims = np.stack([(embs @ np.stack(t.gallery).T).max(axis=1) for t in pending])  # (T, M)
        sims = np.where(gate[:, cols], sims, -1.0)
        for r, c in _gated_assignment(sims, REID_MATCH_THRESHOLD):
            target, box = pending[r], valid_boxes[c]
            target.tracker.mark_matched(box)
            scores[target] = float(sims[r, c])
            matched[target] = box
            claimed.add(box)

    def _associate_by_reid(self, detections, matched: dict, claimed: set,
                           scores: dict, cache: _EmbeddingCache) -> None:
        """Stage 2: Re-ID for confirmed targets that IOU couldn't place --
        recovery when lost, plus a periodic (staggered) confirmation while
        coasting. A coasting target only switches to a new person if that
        candidate beats its last locked score by REID_SWITCH_MARGIN, so a
        marginal/noisy similarity can't steal a lock that's doing fine."""
        # Lost targets are searched for every REID_LOST_RESCAN_INTERVAL
        # frames, not every frame: each search re-embeds EVERY unclaimed
        # person in view, and in a crowd with the target cap reached that
        # meant re-embedding dozens of untracked people on every frame for
        # a recovery that can just as well happen a few frames later.
        lost_search_frame = self._frame_count % REID_LOST_RESCAN_INTERVAL == 0
        rescan = [
            t for t in self.targets
            if t.confirmed and t not in matched
            and ((t.tracker.lost() and (lost_search_frame or t.total_frames_seen == 0))
                 or (self._frame_count + t.id) % REID_RESCAN_INTERVAL == 0)
        ]
        if not rescan:
            return
        available = [d.box for d in detections if d.box not in claimed]
        valid_boxes, embs = cache.embed_boxes(available)
        if not valid_boxes:
            return

        sims = np.stack([(embs @ np.stack(t.gallery).T).max(axis=1) for t in rescan])  # (T, M)
        for target, row in zip(rescan, sims):
            scores[target] = float(row.max())

        for r, c in _gated_assignment(sims, REID_MATCH_THRESHOLD):
            target, score, box = rescan[r], float(sims[r, c]), valid_boxes[c]
            should_switch = (
                target.tracker.lost()
                or target.locked_score is None
                or score >= target.locked_score + REID_SWITCH_MARGIN
            )
            if not should_switch:
                continue
            target.tracker.initialize(box)
            target.locked_score = score
            matched[target] = box
            claimed.add(box)

    def _auto_track_new_people(self, warped_frame, detections, claimed: set,
                               ts: float, cache: _EmbeddingCache) -> list[dict]:
        """
        When ENABLE_AUTO_TRACK_ALL is on, every detected person not already
        claimed by an existing target this frame becomes a TENTATIVE target
        (no visible ID yet). It's promoted to a real target only after
        AUTO_TARGET_MIN_HITS consecutive matches, and silently discarded if
        it misses even once before that -- so a one-frame false detection
        never consumes an ID.

        Respects MAX_CONCURRENT_TARGETS and the same duplicate-person check
        add_target uses (a person who briefly dropped out of tracking and
        got re-detected shouldn't be spawned as a second identity).
        """
        room = MAX_CONCURRENT_TARGETS - len(self.targets)
        unclaimed = [d.box for d in detections if d.box not in claimed]
        if room <= 0 or not unclaimed:
            return []

        # Bigger (closer, more reliably embedded) people first if we're
        # about to hit the cap.
        unclaimed.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        # Only embed as many candidates as could actually become targets
        # (plus slack for ones rejected as duplicates). In a crowd with the
        # cap nearly full, embedding ALL ~25 untracked people every frame
        # just to fill 1-2 free slots was most of the Re-ID cost.
        unclaimed = unclaimed[:room + 3]
        valid_boxes, embeddings = cache.embed_boxes(unclaimed)

        new_statuses = []
        for box, embedding in zip(valid_boxes, embeddings):
            if room <= 0:
                break
            if self._find_duplicate_target(embedding, include_tentative=True) is not None:
                continue

            target = Target(self._next_tentative_id, embedding, manual=False)
            self._next_tentative_id -= 1
            target.tracker.initialize(box)
            target.record_seen(box)
            claimed.add(box)
            room -= 1
            self.targets.append(target)

            if AUTO_TARGET_MIN_HITS <= 1:
                target.confirm(self._next_target_id)
                self._next_target_id += 1
                new_statuses.append(self._analyze_target(target, warped_frame, box, None, ts, cache))
        return new_statuses

    def _analyze_target(self, target: Target, warped_frame, matched_box, score,
                        ts: float, cache: _EmbeddingCache) -> dict:
        status = {"id": target.id, "color": target.color, "box": matched_box,
                  "predicted_box": None, "state": "tracking", "score": score,
                  "activity": "unknown", "pose": None, "emotion": None,
                  "gallery_size": len(target.gallery), "distance_m": None,
                  "manual": target.manual, "trail": [],
                  "frames_since_seen": target.frames_since_seen}

        if matched_box is None:
            if target.tracker.is_locked:
                # Coasting on the motion model through a brief miss (blur,
                # occlusion). Keep accumulated analysis state -- one missed
                # detection shouldn't wipe the activity history -- but don't
                # report anything as if it were freshly observed.
                status["state"] = "coasting"
                status["predicted_box"] = _clip_box(target.tracker.predicted_box,
                                                    warped_frame.shape)
                status["activity"] = "coasting"
            else:
                status["state"] = "lost"
                target.reset_analysis_state()
            return status

        if ENABLE_TRAILS:
            status["trail"] = list(target.trail)

        # Automatic gallery growth: only once IOU continuity has confirmed
        # the lock for several consecutive frames.
        if (ENABLE_AUTO_GALLERY_UPDATE
                and target.tracker.hits >= _GALLERY_UPDATE_MIN_HITS
                and (self._frame_count + target.id) % AUTO_GALLERY_UPDATE_INTERVAL == 0):
            self._maybe_auto_add_to_gallery(target, matched_box, cache)
            status["gallery_size"] = len(target.gallery)

        # Stagger expensive per-target work across frames using each
        # target's own ID as a phase offset. Without this, ALL targets'
        # heavy analysis (pose model, emotion model) lands on the exact
        # same frame every POSE_ANALYSIS_INTERVAL/EMOTION_ANALYSIS_INTERVAL
        # frames, creating a periodic latency spike that feels like
        # stuttering. Staggering spreads that cost evenly across frames
        # instead, for smoother (if slightly less fresh) playback.
        # Pose itself ran (batched, for every target due this frame) in
        # _run_due_pose_analysis() -- just report the latest result here.
        status["pose"] = target.last_pose_result

        if (ENABLE_EMOTION_ANALYSIS and self.emotion_analyzer.available
                and (self._frame_count + target.id) % EMOTION_ANALYSIS_INTERVAL == 0):
            head_box = (target.last_pose_result.body_part_boxes.get("head")
                        if target.last_pose_result else None)
            if head_box is not None:
                target.last_emotion = self.emotion_analyzer.analyze(_crop(warped_frame, head_box))
        status["emotion"] = target.last_emotion

        if (ENABLE_DISTANCE_ESTIMATION
                and (self._frame_count + target.id) % DISTANCE_ESTIMATION_INTERVAL == 0):
            _bx1, by1, _bx2, by2 = matched_box
            target.last_distance_m = estimate_distance_m(by2 - by1, warped_frame.shape[0])
        status["distance_m"] = target.last_distance_m

        speed_activity = target.activity_classifier.update(matched_box, timestamp=ts)
        is_sitting = target.last_pose_result.is_sitting if target.last_pose_result else None
        posture_label = target.posture_tracker.update(is_sitting)
        status["activity"] = posture_label if posture_label is not None else speed_activity

        if (self._frame_count + target.id) % DB_LOG_INTERVAL == 0:
            log_target_event(
                target.id, status["activity"], posture_label, target.last_emotion,
                target.last_distance_m, matched_box,
            )

        return status

    def _maybe_auto_add_to_gallery(self, target: Target, box, cache: _EmbeddingCache) -> None:
        valid, embs = cache.embed_boxes([box])
        if not valid:
            return
        new_embedding = embs[0]
        max_similarity = float(np.max(np.stack(target.gallery) @ new_embedding))
        if max_similarity >= AUTO_GALLERY_NOVELTY_MAX_SIMILARITY:
            return
        if REID_GALLERY_MAX_SIZE <= 1:
            return
        self._append_to_gallery(target, new_embedding)

    # ------------------------------------------------------------------
    # Annotation -- clean "dashboard" style: corner-bracket boxes instead
    # of solid rectangles, a small nameplate tag, and a dark HUD panel per
    # target with aligned field/value rows (no per-body-part box clutter).
    # ------------------------------------------------------------------
    def _annotate(self, frame, statuses: list[dict], selected_id: int | None = None):
        out = frame.copy()
        visible = [s for s in statuses if s.get("box") is not None]

        for status in statuses:
            if status.get("state") == "coasting" and status.get("predicted_box") is not None:
                self._draw_corner_brackets(out, status["predicted_box"], status["color"],
                                           thickness=1, length_ratio=0.1)
                self._draw_nameplate(out, status["predicted_box"], status, status["color"],
                                     suffix=" ?")

        for status in visible:
            color = status.get("color", (0, 255, 0))
            is_selected = status["id"] == selected_id
            self._draw_trail(out, status.get("trail") or [], color)
            self._draw_corner_brackets(out, status["box"], color, thickness=3 if is_selected else 2)
            self._draw_nameplate(out, status["box"], status, color,
                                 prefix="> " if is_selected else "")
            self._draw_hand_markers(out, status, color)

        self._draw_hud_panels(out, visible, selected_id)

        if not statuses:
            self._draw_notice(out, "SCANNING FOR PEOPLE..." if ENABLE_AUTO_TRACK_ALL
                              else "NO TARGETS -- PRESS 'N' TO ADD ONE")
        elif not visible:
            self._draw_notice(out, "SEARCHING...")

        return out

    def _draw_hud_panels(self, frame, visible: list[dict], selected_id, panel_width: int = 200):
        """Shows full panels for as many targets as fit (selected one
        first), and a single '+N MORE' tag for the rest -- with
        auto-track-all and a crowd, one panel per target would otherwise
        run straight off the right edge of the frame."""
        if not visible:
            return
        ordered = sorted(visible, key=lambda s: (s["id"] != selected_id, s["id"]))
        fit = max(1, (frame.shape[1] - 15) // (panel_width + 10))
        shown = ordered[:max(1, min(HUD_MAX_PANELS, fit))]
        for i, status in enumerate(shown):
            self._draw_target_hud(frame, status, origin_index=i, panel_width=panel_width,
                                  selected=status["id"] == selected_id)
        hidden = len(ordered) - len(shown)
        if hidden > 0:
            ox = 15 + len(shown) * (panel_width + 10)
            if ox + 90 > frame.shape[1]:
                ox, oy = 15, frame.shape[0] - 20
            else:
                oy = 62
            cv2.putText(frame, f"+{hidden} MORE", (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_trail(frame, trail: list, color):
        """Fading polyline of recent ground-contact points -- older segments
        are darker and thinner, so direction of travel reads at a glance."""
        n = len(trail)
        if n < 2:
            return
        for i in range(1, n):
            age = i / (n - 1)   # 0 = oldest, 1 = newest
            seg_color = tuple(int(c * (0.25 + 0.75 * age)) for c in color)
            cv2.line(frame, trail[i - 1], trail[i], seg_color, 1 if age < 0.5 else 2, cv2.LINE_AA)

    @staticmethod
    def _draw_corner_brackets(frame, box, color, thickness=2, length_ratio=0.16):
        """A viewfinder-style corner-bracket outline instead of a solid
        rectangle -- reads as a tracking overlay rather than a crude box."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        length = max(12, int(min(w, h) * length_ratio))

        for (cx, cy), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_nameplate(frame, box, status, color, prefix: str = "", suffix: str = ""):
        """Small filled tag above the box, e.g. 'T01  0.87'."""
        x1, y1, _x2, _y2 = box
        text = f"{prefix}T{status['id']:02d}"
        if status.get("score") is not None:
            text += f"  {status['score']:.2f}"
        text += suffix

        (tw, th), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_x1, tag_y2 = x1, max(th + 8, y1 - 4)
        tag_y1 = tag_y2 - th - 8

        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x1 + tw + 12, tag_y2), color, -1)
        cv2.putText(frame, text, (tag_x1 + 6, tag_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hand_markers(frame, status, color):
        """Instead of drawing all six body-part boxes, only mark something
        when there's actually something to flag: a small circle where an
        elongated object (stick/pole/flag) was detected near a hand."""
        pose = status.get("pose")
        if pose is None or not pose.objects_near_hands:
            return
        box = status.get("box")
        if box is None:
            return
        x1, y1, x2, y2 = box
        mid_y = (y1 + y2) // 2
        for hand_name, _label, _score in pose.objects_near_hands:
            mx = x1 if "left" in hand_name else x2
            cv2.circle(frame, (mx, mid_y), 6, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_notice(frame, text):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x, y = 18, 34
        _blend_rect(frame, x - 10, y - th - 10, x + tw + 10, y + 8, (20, 20, 20), 0.6)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_target_hud(frame, status, origin_index: int, panel_width: int = 200,
                         selected: bool = False):
        """Dashboard-style panel per target: dark background, a colored
        identity accent bar, and right-aligned uppercase field/value rows
        -- stacked left-to-right across the top so multiple targets never
        overlap."""
        color = status.get("color", (0, 255, 0))

        rows = [("STATUS", status.get("activity", "unknown").upper())]

        distance_m = status.get("distance_m")
        if distance_m is not None:
            rows.append(("RANGE", f"{distance_m:.1f} M"))

        pose = status.get("pose")
        if pose is not None:
            rows.append(("LIMBS", "OK" if pose.all_core_limbs_visible else "PARTIAL"))
            if pose.objects_near_hands:
                rows.append(("HOLDING", "OBJECT"))

        emotion = status.get("emotion")
        if emotion is not None:
            rows.append(("EMOTION", emotion.upper()))

        gallery_size = status.get("gallery_size")
        if gallery_size is not None:
            rows.append(("GALLERY", f"{gallery_size}/{REID_GALLERY_MAX_SIZE}"))

        ox = 15 + origin_index * (panel_width + 10)
        oy = 45
        line_h = 20
        header_h = 24
        panel_h = header_h + line_h * len(rows) + 10

        _blend_rect(frame, ox, oy, ox + panel_width, oy + panel_h, (18, 18, 18), 0.55)

        cv2.rectangle(frame, (ox, oy), (ox + 4, oy + panel_h), color, -1)
        if selected:
            cv2.rectangle(frame, (ox, oy), (ox + panel_width, oy + panel_h), color, 1, cv2.LINE_AA)

        header = f"TARGET {status['id']:02d}" + ("  [SEL]" if selected else "")
        if not status.get("manual", True):
            header += "  AUTO"
        cv2.putText(frame, header, (ox + 12, oy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.line(frame, (ox + 12, oy + header_h), (ox + panel_width - 8, oy + header_h),
                 (90, 90, 90), 1, cv2.LINE_AA)

        y = oy + header_h + 15
        for label, value in rows:
            cv2.putText(frame, label, (ox + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (150, 150, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, value[:16], (ox + 88, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (235, 235, 235), 1, cv2.LINE_AA)
            y += line_h
