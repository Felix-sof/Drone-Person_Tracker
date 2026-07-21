"""
Orchestrates the full flow, frame by frame, for MULTIPLE simultaneous
targets:

  raw frame
    -> ego-motion compensation   (cancel camera shake/drift, shared)
    -> person detection          (YOLO, ONE shared pass for all targets)
    -> for each target (own gallery, own tracker, own activity/pose state):
         -> IOU-continuity match against detections NOT already claimed
            by another target this frame
         -> if that fails: Re-ID rescan against remaining unclaimed
            detections, with switch-margin hysteresis
         -> periodic pose analysis, emotion analysis, activity/posture
         -> periodic automatic gallery growth
    -> annotated output frame (each target in its own color, dashboard-
       style HUD panel, corner-bracket box instead of a solid rectangle)

The "claim" mechanism (a set of already-assigned detection boxes, cleared
every frame) is what prevents two targets from locking onto the same
physical person, and what lets each target track independently without
stepping on the others.
"""

import cv2
import numpy as np

from config import (
    AUTO_GALLERY_NOVELTY_MAX_SIMILARITY,
    AUTO_GALLERY_UPDATE_INTERVAL,
    DUPLICATE_TARGET_SIMILARITY_THRESHOLD,
    EMOTION_ANALYSIS_INTERVAL,
    ENABLE_AUTO_GALLERY_UPDATE,
    ENABLE_EMOTION_ANALYSIS,
    ENABLE_MOTION_COMPENSATION,
    ENABLE_TILED_DETECTION,
    MAX_CONCURRENT_TARGETS,
    POSE_ANALYSIS_INTERVAL,
    REID_GALLERY_MAX_SIZE,
    REID_RESCAN_INTERVAL,
    REID_SWITCH_MARGIN,
    TRACKER_MAX_MISSED_FRAMES,
)
from src.detection import PersonDetector
from src.emotion import EmotionAnalyzer
from src.motion_compensation import EgoMotionCompensator
from src.pose_analysis import PoseAnalyzer
from src.reid import ReIDEmbedder, best_match
from src.target import Target
from src.tracker import iou


class AddTargetResult:
    def __init__(self, ok: bool, target_id: int | None, message: str):
        self.ok = ok
        self.target_id = target_id
        self.message = message


class DronePersonTrackingPipeline:
    def __init__(self):
        self.detector = PersonDetector()
        self.embedder = ReIDEmbedder()
        self.motion_compensator = EgoMotionCompensator()
        self.pose_analyzer = PoseAnalyzer()
        self.emotion_analyzer = EmotionAnalyzer()

        self.targets: list[Target] = []
        self._next_target_id = 1
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Target management
    # ------------------------------------------------------------------
    def add_target(self, frame_bgr: np.ndarray) -> AddTargetResult:
        """Captures the largest person in frame_bgr as a brand-new target,
        UNLESS they already match an existing target's gallery closely
        enough to be considered the same physical person (in which case
        creation is refused -- use add_angle() for that existing target
        instead)."""
        if len(self.targets) >= MAX_CONCURRENT_TARGETS:
            return AddTargetResult(False, None,
                                    f"Already tracking the max of {MAX_CONCURRENT_TARGETS} targets.")

        detections = self.detector.detect(frame_bgr)
        if not detections:
            return AddTargetResult(False, None, "No person detected in this frame.")

        largest = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        x1, y1, x2, y2 = largest.box
        crop = frame_bgr[y1:y2, x1:x2]
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

        new_id = self._next_target_id
        self._next_target_id += 1
        self.targets.append(Target(new_id, embedding))
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

        largest = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        x1, y1, x2, y2 = largest.box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return AddTargetResult(False, None, "No person detected in this frame.")

        embedding = self.embedder.embed(crop)
        target.gallery.append(embedding)
        if len(target.gallery) > REID_GALLERY_MAX_SIZE:
            target.gallery.pop(0 if len(target.gallery) == 1 else 1)
        return AddTargetResult(True, target_id, f"Angle added to target #{target_id}.")

    def remove_target(self, target_id: int) -> bool:
        before = len(self.targets)
        self.targets = [t for t in self.targets if t.id != target_id]
        return len(self.targets) < before

    def get_target(self, target_id: int) -> Target | None:
        return self._find_target(target_id)

    def _find_target(self, target_id: int) -> Target | None:
        for t in self.targets:
            if t.id == target_id:
                return t
        return None

    def _find_duplicate_target(self, embedding: np.ndarray) -> int | None:
        for target in self.targets:
            gallery_array = np.stack(target.gallery)
            max_sim = float(np.max(gallery_array @ embedding))
            if max_sim >= DUPLICATE_TARGET_SIMILARITY_THRESHOLD:
                return target.id
        return None

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray):
        self._frame_count += 1

        warped_frame, _affine = (
            self.motion_compensator.step(frame_bgr) if ENABLE_MOTION_COMPENSATION
            else (frame_bgr, None)
        )
        detections = (self.detector.detect_tiled(warped_frame) if ENABLE_TILED_DETECTION
                      else self.detector.detect(warped_frame))

        claimed_boxes: set = set()
        statuses = []

        for target in self.targets:
            status = self._update_target(target, warped_frame, detections, claimed_boxes)
            statuses.append(status)

        annotated = self._annotate(warped_frame, statuses)
        return annotated, statuses

    def _update_target(self, target: Target, warped_frame, detections, claimed_boxes: set) -> dict:
        status = {"id": target.id, "color": target.color, "box": None, "score": None,
                  "activity": "unknown", "pose": None, "emotion": None,
                  "gallery_size": len(target.gallery)}

        was_already_locked = target.tracker.is_locked and not target.tracker.lost()
        matched_box = None

        # 1) Try IOU continuity against detections no one has claimed yet.
        if was_already_locked:
            available = [d for d in detections if d.box not in claimed_boxes]
            best_iou, best_box = 0.0, None
            for d in available:
                score = iou(target.tracker.current_box, d.box)
                if score > best_iou:
                    best_iou, best_box = score, d.box
            if best_iou >= target.tracker.iou_threshold:
                matched_box = best_box
                target.tracker.current_box = best_box
                target.tracker.missed_frames = 0
            else:
                target.tracker.missed_frames += 1
                if target.tracker.missed_frames > TRACKER_MAX_MISSED_FRAMES:
                    target.tracker.is_locked = False
                    target.tracker.current_box = None

        # 2) Re-ID rescan (recovery, or periodic confirmation) against
        #    whatever's still unclaimed.
        needs_rescan = (
            matched_box is None
            and (target.tracker.lost()
                 or (self._frame_count + target.id) % REID_RESCAN_INTERVAL == 0)
        )
        if needs_rescan:
            available = [d for d in detections if d.box not in claimed_boxes]
            crops = [warped_frame[d.box[1]:d.box[3], d.box[0]:d.box[2]] for d in available]
            valid = [(d, c) for d, c in zip(available, crops) if c.size > 0]
            if valid:
                candidate_embs = self.embedder.embed_batch([c for _, c in valid])
                gallery_array = np.stack(target.gallery)
                idx, score = best_match(gallery_array, candidate_embs)
                status["score"] = score

                if idx is not None:
                    should_switch = (
                        not was_already_locked
                        or target.locked_score is None
                        or score >= target.locked_score + REID_SWITCH_MARGIN
                    )
                    if should_switch:
                        matched_box = valid[idx][0].box
                        target.tracker.initialize(matched_box)
                        target.locked_score = score

        if matched_box is not None:
            claimed_boxes.add(matched_box)

        status["box"] = matched_box

        if matched_box is None:
            target.last_pose_result = None
            target.last_emotion = None
            target.locked_score = None
            target.activity_classifier.reset()
            target.posture_tracker.reset()
            return status

        # Automatic gallery growth (per target, same safety rule: only while
        # confidently tracking with no missed frames).
        if (ENABLE_AUTO_GALLERY_UPDATE
                and target.tracker.missed_frames == 0
                and (self._frame_count + target.id) % AUTO_GALLERY_UPDATE_INTERVAL == 0):
            self._maybe_auto_add_to_gallery(target, warped_frame, matched_box)
            status["gallery_size"] = len(target.gallery)

        # Stagger expensive per-target work across frames using each
        # target's own ID as a phase offset. Without this, ALL targets'
        # heavy analysis (pose model, emotion model) lands on the exact
        # same frame every POSE_ANALYSIS_INTERVAL/EMOTION_ANALYSIS_INTERVAL
        # frames, creating a periodic latency spike that feels like
        # stuttering. Staggering spreads that cost evenly across frames
        # instead, for smoother (if slightly less fresh) playback.
        if (self._frame_count + target.id) % POSE_ANALYSIS_INTERVAL == 0:
            target.last_pose_result = self.pose_analyzer.analyze(warped_frame, matched_box)
        status["pose"] = target.last_pose_result

        if (ENABLE_EMOTION_ANALYSIS and self.emotion_analyzer.available
                and (self._frame_count + target.id) % EMOTION_ANALYSIS_INTERVAL == 0):
            head_box = (target.last_pose_result.body_part_boxes.get("head")
                        if target.last_pose_result else None)
            if head_box is not None:
                hx1, hy1, hx2, hy2 = head_box
                head_crop = warped_frame[hy1:hy2, hx1:hx2]
                target.last_emotion = self.emotion_analyzer.analyze(head_crop)
        status["emotion"] = target.last_emotion

        speed_activity = target.activity_classifier.update(matched_box)
        is_sitting = target.last_pose_result.is_sitting if target.last_pose_result else None
        posture_label = target.posture_tracker.update(is_sitting)
        status["activity"] = posture_label if posture_label is not None else speed_activity

        return status

    def _maybe_auto_add_to_gallery(self, target: Target, frame_bgr, box) -> None:
        x1, y1, x2, y2 = box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return

        new_embedding = self.embedder.embed(crop)
        gallery_array = np.stack(target.gallery)
        max_similarity = float(np.max(gallery_array @ new_embedding))
        if max_similarity >= AUTO_GALLERY_NOVELTY_MAX_SIMILARITY:
            return

        if len(target.gallery) >= REID_GALLERY_MAX_SIZE:
            if len(target.gallery) > 1:
                target.gallery.pop(1)
            else:
                return
        target.gallery.append(new_embedding)

    # ------------------------------------------------------------------
    # Annotation -- clean "dashboard" style: corner-bracket boxes instead
    # of solid rectangles, a small nameplate tag, and a dark HUD panel per
    # target with aligned field/value rows (no per-body-part box clutter).
    # ------------------------------------------------------------------
    def _annotate(self, frame, statuses: list[dict]):
        out = frame.copy()
        any_locked = False

        for i, status in enumerate(statuses):
            box = status.get("box")
            color = status.get("color", (0, 255, 0))
            if box is None:
                continue
            any_locked = True

            self._draw_corner_brackets(out, box, color)
            self._draw_nameplate(out, box, status, color)
            self._draw_hand_markers(out, status, color)
            self._draw_target_hud(out, status, origin_index=i)

        if not statuses:
            self._draw_notice(out, "NO TARGETS -- PRESS 'N' TO ADD ONE")
        elif not any_locked:
            self._draw_notice(out, "SEARCHING...")

        return out

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
    def _draw_nameplate(frame, box, status, color):
        """Small filled tag above the box, e.g. 'T01  0.87' -- replaces the
        old large colored 'TARGET #1' text."""
        x1, y1, _x2, _y2 = box
        text = f"T{status['id']:02d}"
        if status.get("score") is not None:
            text += f"  {status['score']:.2f}"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
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
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - th - 10), (x + tw + 10, y + 8), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_target_hud(frame, status, origin_index: int, panel_width: int = 200):
        """Dashboard-style panel per target: dark background, a colored
        identity accent bar, and right-aligned uppercase field/value rows
        -- stacked left-to-right across the top so multiple targets never
        overlap."""
        color = status.get("color", (0, 255, 0))

        rows = [("STATUS", status.get("activity", "unknown").upper())]

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

        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + panel_width, oy + panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

        cv2.rectangle(frame, (ox, oy), (ox + 4, oy + panel_h), color, -1)

        cv2.putText(frame, f"TARGET {status['id']:02d}", (ox + 12, oy + 17),
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
