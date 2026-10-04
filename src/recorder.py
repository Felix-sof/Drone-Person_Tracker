"""
Session recording and post-mission export.

  VideoRecorder  -- writes the annotated output (exactly what the operator
                    sees) to a video file, for debriefs and evidence.
  TrackExporter  -- writes one CSV row per visible target per processed
                    frame, plus a JSON summary per target at the end
                    (first/last seen, how long they were visible, how far
                    they moved, dominant activity). This is what you hand
                    to whoever plans the next search leg, or load into a
                    notebook/GIS tool.

Coordinates are in the motion-COMPENSATED frame (the same frame the boxes
are drawn on), not raw camera pixels.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2

_CSV_FIELDS = [
    "frame", "time_s", "target_id", "state",
    "x1", "y1", "x2", "y2", "cx", "cy",
    "reid_score", "activity", "distance_m", "emotion", "limbs_visible", "holding_object",
]


class VideoRecorder:
    """Lazily opens the writer on the first frame, so the output size always
    matches whatever actually comes out of the pipeline (rotation, source
    resolution) instead of having to be known up front."""

    def __init__(self, path: str, fps: float = 25.0, fourcc: str = "mp4v"):
        self.path = Path(path)
        self.fps = fps if fps and fps > 1 else 25.0
        self.fourcc = fourcc
        self._writer = None
        self.frames_written = 0

    def write(self, frame) -> None:
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {self.path}")
        self._writer.write(frame)
        self.frames_written += 1

    @property
    def active(self) -> bool:
        return self._writer is not None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class _TargetSummary:
    def __init__(self, target_id: int):
        self.target_id = target_id
        self.first_seen_frame = None
        self.last_seen_frame = None
        self.first_seen_s = None
        self.last_seen_s = None
        self.frames_visible = 0
        self.path_length_px = 0.0
        self._last_point = None
        self.activities = Counter()
        self.emotions = Counter()
        self.min_distance_m = None
        self.max_distance_m = None
        self.best_reid_score = None
        self.last_box = None

    def add(self, frame_idx, time_s, status) -> None:
        box = status["box"]
        if self.first_seen_frame is None:
            self.first_seen_frame, self.first_seen_s = frame_idx, time_s
        self.last_seen_frame, self.last_seen_s = frame_idx, time_s
        self.frames_visible += 1
        self.last_box = list(box)

        point = ((box[0] + box[2]) / 2.0, box[3])
        if self._last_point is not None:
            self.path_length_px += math.dist(point, self._last_point)
        self._last_point = point

        if status.get("activity"):
            self.activities[status["activity"]] += 1
        if status.get("emotion"):
            self.emotions[status["emotion"]] += 1
        d = status.get("distance_m")
        if d is not None:
            self.min_distance_m = d if self.min_distance_m is None else min(self.min_distance_m, d)
            self.max_distance_m = d if self.max_distance_m is None else max(self.max_distance_m, d)
        s = status.get("score")
        if s is not None:
            self.best_reid_score = s if self.best_reid_score is None else max(self.best_reid_score, s)

    def to_dict(self) -> dict:
        def r(v, nd=2):
            return round(v, nd) if isinstance(v, float) else v
        return {
            "target_id": self.target_id,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "first_seen_s": r(self.first_seen_s, 3),
            "last_seen_s": r(self.last_seen_s, 3),
            "frames_visible": self.frames_visible,
            "path_length_px": r(self.path_length_px, 1),
            "dominant_activity": self.activities.most_common(1)[0][0] if self.activities else None,
            "activity_breakdown": dict(self.activities),
            "dominant_emotion": self.emotions.most_common(1)[0][0] if self.emotions else None,
            "min_distance_m": r(self.min_distance_m),
            "max_distance_m": r(self.max_distance_m),
            "best_reid_score": r(self.best_reid_score, 3),
            "last_box": self.last_box,
        }


class TrackExporter:
    """
    Usage:
        exporter = TrackExporter("out/tracks.csv")
        exporter.add_frame(frame_idx, time_s, statuses)   # every processed frame
        exporter.close()   # also writes out/tracks.summary.json
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.summary_path = self.csv_path.with_suffix(".summary.json")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()
        self._summaries: dict[int, _TargetSummary] = {}
        self.rows_written = 0
        self._last_time_s = None

    def add_frame(self, frame_idx: int, time_s: float | None, statuses: list[dict]) -> None:
        if self._file is None:
            return
        self._last_time_s = time_s
        for status in statuses:
            box = status.get("box")
            if box is None:
                continue
            pose = status.get("pose")
            x1, y1, x2, y2 = (int(v) for v in box)
            self._writer.writerow({
                "frame": frame_idx,
                "time_s": f"{time_s:.3f}" if time_s is not None else "",
                "target_id": status["id"],
                "state": status.get("state", "tracking"),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                "reid_score": f"{status['score']:.3f}" if status.get("score") is not None else "",
                "activity": status.get("activity", ""),
                "distance_m": (f"{status['distance_m']:.2f}"
                               if status.get("distance_m") is not None else ""),
                "emotion": status.get("emotion") or "",
                "limbs_visible": ("" if pose is None else int(pose.all_core_limbs_visible)),
                "holding_object": ("" if pose is None else int(bool(pose.objects_near_hands))),
            })
            self.rows_written += 1
            summary = self._summaries.get(status["id"])
            if summary is None:
                summary = self._summaries[status["id"]] = _TargetSummary(status["id"])
            summary.add(frame_idx, time_s, status)

    def summary(self) -> dict:
        return {
            "targets": [s.to_dict() for s in sorted(self._summaries.values(),
                                                    key=lambda s: s.target_id)],
            "total_targets": len(self._summaries),
            "rows": self.rows_written,
        }

    def close(self) -> None:
        if self._file is None:
            return
        self._file.close()
        self._file = None
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2, ensure_ascii=False)
