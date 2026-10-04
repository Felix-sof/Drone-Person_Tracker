"""
Tests for session export (src/recorder.py), video-clock activity
classification (src/activity.py), and the non-blocking DB logger (src/db.py).
"""

import csv
import json
import time

import numpy as np

from src.activity import ActivityClassifier
from src.recorder import TrackExporter, VideoRecorder


def _status(target_id, box, activity="walking", distance=None, score=None):
    return {"id": target_id, "box": box, "state": "tracking", "activity": activity,
            "distance_m": distance, "score": score, "emotion": None, "pose": None}


def test_track_exporter_writes_csv_and_summary(tmp_path):
    exporter = TrackExporter(str(tmp_path / "tracks.csv"))
    exporter.add_frame(1, 0.0, [_status(1, (0, 0, 10, 20), distance=12.0, score=0.8),
                                {"id": 2, "box": None, "state": "coasting"}])
    exporter.add_frame(2, 0.1, [_status(1, (3, 4, 13, 24), activity="running", distance=10.0)])
    exporter.add_frame(3, 0.2, [_status(1, (6, 8, 16, 28), activity="running")])
    exporter.close()

    with open(tmp_path / "tracks.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3                       # the box=None status is skipped
    assert rows[0]["target_id"] == "1" and rows[0]["reid_score"] == "0.800"

    summary = json.loads((tmp_path / "tracks.summary.json").read_text(encoding="utf-8"))
    t1 = summary["targets"][0]
    assert summary["total_targets"] == 1
    assert t1["frames_visible"] == 3
    assert t1["first_seen_s"] == 0.0 and t1["last_seen_s"] == 0.2
    assert t1["dominant_activity"] == "running"
    assert t1["min_distance_m"] == 10.0 and t1["max_distance_m"] == 12.0
    assert t1["path_length_px"] == 10.0         # two 3-4-5 steps of the bottom-center point


def test_video_recorder_writes_frames(tmp_path):
    path = tmp_path / "out" / "run.avi"
    rec = VideoRecorder(str(path), fps=10, fourcc="MJPG")
    for i in range(5):
        rec.write(np.full((48, 64, 3), i * 40, dtype=np.uint8))
    rec.close()
    assert rec.frames_written == 5
    assert path.exists() and path.stat().st_size > 0


def test_activity_uses_video_clock_not_wall_clock():
    """A box moving 1 box-height per VIDEO second is 'walking' -- even if
    the frames are fed in instantly (analysis faster than real time) or
    slowly (analysis slower than real time)."""
    clf = ActivityClassifier()
    label = None
    for i in range(10):
        t = i * 0.1
        y_shift = int(100 * t)            # box height 100 -> 1.0 heights/s
        label = clf.update((0, y_shift, 40, 100 + y_shift), timestamp=t)
    assert label == "walking"


def test_activity_resets_when_video_time_goes_backwards():
    clf = ActivityClassifier()
    for i in range(5):
        clf.update((i * 50, 0, 40 + i * 50, 100), timestamp=i * 0.1)
    # Loop back to t=0 -- must not compute a negative/garbage speed.
    assert clf.update((0, 0, 40, 100), timestamp=0.0) == "unknown"


def test_db_logging_never_blocks(monkeypatch):
    """With the DB unreachable / client missing, log calls must return
    immediately (work happens on a background thread)."""
    import src.db as db

    logger = db._AsyncMySQLLogger()
    monkeypatch.setattr(logger, "_get_connection", lambda: (time.sleep(0.5), None)[1])
    monkeypatch.setattr(db, "_logger", logger)
    logger._disabled = False

    start = time.perf_counter()
    for i in range(50):
        db.log_target_event(1, "walking", None, None, 3.0, (0, 0, 10, 10))
    assert time.perf_counter() - start < 0.2
    logger.shutdown(timeout=2.0)


def test_db_queue_drops_instead_of_growing(monkeypatch):
    import src.db as db

    logger = db._AsyncMySQLLogger()
    logger._disabled = False
    monkeypatch.setattr(logger, "_ensure_worker", lambda: None)   # nothing drains
    for _ in range(db._QUEUE_MAX + 10):
        logger.submit((1,))
    assert logger.dropped_events == 10
