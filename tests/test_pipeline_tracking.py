"""
End-to-end tests of the multi-target association logic in src/pipeline.py,
using lightweight fakes instead of the real YOLO / Re-ID / pose / emotion
models (the pipeline accepts injected components), so these run in
milliseconds with no weights or GPU.

The fake Re-ID embedder describes a crop by which fraction of its pixels
match each known "clothing" color, so each synthetic person (a solid-color
rectangle) has its own identity -- distinct colors are orthogonal, and a
partially occluded crop gets a proportional mixture, like a real embedding.
"""

import numpy as np
import pytest

import src.pipeline as pipeline_mod
from src.detection import Detection
from src.pipeline import DronePersonTrackingPipeline, _gated_assignment

FRAME_H, FRAME_W = 480, 640
RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)
PALETTE = np.array([RED, GREEN, BLUE, (0, 255, 255), (255, 0, 255), (255, 255, 0),
                    (128, 128, 255), (255, 128, 128), (128, 255, 128)], dtype=np.uint8)


class FakeDetector:
    def __init__(self):
        self.boxes = []

    def detect(self, frame):
        return [Detection(box=b, confidence=0.9) for b in self.boxes]

    detect_tiled = detect


class FakeEmbedder:
    def embed_batch(self, crops):
        out = []
        for c in crops:
            pixels = c.reshape(-1, 1, 3)
            v = (pixels == PALETTE[None]).all(axis=-1).mean(axis=0).astype(np.float64) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.array(out)

    def embed(self, crop):
        return self.embed_batch([crop])[0]


class FakeMotion:
    def step(self, frame, target_boxes=None):
        return frame, None


class FakePose:
    def analyze(self, frame, box):
        return None


class FakeEmotion:
    available = False


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Never touch a real database from tests.
    monkeypatch.setattr(pipeline_mod, "log_target_event", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_mod, "ENABLE_MOTION_COMPENSATION", False)
    monkeypatch.setattr(pipeline_mod, "ENABLE_DISTANCE_ESTIMATION", False)
    monkeypatch.setattr(pipeline_mod, "AUTO_TARGET_MIN_HITS", 3)
    monkeypatch.setattr(pipeline_mod, "LOST_TARGET_EXPIRY_FRAMES", 5)


def _make_pipeline():
    detector = FakeDetector()
    p = DronePersonTrackingPipeline(detector=detector, embedder=FakeEmbedder(),
                                    motion_compensator=FakeMotion(), pose_analyzer=FakePose(),
                                    emotion_analyzer=FakeEmotion())
    return p, detector


def _frame(people):
    """people: list of (box, color)."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), color in people:
        frame[y1:y2, x1:x2] = color
    return frame


def _step(p, detector, people, ts):
    detector.boxes = [box for box, _ in people]
    _annotated, statuses = p.process_frame(_frame(people), timestamp=ts)
    return statuses


# ----------------------------------------------------------------------
def test_gated_assignment_beats_greedy_swap():
    """Greedy (row-by-row best) would give target 0 column 0 and leave
    target 1 unmatched; the global solution matches both."""
    ious = np.array([[0.60, 0.50],
                     [0.55, 0.10]])
    pairs = dict(_gated_assignment(ious, 0.3))
    assert pairs == {0: 1, 1: 0}


def test_gated_assignment_respects_threshold():
    assert _gated_assignment(np.array([[0.2, 0.1]]), 0.3) == []
    assert _gated_assignment(np.zeros((0, 3)), 0.3) == []


def test_auto_target_is_tentative_until_min_hits(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()
    person = [((100, 100, 140, 200), RED)]

    assert _step(p, det, person, 0.0) == []
    assert _step(p, det, person, 0.1) == []
    statuses = _step(p, det, person, 0.2)
    assert [s["id"] for s in statuses] == [1]
    assert statuses[0]["state"] == "tracking"


def test_one_frame_false_positive_never_consumes_an_id(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()

    _step(p, det, [((400, 50, 430, 120), BLUE)], 0.0)   # blip
    person = [((100, 100, 140, 200), RED)]
    for i in range(1, 5):
        statuses = _step(p, det, person, i * 0.1)
    assert [s["id"] for s in statuses] == [1]
    assert len(p.targets) == 1


def test_lost_auto_target_expires_and_frees_slot(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()
    person = [((100, 100, 140, 200), RED)]
    for i in range(3):
        _step(p, det, person, i * 0.1)
    assert len(p.confirmed_targets) == 1

    # Tracker gives up after TRACKER_MAX_MISSED_FRAMES (10), then the
    # expiry window (patched to 5) must elapse.
    for i in range(3, 25):
        _step(p, det, [], i * 0.1)
    assert p.targets == []


def test_manual_target_never_expires(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", False)
    p, det = _make_pipeline()
    person = [((100, 100, 140, 200), RED)]
    det.boxes = [person[0][0]]
    result = p.add_target(_frame(person))
    assert result.ok and result.target_id == 1

    for i in range(200):
        _step(p, det, [], i * 0.1)
    assert [t.id for t in p.targets] == [1]

    # ...and is re-acquired by Re-ID when they come back.
    statuses = _step(p, det, person, 30.0)
    assert statuses[0]["box"] == person[0][0]


def test_two_people_crossing_keep_their_ids(monkeypatch):
    """Two differently-dressed people walk toward each other and pass,
    their boxes overlapping on the way (one slightly in front of the
    other, like two people on a path seen from above-behind). IDs must
    stay attached to the right person throughout."""
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()

    def people_at(step):
        a_x = 60 + 18 * step
        b_x = 520 - 18 * step
        return [((a_x, 150, a_x + 40, 260), RED), ((b_x, 220, b_x + 40, 330), GREEN)]

    color_of_id = {}
    for step in range(26):
        people = people_at(step)
        frame = _frame(people)
        statuses = _step(p, det, people, step * 0.1)
        for s in statuses:
            if s["box"] is None:
                continue
            x1, y1, x2, y2 = s["box"]
            color = tuple(int(c) for c in frame[(y1 + y2) // 2, (x1 + x2) // 2])
            color_of_id.setdefault(s["id"], color)
            assert color_of_id[s["id"]] == color, f"ID {s['id']} swapped at step {step}"

    assert set(color_of_id) == {1, 2}


def test_fast_mover_is_auto_confirmed(monkeypatch):
    """A person moving ~1 box-width per frame fails plain IOU every frame;
    gated Re-ID continuation must let the tentative track survive long
    enough to be confirmed, and then keep a single ID."""
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()
    ids_seen = set()
    for step in range(12):
        x = 40 + 40 * step
        statuses = _step(p, det, [((x, 150, x + 40, 250), RED)], step * 0.1)
        ids_seen.update(s["id"] for s in statuses if s["box"] is not None)
    assert ids_seen == {1}


def test_coasting_through_short_occlusion_keeps_state(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", False)
    p, det = _make_pipeline()
    person = [((100, 100, 140, 200), RED)]
    det.boxes = [person[0][0]]
    p.add_target(_frame(person))

    for i in range(5):
        _step(p, det, person, i * 0.1)
    statuses = _step(p, det, [], 0.5)
    assert statuses[0]["state"] == "coasting"
    assert statuses[0]["predicted_box"] is not None
    assert len(p.targets[0].trail) == 5          # history kept, not wiped

    statuses = _step(p, det, person, 0.6)
    assert statuses[0]["state"] == "tracking"


def test_max_concurrent_targets_cap(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    monkeypatch.setattr(pipeline_mod, "MAX_CONCURRENT_TARGETS", 2)
    p, det = _make_pipeline()
    people = [((50, 100, 90, 200), RED), ((200, 100, 240, 200), GREEN),
              ((400, 100, 440, 200), BLUE)]
    for i in range(4):
        statuses = _step(p, det, people, i * 0.1)
    assert len(p.targets) == 2
    assert len(statuses) == 2


def test_annotate_handles_many_targets_without_error(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()
    people = [((10 + 75 * i, 300, 50 + 75 * i, 400), tuple(int(v) for v in PALETTE[i]))
              for i in range(8)]
    for i in range(4):
        _step(p, det, people, i * 0.1)
    annotated, statuses = p.process_frame(_frame(people), timestamp=0.5, selected_id=3)
    assert annotated.shape == (FRAME_H, FRAME_W, 3)
    assert len(statuses) == 8


class PanningMotion:
    """Fake camera-motion estimator: the scene shifts by (dx, dy) per frame."""
    def __init__(self, dx, dy):
        self.affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        self.first = True

    def estimate(self, frame, target_boxes=None):
        if self.first:
            self.first = False
            return None
        return self.affine


def test_moving_drone_static_people_read_still_and_keep_ids(monkeypatch):
    """The drone flies along the field, so every (standing) person slides
    up the image by 45px per processed frame -- more than their own box
    width. Track-level camera motion compensation must keep each ID on
    its person AND must not report them as moving."""
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    monkeypatch.setattr(pipeline_mod, "ENABLE_MOTION_COMPENSATION", True)
    monkeypatch.setattr(pipeline_mod, "MOTION_COMP_MODE", "tracks")
    det = FakeDetector()
    p = DronePersonTrackingPipeline(detector=det, embedder=FakeEmbedder(),
                                    motion_compensator=PanningMotion(0, -45),
                                    pose_analyzer=FakePose(), emotion_analyzer=FakeEmotion())
    color_of_id = {}
    for step in range(9):
        y = 420 - 45 * step
        people = [((100, y, 140, y + 50), RED), ((300, y, 340, y + 50), GREEN)]
        frame = _frame(people)
        statuses = _step(p, det, people, step * 0.1)
        for s in statuses:
            if s["box"] is None:
                continue
            x1, y1, x2, y2 = s["box"]
            color = tuple(int(c) for c in frame[(y1 + y2) // 2, (x1 + x2) // 2])
            assert color_of_id.setdefault(s["id"], color) == color
    assert set(color_of_id) == {1, 2}
    assert all(s["activity"] == "still" for s in statuses), [s["activity"] for s in statuses]


def test_without_camera_compensation_static_people_look_like_running(monkeypatch):
    """Control for the test above: same footage, compensation off -> the
    camera's motion is (wrongly) measured as the people running. Shows the
    test actually exercises the compensation."""
    monkeypatch.setattr(pipeline_mod, "ENABLE_AUTO_TRACK_ALL", True)
    p, det = _make_pipeline()
    for step in range(9):
        y = 420 - 45 * step
        statuses = _step(p, det, [((100, y, 140, y + 50), RED)], step * 0.1)
    assert statuses and statuses[0]["activity"] == "running"
