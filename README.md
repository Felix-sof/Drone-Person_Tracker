# Drone Person Tracker (Prototype)

A computer vision system that recognizes a person from a reference photo and then finds
and tracks them in a live/recorded camera feed. Built around a search-and-rescue scenario:
"upload a photo of the missing person, have the system find and track them from the air."

## Architecture

```
Reference Photo ──► Person Detection ──► Re-ID Embedding ──► Reference Vector
                                                                     │
Live Frame ──► Ego-Motion Compensation ──► Person Detection ──► Re-ID Match ──► Tracker
```

### Why Ego-Motion Compensation?

A drone is never perfectly still (wind, vibration, flight motion), so in the raw footage
the background itself appears to "move," which confuses a tracker. `src/motion_compensation.py`
estimates the camera's own motion (a global affine transform) between consecutive frames
using sparse optical flow + RANSAC, and aligns the frame accordingly. Whatever motion
remains after this step is real object motion -- detection and tracking run on this
"cleaned" frame.

A few refinements on top of the base approach, covered by `tests/test_motion_compensation.py`:

- **Downscaled estimation** (`MOTION_COMP_ESTIMATION_WIDTH`): motion is estimated on a
  shrunk copy of the frame for speed, then the translation component is rescaled back up
  to full resolution before warping -- the rotation/scale part of the affine is left
  untouched, only translation needs rescaling.
- **Target-exclusion mask**: corner features are NOT picked from inside a currently
  tracked target's bounding box, so a moving person can't corrupt the background-motion
  estimate (which is supposed to reflect the STATIC scene, not the thing that's supposed
  to be moving).
- **Safety crop after warping**: `warpAffine` stretches replicate-padded edge pixels into
  thin streaks near the frame border; a small crop discards that visible artifact rather
  than showing it to the user.
- **Cumulative rotation cap** (`MAX_CUMULATIVE_ROTATION_DEG`): several individually-small,
  same-direction rotation estimates can compound into a frame that's visibly rotated by
  tens of degrees over time, even though each single-frame estimate looked plausible on
  its own. The cumulative transform's rotation is checked every frame (not just at the
  periodic re-anchor interval) and reset the moment it exceeds this cap.

## Tactical Control Panel

Press `h` to toggle an on-screen operator console (`src/control_panel.py`) listing every
keyboard shortcut plus live system status -- how many targets are locked, paused/active
state, and whether motion compensation / tiling / distance estimation are currently on.
Hidden by default so it doesn't clutter the view; costs nothing when hidden. Labels are
English-only on purpose: `cv2.putText`'s built-in Hershey fonts don't render non-ASCII
characters cleanly, so this keeps the panel crisp at small sizes.

## Config Validation

`src/config_validation.py` runs a set of startup sanity checks on `config.py` (call
`validate_config()` once before the pipeline starts processing frames). This catches
config mistakes that would otherwise fail silently and produce a plausible-but-wrong
result -- e.g. an out-of-range `CAMERA_VERTICAL_FOV_DEG` silently producing a wrong
distance estimate instead of an obvious error at startup. Covers distance-estimation
settings, Re-ID similarity thresholds, tiling settings, and a few runtime/frame settings.

## Tests

```bash
pip install pytest
pytest tests/
```

`tests/test_motion_compensation.py` covers the downscaled-estimation rescaling math, the
target-exclusion mask, the post-warp safety crop, and the cumulative-rotation-cap
regression (see above) -- these are geometry/plumbing checks, not full optical-flow
integration tests, so they don't need a real camera frame.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

On first run, `ultralytics` will automatically download the `yolov8n.pt` weights.

## Running

```bash
# Start with a reference photo
python app/main.py --reference data/reference/person.jpg

# Or start the webcam and press 'r' to capture the reference live
python app/main.py
```

Keys: `n` = add a new target, `a` = add another angle to the selected target, `1`-`9` =
select a target, `x` = remove the selected target, `p` = pause/resume (video files only),
`q` = quit.

## Multi-Target Tracking

The system can track several people at once (default cap of 5, `MAX_CONCURRENT_TARGETS`),
each independently. Every target has its own gallery, its own tracker, and its own
activity/posture state -- they don't interfere with each other. To prevent the same
physical person from being counted as two different targets, adding a new target (`n`)
first checks whether they already match an existing target's gallery
(`DUPLICATE_TARGET_SIMILARITY_THRESHOLD`); if so, creation is refused and the existing
target's ID is reported instead -- use `a` to add an angle to that target.

Once a person is "claimed" by a target in a given frame, no other target can claim the
same box in that same frame -- this is what prevents double-counting.

**Selecting the right person for `n` / `a`:** with several people in frame, "who does
`n` (add target) or `a` (add angle) actually pick?" needs to be unambiguous. `add_target`
excludes any detection that overlaps an existing target's current box (IoU >= 0.3) before
picking the largest remaining person -- otherwise it could re-select an already-tracked
person instead of the new one, and falsely refuse with "already tracked". `add_angle`
matches strictly against the TARGET'S OWN last-known box (also via IoU) rather than the
largest person in frame -- otherwise, with multiple targets on screen, it could silently
add a different person's embedding into the wrong target's gallery and corrupt future
Re-ID matches for both targets involved.

## Module Layout

| File | Responsibility |
|---|---|
| `src/detection.py` | Person detection via YOLO |
| `src/reid.py` | Embedding extraction from a person crop (real Re-ID via OSNet / ResNet18 fallback) + gallery matching |
| `src/motion_compensation.py` | Camera motion estimation and compensation (toggleable, see `ENABLE_MOTION_COMPENSATION`) |
| `src/tracker.py` | Lightweight IOU-based frame-to-frame tracking (avoids running Re-ID every frame) |
| `src/target.py` | Independent per-target state (gallery, tracker, activity/posture) |
| `src/activity.py` | Movement state based on relative speed (still/walking/running) |
| `src/posture.py` | Tracks the sitting/rising transition over time |
| `src/pose_analysis.py` | Limb visibility, hand-held object detection, sitting estimation, body-part boxes |
| `src/emotion.py` | Facial expression analysis (DeepFace, FER2013-based) |
| `src/distance.py` | Rough monocular distance estimation from apparent box height |
| `src/control_panel.py` | Toggleable on-screen operator console (shortcuts + live system status) |
| `src/config_validation.py` | Startup sanity checks for `config.py` values |
| `src/pipeline.py` | Multi-target orchestration -- shared detection, per-target Re-ID/tracking/analysis |
| `app/main.py` | Live demo over webcam/video, multi-target controls |

### Choosing a Re-ID Backend

In `config.py`, `REID_BACKEND = "osnet"` (default) uses a real Re-ID model (OSNet via the
`torchreid` package, trained on person-specific datasets like Market-1501). If `torchreid`
isn't installed or weight download fails, the system automatically falls back to
`"resnet18"` (general-purpose, weaker, but has no extra dependency).

### Activity and Pose Analysis Limitations

- **Movement state**: based on relative pixel speed, NOT real-world speed (a monocular
  camera has no depth/scale reference). Thresholds may need retuning as distance to the
  camera changes (`ACTIVITY_WALK_THRESHOLD_HEIGHTS_S`, `ACTIVITY_RUN_THRESHOLD_HEIGHTS_S`).
- **Limb visibility**: a keypoint failing to be detected can be caused by camera angle,
  shadow, or occlusion -- this is NOT a medical injury diagnosis, only "not visible to the
  camera right now" information. Interpretation should always be left to a human operator.
- **Hand-held object detection**: runs on a small region around the wrist; this is a rough
  proximity signal, not a verified "grasp" confirmation. There's also a contour-based shape
  heuristic for long/thin objects (sticks, flag poles) that COCO has no class for -- this
  flags object SHAPE only, not identity.
- **Sitting/rising detection**: based on the hip-knee-ankle angle (a narrow angle = sitting).
  Can be fooled by extreme camera angles (heavily side-on or top-down shots); this is not a
  calibrated measurement.
- **Body-part boxes**: separate boxes for head/torso/arms/legs, computed from keypoint
  positions. If a keypoint isn't confidently visible, no box is drawn for that part (so you
  won't always see all six boxes -- that's expected).
- **Facial expression (emotion) analysis**: uses DeepFace's pretrained FER2013-based model,
  7 categories (angry/disgust/fear/happy/sad/surprise/neutral). This is a known-imperfect,
  general-purpose classifier -- not a definitive psychological assessment, and should be
  treated as a rough signal for a human operator to interpret. If `deepface` isn't installed
  or fails to load, the feature disables itself silently; the app doesn't crash.

## Small/Distant Object Support (Tiling)

When a drone shoots from altitude, a person can occupy very few pixels, and YOLO
downscaling the whole frame in one pass can miss the detection. Setting
`ENABLE_TILED_DETECTION = True` in `config.py` makes the system split each frame into
overlapping tiles, scan each tile separately, then merge the results (deduplicating with
NMS). This improves detection quality but **noticeably slows things down**, since multiple
YOLO calls run per frame. Off by default -- there's no benefit for something like a webcam
where the person already appears large, only added latency. Recommended when testing with
high-altitude drone footage or datasets like VisDrone.

Related settings:
- `TILE_SIZE_PX`: pixel size of each tile (default 640, YOLO's native input size)
- `TILE_OVERLAP_RATIO`: overlap ratio between neighboring tiles (so edge objects aren't split)
- `TILING_NMS_IOU_THRESHOLD`: threshold for merging duplicate detections in overlapping regions

## Distance Estimation

With `ENABLE_DISTANCE_ESTIMATION = True` (default) in `config.py`, a rough distance
estimate (e.g. "15.3M") is shown in the HUD panel for each target. Method: the classic
pinhole-camera approximation -- distance is computed from the target's apparent pixel
height on screen, the camera's vertical field of view (FOV), and an assumed person height
(1.7m).

**Important:** you need to set `CAMERA_VERTICAL_FOV_DEG` to match **your actual camera** --
the wrong FOV silently produces a plausible-looking but wrong distance. This method also
assumes the target is standing upright and fully visible; someone crouching, sitting, or
partially visible will appear farther away than they really are (because their apparent
height shrinks, but not due to distance).

## Thermal / Infrared Camera Support

The video input layer (`app/main.py --video`) accepts anything `cv2.VideoCapture`
understands -- this includes a thermal camera that exposes itself as a standard UVC
webcam, so it can be connected at the I/O level with no extra code changes. **However**,
here's what doesn't work: every detection/pose/Re-ID model in this project is trained on
ordinary RGB imagery. In thermal frames, people appear as undifferentiated bright blobs
(no clothing color/texture), which measurably degrades off-the-shelf RGB model accuracy
(published results on datasets like Teledyne FLIR ADAS demonstrate this). Adding real
thermal support is a model-training project (e.g. fine-tuning YOLO on a thermal dataset),
not a config flag -- not implemented yet, noted here as a future direction.

## Known Limitations / Next Steps

- **Real drone feed**: an RTSP stream URL can be used instead of
  `cv2.VideoCapture(CAMERA_INDEX)` to connect to an actual drone video feed. The
  `app/main.py --video` argument supports both local video files and RTSP/HTTP stream URLs.
- **Multiple similar-looking people**: if someone else is wearing the same color clothing,
  there's a risk of mismatching; adjusting `REID_MATCH_THRESHOLD` and/or adding another
  feature (e.g. gait) may be needed.
- **Thermal camera**: as described above, supported at the I/O level but needs
  thermal-specific training for real model accuracy.