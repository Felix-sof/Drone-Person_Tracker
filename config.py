"""
Central configuration for the drone person-tracking pipeline.
Keep all tunable thresholds here so experiments don't require touching pipeline code.
"""

# --- Detection ---
YOLO_MODEL = "yolov8n.pt"      # nano model: fastest, good enough for prototyping on CPU
YOLO_CONF_THRESHOLD = 0.4
PERSON_CLASS_ID = 0            # COCO class id for "person"

# --- Tiled ("sliced") detection for small/distant objects ---
# YOLO downscales the whole frame to a fixed input size (e.g. 640x640)
# before detecting. If a person is already small in a high-resolution
# frame (e.g. filmed from altitude), that downscale can shrink them past
# the point the model can recognize -- the detection just gets missed.
#
# When enabled, the frame is split into overlapping tiles and each tile is
# run through the detector separately; results are merged back into
# full-frame coordinates and deduplicated with NMS. This trades speed
# (several detector calls per frame instead of one) for the ability to
# catch small/distant people. Leave off for webcam use (person already
# fills a large part of the frame, tiling only adds latency for no gain);
# turn on for footage where people are small (e.g. high-altitude drone
# video, like VisDrone-style clips).
ENABLE_TILED_DETECTION = False
TILE_SIZE_PX = 640
TILE_OVERLAP_RATIO = 0.2        # fraction of tile size overlapped between neighboring tiles
TILING_NMS_IOU_THRESHOLD = 0.45  # merges duplicate detections found in overlap regions

# --- Re-ID (re-identification) ---
# "resnet18"  -> general ImageNet backbone, no extra install, weaker matching.
# "osnet"     -> real person Re-ID model (torchreid), needs `pip install torchreid`.
#                Falls back to resnet18 automatically if torchreid isn't installed.
REID_BACKEND = "osnet"
REID_MATCH_THRESHOLD = 0.72    # cosine similarity above this = "same person" candidate
REID_RESCAN_INTERVAL = 15      # re-run full Re-ID scan every N frames to recover from occlusion
# While the tracker is successfully following a target via IOU (not lost),
# a periodic rescan can still find a DIFFERENT person who happens to score
# slightly higher that frame (motion blur, pose, lighting) and incorrectly
# steal the lock -- this is what "target randomly jumps to someone else"
# looks like. Require the new candidate to beat the currently tracked
# target's own current similarity by this margin before switching, so a
# marginal/noisy difference can't override a target that's tracking fine.
REID_SWITCH_MARGIN = 0.05
# Multi-shot reference gallery: instead of a single reference embedding,
# you can capture several (e.g. front view, back view, side view) of the
# same target -- a candidate is matched against whichever gallery shot it
# resembles most. Very useful once someone turns around and their back
# view no longer resembles a front-view-only reference. Capped so the
# gallery doesn't grow unbounded (oldest shot is dropped when full).
REID_GALLERY_MAX_SIZE = 5

# --- Automatic gallery growth ---
# Instead of relying only on manually pressing 'a', periodically capture
# the currently-tracked target's appearance and add it to the gallery
# automatically -- IF it looks meaningfully different from what's already
# stored (a new angle/pose). This lets the gallery fill in with front/side/
# back views on its own as the target moves, without you needing to catch
# the right moment.
#
# Safety: only auto-added while the IOU tracker has been confidently
# following the target with NO missed frames since its last confirmed
# match -- so a tracking drift onto the wrong person can't poison the
# gallery. The very first (manually captured) reference shot is never
# evicted when the gallery is full.
ENABLE_AUTO_GALLERY_UPDATE = True
AUTO_GALLERY_UPDATE_INTERVAL = 20             # frames between auto-capture attempts
AUTO_GALLERY_NOVELTY_MAX_SIMILARITY = 0.90    # skip if already too similar to a stored shot

# --- Multi-target tracking ---
MAX_CONCURRENT_TARGETS = 5
# When capturing what's supposed to be a NEW target, check it against every
# EXISTING target's gallery first. If it matches one this well, refuse to
# create a duplicate ID for the same physical person -- this is what
# prevents "the same person counted as two different targets".
DUPLICATE_TARGET_SIMILARITY_THRESHOLD = 0.75

# --- Activity / movement state ---
# Classification is based on how far the tracked box's center moves per
# second, expressed as a RATIO of the target's own box height (not raw
# pixels). This makes the thresholds work regardless of how far the camera
# is from the subject: someone running fills fewer raw pixels/second when
# small and distant (e.g. high-altitude drone footage) than when close to
# a webcam, but "box-heights per second" stays comparable in both cases.
ACTIVITY_HISTORY_SECONDS = 1.5
ACTIVITY_WALK_THRESHOLD_HEIGHTS_S = 0.5   # box-heights/second
ACTIVITY_RUN_THRESHOLD_HEIGHTS_S = 1.5    # box-heights/second
# Someone running STRAIGHT toward/away from the camera barely moves
# laterally -- only their box SIZE changes. That "radial" signal is
# naturally smaller in magnitude than lateral motion for the same real
# speed (basic perspective: the closer they already are, the more a given
# approach speed changes apparent size; far away, the same speed barely
# changes it). This scales the radial signal up before comparing it
# against the same walk/run thresholds as lateral motion. If runners
# heading toward the camera still register as "still" until they're
# quite close, raise this; if radial motion is over-triggering "running"
# on people who are just slightly changing distance, lower it.
ACTIVITY_RADIAL_SPEED_SCALE = 4.0

# --- Pose analysis (limb visibility + hand-object proximity) ---
POSE_MODEL = "yolov8n-pose.pt"
POSE_KEYPOINT_CONF_THRESHOLD = 0.5   # below this, a keypoint counts as "not visible"
# COCO-17 keypoint indices used to summarize "core limb" visibility.
POSE_LIMB_KEYPOINTS = {
    "left_wrist": 9, "right_wrist": 10,
    "left_ankle": 15, "right_ankle": 16,
    "left_elbow": 7, "right_elbow": 8,
    "left_knee": 13, "right_knee": 14,
}
# Wrist keypoint indices specifically, used for hand-object proximity checks.
POSE_WRIST_KEYPOINTS = {"left_wrist": 9, "right_wrist": 10}
HAND_OBJECT_ROI_RADIUS_PX = 45        # search radius around each wrist
# Pose analysis (limb visibility + hand-object check) is one of the most
# expensive per-target additions -- runs its own YOLO model call. With
# multiple targets tracked at once the cost multiplies (N targets = N calls
# every interval), so this is deliberately not-too-frequent. Raise further
# (e.g. 15-20) if you're tracking several targets at once and performance
# matters more than analysis freshness.
POSE_ANALYSIS_INTERVAL = 8

# --- Elongated hand-held object detection (e.g. stick, flag pole) ---
# COCO detection classes don't include "stick" or "flag", so long thin
# objects are caught with a geometric heuristic instead: look for an
# elongated contour near the wrist. This is approximate -- it flags SHAPE,
# not object identity -- and works best against a fairly plain background.
STICK_ASPECT_RATIO_THRESHOLD = 3.0   # long-side / short-side of the shape
STICK_MIN_LENGTH_PX = 40

# --- Posture (sitting / standing / rising) ---
# Estimated from the hip-knee-ankle angle: a bent knee (small angle) reads
# as sitting, a nearly straight leg reads as standing. Like the activity
# speed thresholds, this is a heuristic on 2D pixel geometry, not a
# calibrated biomechanical measurement -- extreme camera angles can fool it.
SITTING_KNEE_ANGLE_THRESHOLD_DEG = 130
# How many frames to keep showing "kalkıyor" (rising) after a sitting ->
# standing transition is detected, so the label doesn't just flash by.
POSTURE_TRANSITION_DISPLAY_FRAMES = 15

# --- Body-part specific boxes (head / torso / arms / legs) ---
BODY_PART_BOX_PADDING_PX = 12

# --- Facial expression (emotion) analysis ---
# Uses DeepFace's pretrained FER2013-based emotion model if the `deepface`
# package is installed; the feature disables itself gracefully (rather than
# crashing) if it isn't. This is a coarse, well-known-imperfect classifier
# (7 categories: angry, disgust, fear, happy, sad, surprise, neutral) --
# treat it as a rough signal for a human operator, not a certified reading
# of someone's mental state.
# Set to False to skip it entirely -- useful for performance when tracking
# several targets at once, or when faces are too small/distant to matter
# (e.g. high-altitude footage) and the analysis is just wasted computation.
ENABLE_EMOTION_ANALYSIS = True
EMOTION_ANALYSIS_INTERVAL = 25   # frames; this is the single most expensive add-on
                                  # (a separate deep model call) -- costs multiply
                                  # with multiple targets, raise further if slow
EMOTION_MIN_FACE_SIZE_PX = 40    # skip analysis on tiny/low-res face crops

# --- Ego-motion compensation ---
# Designed to cancel HOVER jitter (wind, vibration) -- small, roughly
# random frame-to-frame shake around a fixed position. It is NOT meant for
# footage where the camera is deliberately panning/flying in one direction
# for a sustained period (e.g. following runners, flying down a street):
# in that case the compensator tries to "undo" motion that's actually
# intentional, which shows up as growing stretch/smear artifacts near the
# frame edges. Set this to False for panning/flying footage; keep it True
# for footage from a roughly stationary hovering camera.
ENABLE_MOTION_COMPENSATION = True
MAX_CORNERS = 300              # goodFeaturesToTrack: how many background points to track
MIN_CORNERS_REQUIRED = 30      # below this, skip compensation for that frame (unreliable)
RANSAC_REPROJ_THRESHOLD = 3.0  # pixels; used by estimateAffinePartial2D
# How often (in frames) to re-anchor the cumulative transform to the CURRENT
# frame instead of the very first frame. Without this, small per-frame
# estimation errors compound over hundreds of frames into severe warping/
# streaking artifacts. Set to a smaller number for shakier footage.
REANCHOR_INTERVAL = 20
# Reject an estimated affine transform outright if it implies more than this
# much rotation (degrees) or scale change between two consecutive frames --
# a real camera cannot jump this much in 1/30s, so bigger values mean the
# estimation degenerated (e.g. not enough trackable texture in view).
MAX_PLAUSIBLE_ROTATION_DEG = 15.0
MAX_PLAUSIBLE_SCALE_DELTA = 0.15

# --- Tracker (frame-to-frame continuity, IOU-based) ---
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MAX_MISSED_FRAMES = 10  # how many frames a track can go undetected before dropping

# --- Runtime ---
CAMERA_INDEX = 0                # 0 = default webcam
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

# Some laptop webcams report frames pre-rotated (hardware/driver quirk).
# If your live window looks sideways or upside down, set this to one of:
# None, "90_CW", "90_CCW", "180"
ROTATE_FRAME = None

# For --video playback: how many frames to SKIP (display raw, no analysis)
# between each frame that actually gets run through the full pipeline
# (detection, Re-ID, pose, etc). Real-time playback isn't possible if the
# per-frame analysis takes longer than the video's own frame interval --
# this trades temporal resolution for wall-clock speed instead of just
# playing slower than the source video. 0 = analyze every frame (most
# accurate, but can feel like slow motion on CPU, especially with several
# targets). Raise this (2-4) for noticeably smoother playback; the tracked
# boxes update less often but the video itself plays back near real-time.
VIDEO_PROCESS_EVERY_N_FRAMES = 2