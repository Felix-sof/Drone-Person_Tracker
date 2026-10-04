"""
Startup sanity checks for config.py values.

Catches config mistakes that would otherwise fail silently -- e.g. an
out-of-range FOV producing a "plausible but wrong" distance estimate
instead of an obvious crash. Call validate_config() once at app startup,
before the pipeline starts processing frames.
"""

import os
import config


class ConfigError(Exception):
    """Raised when config.py contains an invalid or inconsistent value."""
    pass


def validate_config():
    errors = []

    # --- Distance estimation ---
    if not (0 < config.CAMERA_VERTICAL_FOV_DEG < 180):
        errors.append(
            f"CAMERA_VERTICAL_FOV_DEG={config.CAMERA_VERTICAL_FOV_DEG} is out of "
            f"the physically valid range (0, 180). Check your camera's spec sheet."
        )
    if config.ASSUMED_PERSON_HEIGHT_M <= 0:
        errors.append(
            f"ASSUMED_PERSON_HEIGHT_M={config.ASSUMED_PERSON_HEIGHT_M} must be positive."
        )
    if config.DISTANCE_ESTIMATION_INTERVAL < 1:
        errors.append("DISTANCE_ESTIMATION_INTERVAL must be >= 1.")

    # --- Re-ID thresholds (must be valid cosine-similarity values) ---
    for name in ("REID_MATCH_THRESHOLD", "DUPLICATE_TARGET_SIMILARITY_THRESHOLD",
                 "AUTO_GALLERY_NOVELTY_MAX_SIMILARITY"):
        value = getattr(config, name)
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name}={value} must be between 0.0 and 1.0.")

    if config.REID_BACKEND not in ("osnet", "resnet18"):
        errors.append(
            f"REID_BACKEND={config.REID_BACKEND!r} is not recognized "
            f"(expected 'osnet' or 'resnet18')."
        )

    # --- Tiling ---
    if config.ENABLE_TILED_DETECTION:
        if config.TILE_SIZE_PX <= 0:
            errors.append("TILE_SIZE_PX must be positive.")
        if not (0.0 <= config.TILE_OVERLAP_RATIO < 1.0):
            errors.append("TILE_OVERLAP_RATIO must be in [0.0, 1.0).")
        if config.TILE_BATCH_SIZE < 1:
            errors.append("TILE_BATCH_SIZE must be >= 1.")
        if config.TILE_SIZE_PX % 32 != 0:
            errors.append(f"TILE_SIZE_PX={config.TILE_SIZE_PX} must be a multiple of 32 "
                          f"(YOLO's stride; required for batched GPU tiles).")

    # --- Runtime ---
    if config.MAX_CONCURRENT_TARGETS < 1:
        errors.append("MAX_CONCURRENT_TARGETS must be >= 1.")
    if config.MOTION_COMP_MODE not in ("tracks", "warp"):
        errors.append(f"MOTION_COMP_MODE={config.MOTION_COMP_MODE!r} must be 'tracks' or 'warp'.")
    if config.PROCESS_MAX_SIDE is not None and config.PROCESS_MAX_SIDE < 320:
        errors.append("PROCESS_MAX_SIDE must be None or >= 320.")
    if config.ROTATE_FRAME not in (None, "90_CW", "90_CCW", "180"):
        errors.append(
            f"ROTATE_FRAME={config.ROTATE_FRAME!r} is not recognized "
            f"(expected None, '90_CW', '90_CCW', or '180')."
        )
    if config.FRAME_WIDTH <= 0 or config.FRAME_HEIGHT <= 0:
        errors.append("FRAME_WIDTH and FRAME_HEIGHT must be positive.")

    # --- Target lifecycle / tracker ---
    if config.AUTO_TARGET_MIN_HITS < 1:
        errors.append("AUTO_TARGET_MIN_HITS must be >= 1.")
    if config.LOST_TARGET_EXPIRY_FRAMES < 1:
        errors.append("LOST_TARGET_EXPIRY_FRAMES must be >= 1.")
    if config.TRACKER_MAX_MISSED_FRAMES < 0:
        errors.append("TRACKER_MAX_MISSED_FRAMES must be >= 0.")
    if not (0.0 < config.TRACKER_IOU_THRESHOLD <= 1.0):
        errors.append(f"TRACKER_IOU_THRESHOLD={config.TRACKER_IOU_THRESHOLD} must be in (0.0, 1.0].")
    if config.REID_CONTINUATION_GATE_HEIGHTS <= 0:
        errors.append("REID_CONTINUATION_GATE_HEIGHTS must be positive.")
    if config.REID_GALLERY_MAX_SIZE < 1:
        errors.append("REID_GALLERY_MAX_SIZE must be >= 1.")
    for name in ("REID_RESCAN_INTERVAL", "REID_LOST_RESCAN_INTERVAL", "POSE_ANALYSIS_INTERVAL", "EMOTION_ANALYSIS_INTERVAL",
                 "AUTO_GALLERY_UPDATE_INTERVAL", "DB_LOG_INTERVAL"):
        if getattr(config, name) < 1:
            errors.append(f"{name} must be >= 1 (it's used as a modulo interval).")

    # --- Display / logging ---
    if config.TRAIL_LENGTH < 0:
        errors.append("TRAIL_LENGTH must be >= 0.")
    if config.HUD_MAX_PANELS < 1:
        errors.append("HUD_MAX_PANELS must be >= 1.")
    if config.DB_RECONNECT_BACKOFF_S < 0:
        errors.append("DB_RECONNECT_BACKOFF_S must be >= 0.")
    if config.VIDEO_PROCESS_EVERY_N_FRAMES < 0:
        errors.append("VIDEO_PROCESS_EVERY_N_FRAMES must be >= 0.")

    # --- Model files (warn, don't hard-fail -- ultralytics can auto-download) ---
    if not os.path.isabs(config.YOLO_MODEL) and not config.YOLO_MODEL.endswith(".pt"):
        errors.append(f"YOLO_MODEL={config.YOLO_MODEL!r} doesn't look like a .pt weights file.")

    if errors:
        message = "Invalid config.py settings:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(message)