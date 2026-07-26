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

    # --- Runtime ---
    if config.MAX_CONCURRENT_TARGETS < 1:
        errors.append("MAX_CONCURRENT_TARGETS must be >= 1.")
    if config.ROTATE_FRAME not in (None, "90_CW", "90_CCW", "180"):
        errors.append(
            f"ROTATE_FRAME={config.ROTATE_FRAME!r} is not recognized "
            f"(expected None, '90_CW', '90_CCW', or '180')."
        )
    if config.FRAME_WIDTH <= 0 or config.FRAME_HEIGHT <= 0:
        errors.append("FRAME_WIDTH and FRAME_HEIGHT must be positive.")

    # --- Model files (warn, don't hard-fail -- ultralytics can auto-download) ---
    if not os.path.isabs(config.YOLO_MODEL) and not config.YOLO_MODEL.endswith(".pt"):
        errors.append(f"YOLO_MODEL={config.YOLO_MODEL!r} doesn't look like a .pt weights file.")

    if errors:
        message = "Invalid config.py settings:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(message)