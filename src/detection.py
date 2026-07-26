"""
Person detection wrapper around Ultralytics YOLO.
Kept intentionally thin -- swapping YOLO versions/weights later shouldn't
touch any other module.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    PERSON_CLASS_ID,
    TILE_OVERLAP_RATIO,
    TILE_SIZE_PX,
    TILING_NMS_IOU_THRESHOLD,
    YOLO_CONF_THRESHOLD,
    YOLO_MODEL,
)


@dataclass
class Detection:
    box: tuple      # (x1, y1, x2, y2) in pixel coords
    confidence: float


def _tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """
    Start coordinates (along one axis) for tiles covering [0, total),
    each `tile` pixels wide/tall, stepping by `stride`, with a final tile
    snapped to the edge so the whole dimension gets covered even when it
    doesn't divide evenly by the stride.
    """
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    last_covered_end = starts[-1] + tile
    if last_covered_end < total:
        starts.append(total - tile)
    return starts


class PersonDetector:
    def __init__(self, model_path: str = YOLO_MODEL):
        self.model = YOLO(model_path)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Standard single-pass detection on the whole frame."""
        results = self.model.predict(
            frame_bgr,
            classes=[PERSON_CLASS_ID],
            conf=YOLO_CONF_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            detections.append(Detection(box=(x1, y1, x2, y2), confidence=conf))
        return detections

    def detect_tiled(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Slices the frame into overlapping TILE_SIZE_PX tiles, runs detection
        on each tile separately, maps boxes back to full-frame coordinates,
        and merges overlapping duplicates (common near tile borders, where
        the same person can get detected in two neighboring tiles) with NMS.

        Costs roughly (num_tiles) detector calls instead of 1 -- use this
        only when small/distant objects are actually being missed by
        `detect()`, not as a default.
        """
        h, w = frame_bgr.shape[:2]
        stride = max(1, int(TILE_SIZE_PX * (1 - TILE_OVERLAP_RATIO)))

        xs = _tile_starts(w, TILE_SIZE_PX, stride)
        ys = _tile_starts(h, TILE_SIZE_PX, stride)

        all_boxes_xywh = []
        all_confs = []

        for y in ys:
            for x in xs:
                tile = frame_bgr[y:y + TILE_SIZE_PX, x:x + TILE_SIZE_PX]
                if tile.size == 0:
                    continue
                for det in self.detect(tile):
                    bx1, by1, bx2, by2 = det.box
                    gx1, gy1 = x + bx1, y + by1
                    gw, gh = bx2 - bx1, by2 - by1
                    all_boxes_xywh.append([int(gx1), int(gy1), int(gw), int(gh)])
                    all_confs.append(det.confidence)

        if not all_boxes_xywh:
            return []

        keep_indices = cv2.dnn.NMSBoxes(
            all_boxes_xywh, all_confs, YOLO_CONF_THRESHOLD, TILING_NMS_IOU_THRESHOLD
        )
        keep_indices = np.array(keep_indices).flatten() if len(keep_indices) else []

        merged = []
        for i in keep_indices:
            gx, gy, gw, gh = all_boxes_xywh[i]
            merged.append(Detection(box=(gx, gy, gx + gw, gy + gh), confidence=all_confs[i]))
        return merged
