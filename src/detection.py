"""
Person detection wrapper around Ultralytics YOLO.
Kept intentionally thin -- swapping YOLO versions/weights later shouldn't
touch any other module.
"""

from dataclasses import dataclass

import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)
from src.device import get_inference_device

from config import (
    PERSON_CLASS_NAMES,
    TILE_BATCH_SIZE,
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
        self.device, self.half = get_inference_device()
        # Map configured person-like class NAMES to this model's own class
        # IDs. Different YOLO weights use different label sets (stock
        # COCO weights: just "person"; VisDrone-style weights: separate
        # "pedestrian"/"people"/"person" classes) -- resolving by name
        # instead of a hardcoded ID keeps this working across models.
        name_to_id = {name.lower(): idx for idx, name in self.model.names.items()}
        self._person_class_ids = [
            name_to_id[n.lower()] for n in PERSON_CLASS_NAMES if n.lower() in name_to_id
        ]
        if not self._person_class_ids:
            raise ValueError(
                f"None of PERSON_CLASS_NAMES={PERSON_CLASS_NAMES} match this "
                f"model's classes: {list(self.model.names.values())}"
            )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Standard single-pass detection on the whole frame."""
        results = self.model.predict(
            frame_bgr,
            classes=self._person_class_ids,
            conf=YOLO_CONF_THRESHOLD,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        if len(results.boxes) == 0:
            return []
        boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        return [
            Detection(box=tuple(int(v) for v in box), confidence=float(conf))
            for box, conf in zip(boxes_xyxy, confs)
        ]

    def detect_tiled(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Slices the frame into overlapping TILE_SIZE_PX tiles, runs detection
        on all tiles, maps boxes back to full-frame coordinates, and merges
        overlapping duplicates (common near tile borders, where the same
        person can get detected in two neighboring tiles) with NMS.

        Tiles are sent to the model in BATCHES (TILE_BATCH_SIZE per call)
        instead of one call per tile. On CUDA, the tiles are also cut
        directly in GPU memory: the frame is uploaded once, converted
        (BGR->RGB, CHW, fp16) on the GPU, sliced into a (N, 3, T, T) tensor
        and fed to the model as-is -- no per-tile CPU crop/letterbox/copy.
        Profiling showed that per-tile CPU work + per-call overhead, not the
        network itself, was most of the tiling cost. Without CUDA (CPU,
        or a machine without an NVIDIA GPU) the same batching is done from
        numpy tiles, so results are identical everywhere -- only speed
        differs.
        """
        h, w = frame_bgr.shape[:2]
        tile = TILE_SIZE_PX
        stride = max(1, int(tile * (1 - TILE_OVERLAP_RATIO)))
        origins = [(x, y) for y in _tile_starts(h, tile, stride)
                   for x in _tile_starts(w, tile, stride)]

        if self.device.startswith("cuda"):
            per_tile = self._predict_tiles_gpu(frame_bgr, origins)
        else:
            per_tile = self._predict_tiles_cpu(frame_bgr, origins)

        all_boxes_xywh = []
        all_confs = []
        for (x, y), result in zip(origins, per_tile):
            if len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), conf in zip(boxes, confs):
                # Boxes in padded regions (frame smaller than a tile) are
                # clipped back to the real frame.
                gx1, gy1 = min(x + bx1, w), min(y + by1, h)
                gx2, gy2 = min(x + bx2, w), min(y + by2, h)
                if gx2 - gx1 < 1 or gy2 - gy1 < 1:
                    continue
                all_boxes_xywh.append([int(gx1), int(gy1), int(gx2 - gx1), int(gy2 - gy1)])
                all_confs.append(float(conf))

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

    def _predict_batch(self, source):
        return self.model.predict(
            source,
            classes=self._person_class_ids,
            conf=YOLO_CONF_THRESHOLD,
            device=self.device,
            half=self.half,
            verbose=False,
        )

    def _predict_tiles_gpu(self, frame_bgr: np.ndarray, origins: list) -> list:
        tile = TILE_SIZE_PX
        h, w = frame_bgr.shape[:2]
        frame = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(self.device, non_blocking=True)
        frame = frame.permute(2, 0, 1).flip(0)            # HWC BGR -> CHW RGB
        frame = (frame.half() if self.half else frame.float()) / 255.0
        # Frames smaller than one tile: pad (YOLO's letterbox gray) so every
        # tile is exactly tile x tile -- a batch tensor needs equal shapes.
        pad_h, pad_w = max(0, tile - h), max(0, tile - w)
        if pad_h or pad_w:
            frame = torch.nn.functional.pad(frame, (0, pad_w, 0, pad_h), value=114 / 255.0)

        results = []
        with torch.inference_mode():
            for i in range(0, len(origins), TILE_BATCH_SIZE):
                chunk = origins[i:i + TILE_BATCH_SIZE]
                batch = torch.stack([frame[:, y:y + tile, x:x + tile] for x, y in chunk])
                results.extend(self._predict_batch(batch))
        return results

    def _predict_tiles_cpu(self, frame_bgr: np.ndarray, origins: list) -> list:
        tile = TILE_SIZE_PX
        results = []
        for i in range(0, len(origins), TILE_BATCH_SIZE):
            chunk = origins[i:i + TILE_BATCH_SIZE]
            crops = [frame_bgr[y:y + tile, x:x + tile] for x, y in chunk]
            results.extend(self._predict_batch(crops))
        return results
