"""
src/device.py

Shared GPU/CPU device selection for the two YOLO models used in this
project (person detection + pose estimation). Centralized so both pick
the same device the same way, and so turning on GPU acceleration is a
one-line change in each caller instead of duplicated detection logic.
"""

import torch


def get_inference_device() -> tuple[str, bool]:
    """
    Returns (device, use_half):
      device   -- "cuda" if a CUDA GPU is available, else "cpu".
      use_half -- True only when running on CUDA. Half precision (fp16)
                  roughly doubles throughput on supported GPUs with
                  negligible impact on detection/pose confidence scores,
                  but it isn't supported (or faster) on CPU, so it's
                  always False there.

    On a machine with no GPU this returns ("cpu", False) -- identical to
    the previous hardcoded behavior, so nothing changes for CPU-only setups.
    """
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False
