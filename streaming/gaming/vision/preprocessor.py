"""
Frame Preprocessor
==================

Resizes and normalizes game frames for NitroGen input.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from streaming.gaming.types import CapturedFrame
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def preprocess_frame(
    frame: CapturedFrame,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess a captured frame for NitroGen input.

    Args:
        frame: Raw captured frame.
        target_size: (width, height) for resize.
        normalize: If True, scale to [0, 1] float32.

    Returns:
        Preprocessed numpy array (H, W, C) or (C, H, W) depending on model needs.
    """
    img = frame.image

    if img is None:
        return np.zeros((*target_size, 3), dtype=np.float32)

    # Resize
    if CV2_AVAILABLE:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        # Basic nearest-neighbor resize without cv2
        h, w = img.shape[:2]
        th, tw = target_size[1], target_size[0]
        row_idx = (np.arange(th) * h // th).astype(int)
        col_idx = (np.arange(tw) * w // tw).astype(int)
        img = img[np.ix_(row_idx, col_idx)]

    # Normalize
    if normalize:
        img = img.astype(np.float32) / 255.0

    return img
