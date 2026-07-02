"""
Capture Base
============

Abstract CaptureSource interface and MockCapture for dev/testing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from streaming.gaming.types import CapturedFrame
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CaptureSource(ABC):
    """Abstract base for all capture sources."""

    @abstractmethod
    def start(self) -> bool:
        """Start capturing. Returns True on success."""
        ...

    @abstractmethod
    def stop(self):
        """Stop capturing."""
        ...

    @abstractmethod
    def grab(self) -> Optional[CapturedFrame]:
        """Grab the latest frame. Returns None if unavailable."""
        ...

    @abstractmethod
    def is_active(self) -> bool:
        ...


class MockCapture(CaptureSource):
    """
    Mock capture source that generates random noise frames.

    Useful for testing the full pipeline without real game capture.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 60):
        self._width = width
        self._height = height
        self._fps = fps
        self._active = False
        self._frame_number = 0
        self._target_dt = 1.0 / fps
        self._last_frame_time = 0.0

    def start(self) -> bool:
        self._active = True
        self._frame_number = 0
        self._last_frame_time = time.time()
        logger.info(f"MockCapture started ({self._width}x{self._height} @ {self._fps}fps)")
        return True

    def stop(self):
        self._active = False
        logger.info("MockCapture stopped")

    def grab(self) -> Optional[CapturedFrame]:
        if not self._active:
            return None

        now = time.time()
        if (now - self._last_frame_time) < self._target_dt:
            return None  # Rate limit

        self._last_frame_time = now
        self._frame_number += 1

        # Generate a random noise frame (cheap, fast)
        image = np.random.randint(
            0, 255,
            (self._height, self._width, 3),
            dtype=np.uint8,
        )

        return CapturedFrame(
            image=image,
            timestamp=now,
            frame_number=self._frame_number,
            width=self._width,
            height=self._height,
            source="mock",
        )

    def is_active(self) -> bool:
        return self._active
