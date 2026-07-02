"""
Frame Buffer
============

Triple-buffer for decoupling capture rate from consumer rate.
Writer (capture) writes to back buffer, reader (vision) reads from front.
"""

from __future__ import annotations

import threading
from typing import Optional

from streaming.gaming.types import CapturedFrame
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FrameBuffer:
    """
    Triple-buffer for frames: back → middle → front.

    - Capture thread writes to back buffer
    - Swap promotes back → middle → front
    - Consumer reads from front buffer (always latest complete frame)
    - Lock-free read on front buffer (only swap needs lock)
    """

    def __init__(self):
        self._back: Optional[CapturedFrame] = None
        self._middle: Optional[CapturedFrame] = None
        self._front: Optional[CapturedFrame] = None
        self._lock = threading.Lock()
        self._write_count = 0
        self._read_count = 0

    def write(self, frame: CapturedFrame):
        """Write a new frame (capture thread)."""
        with self._lock:
            self._back = frame
            # Auto-swap: back → middle → front
            self._middle, self._back = self._back, self._middle
            self._front, self._middle = self._middle, self._front
            self._write_count += 1

    def read(self) -> Optional[CapturedFrame]:
        """Read the latest frame (consumer thread). Returns None if no frame yet."""
        # Front buffer is safe to read without lock after swap
        self._read_count += 1
        return self._front

    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def read_count(self) -> int:
        return self._read_count

    @property
    def has_frame(self) -> bool:
        return self._front is not None
