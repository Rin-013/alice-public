"""
Xbox Capture Stub
=================

Magewell USB capture card based frame acquisition.
Requires Magewell SDK + hardware.
"""

from __future__ import annotations

from typing import Optional

from streaming.gaming.capture.base import CaptureSource
from streaming.gaming.types import CapturedFrame
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class XboxCapture(CaptureSource):
    """
    Stub for Magewell USB capture card.

    Requires Magewell hardware + SDK.
    """

    def __init__(self, device_index: int = 0, width: int = 1920, height: int = 1080):
        self._device_index = device_index
        self._width = width
        self._height = height
        self._active = False
        logger.info(f"XboxCapture stub initialized (device={device_index})")

    def start(self) -> bool:
        logger.error("XboxCapture.start() — stub, requires Magewell hardware")
        return False

    def stop(self):
        self._active = False

    def grab(self) -> Optional[CapturedFrame]:
        return None

    def is_active(self) -> bool:
        return self._active
