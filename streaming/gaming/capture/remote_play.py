"""
Remote Play Capture Stub
========================

Xbox Remote Play window capture via Windows API.
Requires Windows + Xbox Remote Play app running.
"""

from __future__ import annotations

from typing import Optional

from streaming.gaming.capture.base import CaptureSource
from streaming.gaming.types import CapturedFrame
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RemotePlayCapture(CaptureSource):
    """
    Stub for capturing the Xbox Remote Play window on Windows.

    Would use win32gui + DXcam region capture on the Remote Play window.
    """

    def __init__(self, window_title: str = "Xbox Remote Play"):
        self._window_title = window_title
        self._active = False
        logger.info(f"RemotePlayCapture stub initialized (window='{window_title}')")

    def start(self) -> bool:
        logger.error("RemotePlayCapture.start() — stub, requires Windows + Xbox Remote Play")
        return False

    def stop(self):
        self._active = False

    def grab(self) -> Optional[CapturedFrame]:
        return None

    def is_active(self) -> bool:
        return self._active
