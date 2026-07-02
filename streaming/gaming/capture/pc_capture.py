"""
PC Capture Stub
===============

DXGI/DXcam-based screen capture for Windows.
Requires Windows + dxcam package.
"""

from __future__ import annotations

from typing import Optional

from streaming.gaming.capture.base import CaptureSource
from streaming.gaming.types import CapturedFrame
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False


class PCCapture(CaptureSource):
    """
    Stub for DXcam-based Windows screen capture.

    Requires Windows and dxcam package.
    """

    def __init__(self, target_fps: int = 60, region: Optional[tuple] = None):
        self._target_fps = target_fps
        self._region = region
        self._camera = None
        self._active = False

        if not DXCAM_AVAILABLE:
            logger.warning("dxcam not available — PCCapture is a stub")

    def start(self) -> bool:
        if not DXCAM_AVAILABLE:
            logger.error("Cannot start PCCapture: dxcam not installed (Windows only)")
            return False

        try:
            self._camera = dxcam.create(output_color="BGR")
            self._camera.start(target_fps=self._target_fps, region=self._region)
            self._active = True
            logger.info(f"PCCapture started at {self._target_fps} fps")
            return True
        except Exception as e:
            logger.error(f"PCCapture start failed: {e}")
            return False

    def stop(self):
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
        self._active = False

    def grab(self) -> Optional[CapturedFrame]:
        if not self._active or self._camera is None:
            return None

        frame = self._camera.get_latest_frame()
        if frame is None:
            return None

        return CapturedFrame(
            image=frame,
            source="pc_dxcam",
        )

    def is_active(self) -> bool:
        return self._active
