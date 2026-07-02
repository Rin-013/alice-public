"""
OBS Overlay Stub
================

Placeholder for OBS WebSocket integration (stream overlays,
scene switching, etc.). Fully stubbed — needs OBS running.
"""

from __future__ import annotations

from typing import Dict, Optional

from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OBSOverlay:
    """
    Stub for OBS WebSocket overlay control.

    Future: connect via obs-websocket-py to control scenes,
    show game state HUD, display commentary text, etc.
    """

    def __init__(self, host: str = "localhost", port: int = 4455, password: str = ""):
        self._host = host
        self._port = port
        self._connected = False
        logger.info(f"OBSOverlay stub initialized (host={host}:{port})")

    def connect(self) -> bool:
        """Stub — would connect to OBS WebSocket."""
        logger.info("OBSOverlay.connect() — stub, no-op")
        return False

    def disconnect(self):
        """Stub — would disconnect from OBS."""
        self._connected = False

    def set_text(self, source_name: str, text: str):
        """Stub — would set text on an OBS text source."""
        logger.debug(f"OBSOverlay.set_text({source_name}, {text[:50]}...) — stub")

    def switch_scene(self, scene_name: str):
        """Stub — would switch OBS scene."""
        logger.debug(f"OBSOverlay.switch_scene({scene_name}) — stub")

    def show_source(self, source_name: str, visible: bool = True):
        """Stub — would show/hide a source."""
        logger.debug(f"OBSOverlay.show_source({source_name}, {visible}) — stub")

    def get_status(self) -> Dict:
        return {
            "connected": self._connected,
            "host": self._host,
            "port": self._port,
            "stub": True,
        }
