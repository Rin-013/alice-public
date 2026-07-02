"""
Titan Two Stub
==============

Serial-based communication with Titan Two device for hardware controller injection.
Requires Titan Two hardware + USB serial connection.
"""

from __future__ import annotations

from streaming.gaming.input.base import InputController
from streaming.gaming.types import GamepadAction
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TitanTwoController(InputController):
    """
    Stub for Titan Two serial controller.

    Requires Titan Two hardware connected via USB serial.
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        self._port = port
        self._baudrate = baudrate
        self._connected = False
        logger.info(f"TitanTwoController stub initialized (port={port})")

    def connect(self) -> bool:
        logger.error(f"TitanTwoController.connect() — stub, requires Titan Two on {self._port}")
        return False

    def disconnect(self):
        self._connected = False

    def send(self, action: GamepadAction):
        pass  # Stub

    def is_connected(self) -> bool:
        return self._connected
