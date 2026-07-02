"""
Input Base
==========

Abstract InputController interface and MockController for dev/testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from streaming.gaming.types import GamepadAction
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InputController(ABC):
    """Abstract base for all input controllers."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the controller device. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self):
        """Disconnect from the controller device."""
        ...

    @abstractmethod
    def send(self, action: GamepadAction):
        """Send a gamepad action to the device."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        ...


class MockController(InputController):
    """
    Mock controller that logs actions instead of sending them.

    Useful for testing the full pipeline without real hardware.
    """

    def __init__(self, verbose: bool = False):
        self._connected = False
        self._verbose = verbose
        self._action_count = 0
        self._last_action: GamepadAction | None = None

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockController connected")
        return True

    def disconnect(self):
        self._connected = False
        logger.info("MockController disconnected")

    def send(self, action: GamepadAction):
        if not self._connected:
            return
        self._action_count += 1
        self._last_action = action
        if self._verbose:
            nonzero = [(i, float(v)) for i, v in enumerate(action.buttons) if abs(v) > 0.05]
            if nonzero:
                logger.debug(f"MockController action #{self._action_count}: {nonzero}")

    def is_connected(self) -> bool:
        return self._connected

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def last_action(self) -> GamepadAction | None:
        return self._last_action

    def get_status(self) -> Dict:
        return {
            "connected": self._connected,
            "action_count": self._action_count,
            "stub": True,
        }
