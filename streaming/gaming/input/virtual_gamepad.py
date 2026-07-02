"""
Virtual Gamepad Stub
====================

vgamepad-based virtual Xbox 360 controller for Windows.
Requires Windows + vgamepad package.
"""

from __future__ import annotations

from streaming.gaming.input.base import InputController
from streaming.gaming.types import ButtonIndex, GamepadAction
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import vgamepad as vg
    VGAMEPAD_AVAILABLE = True
except ImportError:
    VGAMEPAD_AVAILABLE = False


class VirtualGamepad(InputController):
    """
    Stub for vgamepad-based virtual Xbox 360 controller.

    Requires Windows and vgamepad package.
    """

    def __init__(self):
        self._gamepad = None
        self._connected = False

        if not VGAMEPAD_AVAILABLE:
            logger.warning("vgamepad not available — VirtualGamepad is a stub")

    def connect(self) -> bool:
        if not VGAMEPAD_AVAILABLE:
            logger.error("Cannot connect VirtualGamepad: vgamepad not installed (Windows only)")
            return False

        try:
            self._gamepad = vg.VX360Gamepad()
            self._connected = True
            logger.info("VirtualGamepad connected (VX360)")
            return True
        except Exception as e:
            logger.error(f"VirtualGamepad connect failed: {e}")
            return False

    def disconnect(self):
        self._gamepad = None
        self._connected = False

    def send(self, action: GamepadAction):
        if not self._connected or self._gamepad is None:
            return

        gp = self._gamepad
        try:
            # Sticks
            gp.left_joystick_float(
                x_value_float=action.get(ButtonIndex.LSTICK_X),
                y_value_float=action.get(ButtonIndex.LSTICK_Y),
            )
            gp.right_joystick_float(
                x_value_float=action.get(ButtonIndex.RSTICK_X),
                y_value_float=action.get(ButtonIndex.RSTICK_Y),
            )

            # Triggers
            gp.left_trigger_float(action.get(ButtonIndex.LT))
            gp.right_trigger_float(action.get(ButtonIndex.RT))

            gp.update()
        except Exception as e:
            logger.error(f"VirtualGamepad send error: {e}")

    def is_connected(self) -> bool:
        return self._connected
