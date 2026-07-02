"""
Action Serializer
=================

Converts between NitroGen tensor output and GamepadAction.
"""

from __future__ import annotations

import numpy as np

from streaming.gaming.types import GamepadAction


def tensor_to_action(tensor: np.ndarray, source: str = "nitrogen") -> GamepadAction:
    """
    Convert a raw NitroGen output tensor to GamepadAction.

    Args:
        tensor: 1D float array, length 20 (matching ButtonIndex order).
        source: Source tag.

    Returns:
        GamepadAction with clamped values.
    """
    if tensor.ndim != 1:
        tensor = tensor.flatten()

    # Pad or truncate to 20
    if tensor.shape[0] < 20:
        padded = np.zeros(20, dtype=np.float32)
        padded[: tensor.shape[0]] = tensor
        tensor = padded
    elif tensor.shape[0] > 20:
        tensor = tensor[:20]

    # Clamp digital buttons (0-7, 12-19) to [0, 1], analog (8-11) to [-1, 1]
    buttons = tensor.astype(np.float32)
    buttons[:8] = np.clip(buttons[:8], 0.0, 1.0)
    buttons[8:12] = np.clip(buttons[8:12], -1.0, 1.0)
    buttons[12:] = np.clip(buttons[12:], 0.0, 1.0)

    return GamepadAction(buttons=buttons, source=source)


def action_to_tensor(action: GamepadAction) -> np.ndarray:
    """Convert GamepadAction back to a flat float32 tensor."""
    return action.buttons.copy()
