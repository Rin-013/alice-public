"""
Directive Mixer
===============

Blends NitroGen reflex actions with Alice's strategic directives.

Formula:
    blend = base_weight + urgency * urgency_scale
    blend = clamp(blend + mode_bias, 0, max_weight)
    final = (1 - blend) * nitrogen_action + blend * directive_action

When no directive is active, NitroGen output passes through unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from streaming.gaming.cognitive.directive import DirectiveBuffer
from streaming.gaming.types import BehaviorMode, Directive, GamepadAction
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default mode biases (additive adjustment to blend weight)
DEFAULT_MODE_BIAS: Dict[BehaviorMode, float] = {
    BehaviorMode.TRYHARD: 0.15,
    BehaviorMode.ENTERTAINER: 0.0,
    BehaviorMode.CHILL: -0.1,
    BehaviorMode.RAGE: 0.1,
    BehaviorMode.CURIOUS: -0.05,
    BehaviorMode.CLUTCH: 0.25,
}


class DirectiveMixer:
    """
    Blends NitroGen reflex output with Alice directive output.

    Each frame:
    1. Peek at highest-priority directive from buffer
    2. Calculate blend weight from base + urgency + mode bias
    3. Linearly interpolate between nitrogen and directive actions
    4. Apply exponential smoothing for smooth transitions
    """

    def __init__(
        self,
        directive_buffer: Optional[DirectiveBuffer] = None,
        base_weight: float = 0.3,
        urgency_scale: float = 0.7,
        max_weight: float = 0.95,
        smoothing: float = 0.15,
        passthrough_on_empty: bool = True,
        mode_bias: Optional[Dict[BehaviorMode, float]] = None,
    ):
        self._buffer = directive_buffer or DirectiveBuffer()
        self._base_weight = base_weight
        self._urgency_scale = urgency_scale
        self._max_weight = max_weight
        self._smoothing = smoothing
        self._passthrough_on_empty = passthrough_on_empty
        self._mode_bias = mode_bias or DEFAULT_MODE_BIAS

        # Smoothed blend weight (for gradual transitions)
        self._current_blend = 0.0
        self._current_mode = BehaviorMode.ENTERTAINER
        self._last_directive_desc = ""

    # --- Main mix ---

    def mix(
        self,
        nitrogen_action: GamepadAction,
        mode: BehaviorMode = BehaviorMode.ENTERTAINER,
    ) -> GamepadAction:
        """
        Blend NitroGen action with the top directive.

        Args:
            nitrogen_action: Raw NitroGen output for this frame.
            mode: Current behavior mode.

        Returns:
            Blended GamepadAction.
        """
        self._current_mode = mode
        directive = self._buffer.peek()

        # No directive → passthrough
        if directive is None:
            if self._passthrough_on_empty:
                self._smooth_blend(0.0)
                return GamepadAction(
                    buttons=nitrogen_action.buttons.copy(),
                    timestamp=nitrogen_action.timestamp,
                    source="passthrough",
                )

        if directive is None:
            target_blend = 0.0
            directive_buttons = np.zeros(20, dtype=np.float32)
        else:
            # Calculate target blend
            target_blend = self._calculate_blend(directive, mode)
            directive_buttons = directive.action.buttons
            self._last_directive_desc = directive.description

        # Smooth the blend weight
        self._smooth_blend(target_blend)

        # Interpolate
        blended = (
            (1.0 - self._current_blend) * nitrogen_action.buttons
            + self._current_blend * directive_buttons
        )

        return GamepadAction(
            buttons=blended.astype(np.float32),
            timestamp=nitrogen_action.timestamp,
            source="mixed",
        )

    def _calculate_blend(self, directive: Directive, mode: BehaviorMode) -> float:
        """Calculate target blend weight."""
        blend = self._base_weight + directive.urgency * self._urgency_scale
        blend += self._mode_bias.get(mode, 0.0)
        return float(np.clip(blend, 0.0, self._max_weight))

    def _smooth_blend(self, target: float):
        """Exponential smoothing on blend weight."""
        self._current_blend += self._smoothing * (target - self._current_blend)

    # --- Properties ---

    @property
    def blend_weight(self) -> float:
        return self._current_blend

    @property
    def buffer(self) -> DirectiveBuffer:
        return self._buffer

    def get_status(self) -> Dict:
        return {
            "blend_weight": round(self._current_blend, 3),
            "mode": self._current_mode.value,
            "base_weight": self._base_weight,
            "urgency_scale": self._urgency_scale,
            "max_weight": self._max_weight,
            "last_directive": self._last_directive_desc,
            "buffer": self._buffer.get_status(),
        }
