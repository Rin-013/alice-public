"""Tests for gaming.mixer — directive mixer blending math."""

import numpy as np
import pytest

from streaming.gaming.cognitive.directive import DirectiveBuffer, DirectiveFactory
from streaming.gaming.mixer.directive_mixer import DirectiveMixer
from streaming.gaming.types import BehaviorMode, Directive, DirectivePriority, GamepadAction


class TestDirectiveMixer:
    def _make_mixer(self, **kwargs) -> DirectiveMixer:
        return DirectiveMixer(
            base_weight=kwargs.get("base_weight", 0.3),
            urgency_scale=kwargs.get("urgency_scale", 0.7),
            max_weight=kwargs.get("max_weight", 0.95),
            smoothing=kwargs.get("smoothing", 1.0),  # Instant for testing
            passthrough_on_empty=kwargs.get("passthrough_on_empty", True),
        )

    def test_passthrough_when_no_directive(self):
        """No directive → NitroGen passes through unchanged."""
        mixer = self._make_mixer()
        nitrogen = GamepadAction(
            buttons=np.array([1.0] + [0.0] * 19, dtype=np.float32),
            source="nitrogen",
        )
        result = mixer.mix(nitrogen)
        assert result.source == "passthrough"
        np.testing.assert_allclose(result.buttons, nitrogen.buttons, atol=0.01)

    def test_full_urgency_overrides(self):
        """Urgency 1.0 should heavily bias toward directive."""
        buf = DirectiveBuffer()
        mixer = self._make_mixer(base_weight=0.3, urgency_scale=0.7, smoothing=1.0)
        mixer._buffer = buf

        # Directive: full right stick
        d = DirectiveFactory.custom(
            buttons={},
            urgency=1.0,
            priority=DirectivePriority.CRITICAL,
            duration=10.0,
        )
        d.action.buttons[:] = 0.0
        d.action.buttons[0] = 1.0  # A button
        buf.push(d)

        # NitroGen: neutral
        nitrogen = GamepadAction.neutral()
        result = mixer.mix(nitrogen)

        # blend = 0.3 + 1.0 * 0.7 = 1.0, clamped to 0.95
        # result[0] ≈ (1-0.95)*0 + 0.95*1.0 = 0.95
        assert result.buttons[0] >= 0.9
        assert result.source == "mixed"

    def test_zero_urgency_low_blend(self):
        """Urgency 0.0 should use only base_weight."""
        buf = DirectiveBuffer()
        mixer = self._make_mixer(base_weight=0.3, urgency_scale=0.7, smoothing=1.0)
        mixer._buffer = buf

        d = DirectiveFactory.custom(
            buttons={},
            urgency=0.0,
            duration=10.0,
        )
        d.action.buttons[:] = 1.0
        buf.push(d)

        nitrogen = GamepadAction.neutral()
        result = mixer.mix(nitrogen)

        # blend = 0.3 + 0.0 * 0.7 = 0.3
        # result ≈ 0.7 * 0 + 0.3 * 1.0 = 0.3
        np.testing.assert_allclose(result.buttons, 0.3, atol=0.05)

    def test_mode_bias_affects_blend(self):
        """CLUTCH mode bias (0.25) should increase blend."""
        buf = DirectiveBuffer()
        mixer = self._make_mixer(base_weight=0.3, urgency_scale=0.7, smoothing=1.0)
        mixer._buffer = buf

        d = DirectiveFactory.custom(buttons={}, urgency=0.5, duration=10.0)
        d.action.buttons[:] = 1.0
        buf.push(d)

        nitrogen = GamepadAction.neutral()

        # ENTERTAINER mode (bias=0.0)
        result_ent = mixer.mix(nitrogen, mode=BehaviorMode.ENTERTAINER)
        blend_ent = mixer.blend_weight

        # Reset buffer
        buf.push(DirectiveFactory.custom(buttons={}, urgency=0.5, duration=10.0))
        buf._buffer[-1].action.buttons[:] = 1.0

        # CLUTCH mode (bias=0.25)
        result_clutch = mixer.mix(nitrogen, mode=BehaviorMode.CLUTCH)
        blend_clutch = mixer.blend_weight

        assert blend_clutch > blend_ent

    def test_blend_never_exceeds_max(self):
        """Even with max urgency + max mode bias, blend <= max_weight."""
        buf = DirectiveBuffer()
        mixer = self._make_mixer(
            base_weight=0.5, urgency_scale=1.0, max_weight=0.95, smoothing=1.0
        )
        mixer._buffer = buf

        d = DirectiveFactory.custom(buttons={}, urgency=1.0, duration=10.0)
        buf.push(d)

        nitrogen = GamepadAction.neutral()
        mixer.mix(nitrogen, mode=BehaviorMode.CLUTCH)

        assert mixer.blend_weight <= 0.96  # Small tolerance for float
