"""Tests for gaming.types data structures."""

import time

import numpy as np
import pytest

from streaming.gaming.types import (
    BehaviorMode,
    ButtonIndex,
    CapturedFrame,
    CommentaryPriority,
    CommentaryRequest,
    Directive,
    DirectivePriority,
    GamepadAction,
    GameState,
)


# ---------------------------------------------------------------------------
# GamepadAction
# ---------------------------------------------------------------------------

class TestGamepadAction:
    def test_neutral(self):
        action = GamepadAction.neutral()
        assert action.buttons.shape == (20,)
        assert np.all(action.buttons == 0.0)
        assert action.source == "neutral"

    def test_default_shape(self):
        action = GamepadAction()
        assert action.buttons.shape == (20,)
        assert action.source == "mock"

    def test_set_get(self):
        action = GamepadAction.neutral()
        action.set(ButtonIndex.A, 1.0)
        assert action.get(ButtonIndex.A) == 1.0
        assert action.get(ButtonIndex.B) == 0.0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            GamepadAction(buttons=np.zeros(10))

    def test_list_conversion(self):
        action = GamepadAction(buttons=[0.0] * 20)
        assert action.buttons.shape == (20,)
        assert action.buttons.dtype == np.float32


# ---------------------------------------------------------------------------
# Directive
# ---------------------------------------------------------------------------

class TestDirective:
    def test_not_expired_initially(self):
        d = Directive(action=GamepadAction.neutral(), duration_sec=10.0)
        assert not d.is_expired()
        assert d.remaining_sec() > 9.0

    def test_expired_after_duration(self):
        d = Directive(
            action=GamepadAction.neutral(),
            duration_sec=0.0,
            timestamp=time.time() - 1.0,
        )
        assert d.is_expired()

    def test_manual_expire(self):
        d = Directive(action=GamepadAction.neutral())
        d.expired = True
        assert d.is_expired()

    def test_priority_ordering(self):
        assert DirectivePriority.LOW < DirectivePriority.NORMAL
        assert DirectivePriority.NORMAL < DirectivePriority.HIGH
        assert DirectivePriority.HIGH < DirectivePriority.CRITICAL


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

class TestGameState:
    def test_defaults(self):
        gs = GameState()
        assert gs.health == 1.0
        assert gs.is_safe
        assert not gs.is_critical

    def test_critical(self):
        gs = GameState(health=0.1, in_combat=True)
        assert gs.is_critical

    def test_not_critical_if_safe(self):
        gs = GameState(health=0.1, in_combat=False)
        assert not gs.is_critical

    def test_is_safe(self):
        gs = GameState(in_combat=False, is_dead=False, in_cutscene=False)
        assert gs.is_safe

    def test_not_safe_in_combat(self):
        gs = GameState(in_combat=True)
        assert not gs.is_safe


# ---------------------------------------------------------------------------
# CommentaryRequest
# ---------------------------------------------------------------------------

class TestCommentaryRequest:
    def test_age(self):
        req = CommentaryRequest(text="test", timestamp=time.time() - 5.0)
        assert req.age_sec >= 4.9

    def test_priority_ordering(self):
        assert CommentaryPriority.FILLER < CommentaryPriority.CRITICAL


# ---------------------------------------------------------------------------
# CapturedFrame
# ---------------------------------------------------------------------------

class TestCapturedFrame:
    def test_auto_dimensions(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = CapturedFrame(image=img)
        assert frame.width == 640
        assert frame.height == 480


# ---------------------------------------------------------------------------
# BehaviorMode
# ---------------------------------------------------------------------------

class TestBehaviorMode:
    def test_all_modes(self):
        modes = list(BehaviorMode)
        assert len(modes) == 6
        assert BehaviorMode.TRYHARD in modes
        assert BehaviorMode.ENTERTAINER in modes
