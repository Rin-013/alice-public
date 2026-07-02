"""Tests for gaming.cognitive — behavior modes, emotion bridge, director."""

import time

import numpy as np
import pytest

from streaming.gaming.cognitive.behavior_modes import BehaviorModeManager
from streaming.gaming.cognitive.directive import DirectiveBuffer, DirectiveFactory
from streaming.gaming.cognitive.director import AliceDirector
from streaming.gaming.cognitive.emotion_bridge import (
    EMOTION_MODE_BIAS,
    EMOTION_TTS_INSTRUCT,
    EmotionBridge,
)
from streaming.gaming.types import BehaviorMode, Directive, DirectivePriority, GamepadAction, GameState


# ---------------------------------------------------------------------------
# BehaviorModeManager
# ---------------------------------------------------------------------------

class TestBehaviorModeManager:
    def test_initial_mode(self):
        mgr = BehaviorModeManager(initial_mode=BehaviorMode.CHILL)
        assert mgr.mode == BehaviorMode.CHILL

    def test_transition(self):
        mgr = BehaviorModeManager(transition_cooldown_sec=0.0)
        assert mgr.transition(BehaviorMode.TRYHARD)
        assert mgr.mode == BehaviorMode.TRYHARD
        assert mgr.previous_mode == BehaviorMode.ENTERTAINER

    def test_no_self_transition(self):
        mgr = BehaviorModeManager()
        assert not mgr.transition(BehaviorMode.ENTERTAINER)

    def test_cooldown_blocks(self):
        mgr = BehaviorModeManager(transition_cooldown_sec=100.0)
        mgr.transition(BehaviorMode.TRYHARD, force=True)
        assert not mgr.transition(BehaviorMode.CHILL)

    def test_force_ignores_cooldown(self):
        mgr = BehaviorModeManager(transition_cooldown_sec=100.0)
        mgr.transition(BehaviorMode.TRYHARD, force=True)
        assert mgr.transition(BehaviorMode.CHILL, force=True)
        assert mgr.mode == BehaviorMode.CHILL

    def test_evaluate_with_conditions(self):
        mgr = BehaviorModeManager(
            initial_mode=BehaviorMode.CHILL,
            transition_cooldown_sec=0.0,
        )
        gs = GameState()
        result = mgr.evaluate(gs, {"entered_combat": True})
        assert result == BehaviorMode.TRYHARD
        assert mgr.mode == BehaviorMode.TRYHARD

    def test_listener_called(self):
        mgr = BehaviorModeManager(transition_cooldown_sec=0.0)
        transitions = []
        mgr.on_transition(lambda old, new: transitions.append((old, new)))
        mgr.transition(BehaviorMode.RAGE)
        assert len(transitions) == 1
        assert transitions[0] == (BehaviorMode.ENTERTAINER, BehaviorMode.RAGE)


# ---------------------------------------------------------------------------
# EmotionBridge
# ---------------------------------------------------------------------------

class TestEmotionBridge:
    def test_all_43_emotions_have_mode_bias(self):
        """Every one of the 43 EmotionBERT labels must have an entry."""
        all_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral", "worry", "happiness", "fun", "hate", "autonomy",
            "safety", "understanding", "empty", "enthusiasm", "recreation",
            "sense of belonging", "meaning", "sustenance", "creativity", "boredom",
        ]
        for label in all_labels:
            assert label in EMOTION_MODE_BIAS, f"Missing mode bias for: {label}"

    def test_all_43_emotions_have_tts_instruct(self):
        """Every one of the 43 EmotionBERT labels must have a TTS instruct."""
        all_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral", "worry", "happiness", "fun", "hate", "autonomy",
            "safety", "understanding", "empty", "enthusiasm", "recreation",
            "sense of belonging", "meaning", "sustenance", "creativity", "boredom",
        ]
        for label in all_labels:
            assert label in EMOTION_TTS_INSTRUCT, f"Missing TTS instruct for: {label}"
            assert len(EMOTION_TTS_INSTRUCT[label]) > 0

    def test_tts_instruct_never_empty(self):
        bridge = EmotionBridge()
        # Known emotion
        assert len(bridge.get_tts_instruct("anger")) > 0
        # Unknown emotion falls back to default
        assert len(bridge.get_tts_instruct("nonexistent_emotion")) > 0

    def test_mode_bias_returns_dict(self):
        bridge = EmotionBridge()
        bias = bridge.get_mode_bias("anger")
        assert isinstance(bias, dict)
        assert BehaviorMode.RAGE in bias

    def test_process_emotion_result(self):
        bridge = EmotionBridge()

        class FakeResult:
            primary = "excitement"
            primary_confidence = 0.9

        bias, instruct = bridge.process_emotion_result(FakeResult())
        assert BehaviorMode.ENTERTAINER in bias
        assert "excitement" in instruct.lower() or "energy" in instruct.lower()


# ---------------------------------------------------------------------------
# DirectiveFactory + DirectiveBuffer
# ---------------------------------------------------------------------------

class TestDirectiveFactory:
    def test_dodge_left(self):
        d = DirectiveFactory.dodge_left()
        assert d.action.get(8) < 0  # LSTICK_X = left
        assert d.priority == DirectivePriority.HIGH

    def test_heal(self):
        d = DirectiveFactory.heal()
        assert d.priority == DirectivePriority.CRITICAL
        assert d.urgency == 0.8

    def test_custom(self):
        from streaming.gaming.types import ButtonIndex
        d = DirectiveFactory.custom(
            buttons={ButtonIndex.A: 1.0},
            urgency=0.9,
            description="press A",
        )
        assert d.action.get(ButtonIndex.A) == 1.0
        assert d.description == "press A"


class TestDirectiveBuffer:
    def test_push_and_peek(self):
        buf = DirectiveBuffer()
        d = DirectiveFactory.attack()
        buf.push(d)
        assert buf.size == 1
        assert buf.peek() is not None

    def test_peek_returns_highest_priority(self):
        buf = DirectiveBuffer()
        low = DirectiveFactory.explore_forward()  # LOW priority
        high = DirectiveFactory.heal()             # CRITICAL priority
        buf.push(low)
        buf.push(high)
        top = buf.peek()
        assert top.priority == DirectivePriority.CRITICAL

    def test_expired_directives_removed(self):
        buf = DirectiveBuffer()
        d = Directive(
            action=GamepadAction.neutral(),
            duration_sec=0.0,
            timestamp=time.time() - 1.0,
        )
        buf.push(d)
        assert buf.empty

    def test_clear(self):
        buf = DirectiveBuffer()
        buf.push(DirectiveFactory.attack())
        buf.clear()
        assert buf.empty


# ---------------------------------------------------------------------------
# AliceDirector
# ---------------------------------------------------------------------------

class TestAliceDirector:
    def test_tick_does_not_crash(self):
        director = AliceDirector()
        gs = GameState()
        director.tick(gs)
        assert director._tick_count == 1

    def test_critical_health_generates_heal(self):
        director = AliceDirector()
        gs = GameState(health=0.1, in_combat=True)
        director.tick(gs)
        # Should have pushed a heal directive
        d = director.directive_buffer.peek()
        assert d is not None
        assert "heal" in d.description

    def test_death_generates_commentary(self):
        director = AliceDirector()
        # First tick: alive
        director.tick(GameState(health=0.5, in_combat=True))
        # Second tick: dead
        director.tick(GameState(health=0.0, is_dead=True))
        commentary = director.pop_commentary()
        assert len(commentary) > 0
