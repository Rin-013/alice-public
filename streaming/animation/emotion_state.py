#!/usr/bin/env python3
"""
Emotion State - Accumulator-based emotion with momentum.

Emotions don't flip instantly. Each detection adds weight to that emotion,
other emotions decay, and the dominant emotion only changes when a new one
builds enough momentum. This creates natural, gradual emotional shifts.

EmotionBERT writes here, motion engine reads from here.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict


EMOTION_STATE_FILE = Path(__file__).parent / "emotion_state.json"

# All coarse emotions the system uses
EMOTIONS = ["neutral", "happy", "excited", "sassy", "sad", "angry", "surprised", "thinking", "tired"]

@dataclass
class EmotionState:
    """Current emotion state. `scores` is kept as a per-emotion dict for
    procedural motion's profile blending — set_emotion writes 1.0 on the
    dominant and 0.0 elsewhere."""
    emotion: str = "neutral"
    confidence: float = 1.0
    timestamp: float = 0.0
    is_speaking: bool = False
    scores: Dict[str, float] = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})


def _load_state() -> EmotionState:
    """Load state from file, defaulting to neutral if missing/corrupt."""
    if not EMOTION_STATE_FILE.exists():
        state = EmotionState(timestamp=time.time())
        state.scores["neutral"] = 1.0
        return state

    try:
        with open(EMOTION_STATE_FILE, 'r') as f:
            data = json.load(f)
            if "scores" not in data:
                scores = {e: 0.0 for e in EMOTIONS}
                scores[data.get("emotion", "neutral")] = 1.0
                data["scores"] = scores
            return EmotionState(**data)
    except Exception:
        state = EmotionState(timestamp=time.time())
        state.scores["neutral"] = 1.0
        return state


def _save_state(state: EmotionState):
    """Write state to file."""
    with open(EMOTION_STATE_FILE, 'w') as f:
        json.dump(asdict(state), f, indent=2)


def set_emotion(emotion: str, confidence: float = 1.0, is_speaking: bool = False):
    """
    Set the dominant emotion immediately. Write-through, no accumulator.

    Previously this funneled detections through a multi-step accumulator
    (gain 0.35, threshold 0.45) which was useful when EmotionBERT ran at
    high rate and produced noisy outputs. Mind now reports deliberately
    once per turn, so the accumulator just added a 2-turn lag with no
    upside. Now: caller's emotion lands on the next read.
    """
    emotion = emotion.lower()
    if emotion not in EMOTIONS:
        emotion = "neutral"

    state = EmotionState(
        emotion=emotion,
        confidence=confidence,
        timestamp=time.time(),
        is_speaking=is_speaking,
    )
    state.scores = {e: 0.0 for e in EMOTIONS}
    state.scores[emotion] = 1.0
    _save_state(state)


def get_emotion() -> EmotionState:
    """
    Get current emotion state (called by motion engine).

    Returns:
        EmotionState with dominant emotion, confidence, and full scores
    """
    return _load_state()


def set_speaking(is_speaking: bool):
    """Update speaking state without touching emotion scores."""
    state = _load_state()
    state.is_speaking = is_speaking
    state.timestamp = time.time()
    _save_state(state)


if __name__ == '__main__':
    for emo in ["happy", "excited", "sad", "neutral"]:
        set_emotion(emo)
        s = get_emotion()
        print(f"set({emo}) → emotion={s.emotion} confidence={s.confidence:.2f}")
    print(f"\nState file: {EMOTION_STATE_FILE}")
