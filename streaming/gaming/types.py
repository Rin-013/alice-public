"""
Gaming Types - All data structures for the gaming module.
=========================================================

Central type definitions used across capture, vision, input,
cognitive, mixer, stream, and comms submodules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Gamepad
# ---------------------------------------------------------------------------

class ButtonIndex(IntEnum):
    """Xbox-style button indices for NitroGen tensor output."""
    A = 0
    B = 1
    X = 2
    Y = 3
    LB = 4
    RB = 5
    LT = 6  # analog trigger (0-1)
    RT = 7  # analog trigger (0-1)
    LSTICK_X = 8
    LSTICK_Y = 9
    RSTICK_X = 10
    RSTICK_Y = 11
    DPAD_UP = 12
    DPAD_DOWN = 13
    DPAD_LEFT = 14
    DPAD_RIGHT = 15
    START = 16
    SELECT = 17
    LSTICK_PRESS = 18
    RSTICK_PRESS = 19


@dataclass
class GamepadAction:
    """
    Single frame of gamepad state from NitroGen or directive mixer.

    Values:
        buttons: 20-element array matching ButtonIndex ordering.
                 Digital buttons are 0.0/1.0, analog axes are -1.0..1.0,
                 triggers are 0.0..1.0.
        timestamp: When this action was generated.
        source: 'nitrogen', 'directive', 'mixed', or 'mock'.
    """
    buttons: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.float32))
    timestamp: float = field(default_factory=time.time)
    source: str = "mock"

    def __post_init__(self):
        if not isinstance(self.buttons, np.ndarray):
            self.buttons = np.array(self.buttons, dtype=np.float32)
        if self.buttons.shape != (20,):
            raise ValueError(f"buttons must be shape (20,), got {self.buttons.shape}")

    def get(self, button: ButtonIndex) -> float:
        return float(self.buttons[button])

    def set(self, button: ButtonIndex, value: float):
        self.buttons[button] = value

    @classmethod
    def neutral(cls) -> GamepadAction:
        """Return a no-input (idle) action."""
        return cls(buttons=np.zeros(20, dtype=np.float32), source="neutral")


# ---------------------------------------------------------------------------
# Behavior Modes
# ---------------------------------------------------------------------------

class BehaviorMode(Enum):
    """
    Six behavior modes that govern Alice's gaming personality.

    Each mode biases the directive mixer, commentary style, and avatar emotion.
    """
    TRYHARD = "tryhard"          # Focused, competitive, minimal chatter
    ENTERTAINER = "entertainer"  # Flashy plays, high commentary
    CHILL = "chill"              # Relaxed, chatty, low urgency
    RAGE = "rage"                # Frustrated, aggressive, loud
    CURIOUS = "curious"          # Exploring, investigative, asking questions
    CLUTCH = "clutch"            # High-stakes, intense focus, hype moments


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------

class DirectivePriority(IntEnum):
    """Priority levels for strategic directives."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Directive:
    """
    A strategic instruction from Alice's cognitive layer to the mixer.

    Directives modify NitroGen's reflex actions - e.g. "explore that area",
    "use this item", "dodge left". They are blended with NitroGen output
    via the directive mixer.
    """
    action: GamepadAction
    priority: DirectivePriority = DirectivePriority.NORMAL
    urgency: float = 0.5           # 0.0 (suggestion) to 1.0 (override)
    duration_sec: float = 1.0      # How long this directive should persist
    description: str = ""          # Human-readable intent
    timestamp: float = field(default_factory=time.time)
    expired: bool = False

    def is_expired(self) -> bool:
        if self.expired:
            return True
        return (time.time() - self.timestamp) > self.duration_sec

    def remaining_sec(self) -> float:
        return max(0.0, self.duration_sec - (time.time() - self.timestamp))


# ---------------------------------------------------------------------------
# Game State
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Extracted game state from vision pipeline.

    Populated by GameStateExtractor from NitroGen's frame analysis
    or simple heuristics (health bar detection, etc.).
    """
    # Health/resources (0.0-1.0 normalized)
    health: float = 1.0
    mana: float = 1.0
    stamina: float = 1.0

    # Situational
    in_combat: bool = False
    in_menu: bool = False
    in_cutscene: bool = False
    is_dead: bool = False

    # Positional (game-specific, often unavailable)
    area_name: str = ""
    objective: str = ""

    # Raw features from vision model
    features: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0

    @property
    def is_critical(self) -> bool:
        """Health below 25% and in combat."""
        return self.health < 0.25 and self.in_combat

    @property
    def is_safe(self) -> bool:
        """Not in combat, not dead, not in cutscene."""
        return not self.in_combat and not self.is_dead and not self.in_cutscene


# ---------------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------------

class CommentaryPriority(IntEnum):
    """Priority levels for stream commentary."""
    FILLER = 0       # "Hmm...", idle chatter
    NORMAL = 1       # Standard game commentary
    REACTION = 2     # React to game events
    EXCITEMENT = 3   # Big moments, clutch plays
    CRITICAL = 4     # Death, rage, celebration


@dataclass
class CommentaryRequest:
    """
    A request for Alice to say something on stream.

    Generated by the director or commentary pipeline, consumed by TTS.
    """
    text: str
    priority: CommentaryPriority = CommentaryPriority.NORMAL
    emotion: str = "neutral"               # TTS emotion instruct
    tts_instruct: str = "Speak naturally"  # Full TTS instruct string
    interrupt: bool = False                # Can interrupt current speech
    min_gap_sec: float = 0.0               # Min time since last commentary
    timestamp: float = field(default_factory=time.time)
    source: str = "director"               # Where this request originated

    @property
    def age_sec(self) -> float:
        return time.time() - self.timestamp


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclass
class CapturedFrame:
    """A single captured game frame."""
    image: np.ndarray                      # BGR or RGB HWC array
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0
    width: int = 0
    height: int = 0
    source: str = "mock"

    def __post_init__(self):
        if self.image is not None and self.width == 0:
            if self.image.ndim >= 2:
                self.height, self.width = self.image.shape[:2]
