"""
Behavior Modes - 6-mode state machine for Alice's gaming personality.
=====================================================================

Modes: TRYHARD, ENTERTAINER, CHILL, RAGE, CURIOUS, CLUTCH

Transitions are driven by game state, emotion, and explicit triggers.
Each mode biases the mixer, commentary style, and avatar emotion.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

from streaming.gaming.types import BehaviorMode, GameState
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Transition Rules
# ---------------------------------------------------------------------------

# Each rule: (from_mode, to_mode, condition_name)
# Conditions are evaluated by BehaviorModeManager._check_transition

TransitionRule = Tuple[BehaviorMode, BehaviorMode, str]

DEFAULT_TRANSITIONS: List[TransitionRule] = [
    # Combat pressure → TRYHARD
    (BehaviorMode.CHILL, BehaviorMode.TRYHARD, "entered_combat"),
    (BehaviorMode.ENTERTAINER, BehaviorMode.TRYHARD, "health_critical"),
    (BehaviorMode.CURIOUS, BehaviorMode.TRYHARD, "entered_combat"),

    # Death → RAGE (probabilistic, handled in director)
    (BehaviorMode.TRYHARD, BehaviorMode.RAGE, "died_angry"),
    (BehaviorMode.CLUTCH, BehaviorMode.RAGE, "died_angry"),

    # Low-stakes → CHILL
    (BehaviorMode.TRYHARD, BehaviorMode.CHILL, "combat_ended_safe"),
    (BehaviorMode.RAGE, BehaviorMode.CHILL, "rage_cooldown"),

    # Big moment → CLUTCH
    (BehaviorMode.TRYHARD, BehaviorMode.CLUTCH, "health_critical_combat"),
    (BehaviorMode.ENTERTAINER, BehaviorMode.CLUTCH, "health_critical_combat"),

    # Exploration → CURIOUS
    (BehaviorMode.CHILL, BehaviorMode.CURIOUS, "new_area"),
    (BehaviorMode.ENTERTAINER, BehaviorMode.CURIOUS, "new_area"),

    # Default return → ENTERTAINER
    (BehaviorMode.RAGE, BehaviorMode.ENTERTAINER, "rage_timeout"),
    (BehaviorMode.CLUTCH, BehaviorMode.ENTERTAINER, "clutch_resolved"),
    (BehaviorMode.CURIOUS, BehaviorMode.ENTERTAINER, "curiosity_satisfied"),
]


class BehaviorModeManager:
    """
    State machine managing Alice's behavior mode during gaming.

    The current mode influences:
    - Mixer blend weight (mode_bias in config)
    - Commentary style and frequency
    - Avatar emotion/expression
    """

    def __init__(
        self,
        initial_mode: BehaviorMode = BehaviorMode.ENTERTAINER,
        transition_cooldown_sec: float = 5.0,
        transitions: Optional[List[TransitionRule]] = None,
    ):
        self._mode = initial_mode
        self._prev_mode: Optional[BehaviorMode] = None
        self._transition_cooldown = transition_cooldown_sec
        self._last_transition_time = 0.0
        self._mode_enter_time = time.time()
        self._transitions = transitions or DEFAULT_TRANSITIONS
        self._listeners: List[Callable[[BehaviorMode, BehaviorMode], None]] = []

        logger.info(f"BehaviorModeManager initialized: {self._mode.value}")

    # --- Properties ---

    @property
    def mode(self) -> BehaviorMode:
        return self._mode

    @property
    def previous_mode(self) -> Optional[BehaviorMode]:
        return self._prev_mode

    @property
    def mode_duration(self) -> float:
        """Seconds spent in current mode."""
        return time.time() - self._mode_enter_time

    @property
    def can_transition(self) -> bool:
        """Whether the cooldown has elapsed."""
        return (time.time() - self._last_transition_time) >= self._transition_cooldown

    # --- Transition ---

    def transition(self, new_mode: BehaviorMode, force: bool = False) -> bool:
        """
        Attempt to transition to a new mode.

        Args:
            new_mode: Target mode.
            force: Skip cooldown check.

        Returns:
            True if transition occurred.
        """
        if new_mode == self._mode:
            return False

        if not force and not self.can_transition:
            return False

        self._prev_mode = self._mode
        self._mode = new_mode
        self._last_transition_time = time.time()
        self._mode_enter_time = time.time()

        logger.info(f"Mode transition: {self._prev_mode.value} → {self._mode.value}")

        for listener in self._listeners:
            try:
                listener(self._prev_mode, self._mode)
            except Exception as e:
                logger.error(f"Transition listener error: {e}")

        return True

    def evaluate(self, game_state: GameState, conditions: Dict[str, bool]) -> Optional[BehaviorMode]:
        """
        Evaluate transition rules against current conditions.

        Args:
            game_state: Current game state.
            conditions: Dict of named condition flags (e.g. {"entered_combat": True}).

        Returns:
            New mode if a transition was triggered, else None.
        """
        if not self.can_transition:
            return None

        for from_mode, to_mode, condition_name in self._transitions:
            if self._mode == from_mode and conditions.get(condition_name, False):
                if self.transition(to_mode):
                    return to_mode

        return None

    # --- Listeners ---

    def on_transition(self, callback: Callable[[BehaviorMode, BehaviorMode], None]):
        """Register a callback for mode transitions: callback(old_mode, new_mode)."""
        self._listeners.append(callback)

    # --- Info ---

    def get_status(self) -> Dict:
        return {
            "mode": self._mode.value,
            "previous": self._prev_mode.value if self._prev_mode else None,
            "duration_sec": round(self.mode_duration, 1),
            "can_transition": self.can_transition,
        }
