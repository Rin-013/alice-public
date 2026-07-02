"""
Alice Director - Strategic Decision Engine
===========================================

The director evaluates game state at ~6 Hz, decides on strategic actions,
generates directives for the mixer, and triggers commentary.

This is heuristic/state-machine based — actual strategic reasoning comes
from the v4 Mind/Alice cognitive loop. The director bridges game state
to actionable directives and commentary.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from streaming.gaming.cognitive.behavior_modes import BehaviorModeManager
from streaming.gaming.cognitive.directive import DirectiveBuffer, DirectiveFactory
from streaming.gaming.cognitive.emotion_bridge import EmotionBridge
from streaming.gaming.types import (
    BehaviorMode,
    CommentaryPriority,
    CommentaryRequest,
    Directive,
    GameState,
)
from streaming.gaming.utils.logging_utils import get_logger
from streaming.gaming.utils.timing import Cooldown

logger = get_logger(__name__)


class AliceDirector:
    """
    Strategic decision engine for Alice's gaming behavior.

    Runs at director tick rate (~6 Hz). Each tick:
    1. Evaluates game state
    2. Checks behavior mode transitions
    3. Generates directives for the mixer
    4. Queues commentary requests

    The director does NOT run NitroGen or handle frame capture — it only
    consumes GameState and produces Directives + CommentaryRequests.
    """

    def __init__(
        self,
        mode_manager: Optional[BehaviorModeManager] = None,
        emotion_bridge: Optional[EmotionBridge] = None,
        directive_buffer: Optional[DirectiveBuffer] = None,
        config: Optional[Dict] = None,
    ):
        cfg = config or {}
        self._mode_manager = mode_manager or BehaviorModeManager()
        self._emotion_bridge = emotion_bridge or EmotionBridge()
        self._directive_buffer = directive_buffer or DirectiveBuffer()
        self._factory = DirectiveFactory()

        # Config
        self._health_critical = cfg.get("health_critical_threshold", 0.25)
        self._combat_urgency_boost = cfg.get("combat_urgency_boost", 0.3)
        self._death_rage_chance = cfg.get("death_rage_chance", 0.4)
        self._exploration_curiosity_chance = cfg.get("exploration_curiosity_chance", 0.3)

        # State tracking
        self._prev_game_state: Optional[GameState] = None
        self._was_in_combat = False
        self._was_dead = False
        self._known_areas: set = set()
        self._tick_count = 0

        # Cooldowns
        self._heal_cd = Cooldown(seconds=3.0)
        self._dodge_cd = Cooldown(seconds=0.5)
        self._commentary_cd = Cooldown(seconds=3.0)

        # Pending commentary (consumed by stream.commentary)
        self._pending_commentary: List[CommentaryRequest] = []

        logger.info("AliceDirector initialized")

    # --- Properties ---

    @property
    def mode(self) -> BehaviorMode:
        return self._mode_manager.mode

    @property
    def mode_manager(self) -> BehaviorModeManager:
        return self._mode_manager

    @property
    def directive_buffer(self) -> DirectiveBuffer:
        return self._directive_buffer

    @property
    def emotion_bridge(self) -> EmotionBridge:
        return self._emotion_bridge

    # --- Main tick ---

    def tick(self, game_state: GameState, emotion_primary: str = "neutral") -> None:
        """
        Run one director evaluation cycle.

        Args:
            game_state: Current extracted game state.
            emotion_primary: Current primary emotion from EmotionBERT.
        """
        self._tick_count += 1

        # Build transition conditions from game state
        conditions = self._build_conditions(game_state)

        # Check behavior mode transitions
        self._mode_manager.evaluate(game_state, conditions)

        # Generate directives based on game state + mode
        self._generate_directives(game_state)

        # Generate commentary
        self._generate_commentary(game_state, emotion_primary)

        # Track previous state
        self._prev_game_state = game_state
        self._was_in_combat = game_state.in_combat
        self._was_dead = game_state.is_dead

    # --- Condition building ---

    def _build_conditions(self, gs: GameState) -> Dict[str, bool]:
        """Build named conditions for behavior mode transition rules."""
        just_entered_combat = gs.in_combat and not self._was_in_combat
        just_died = gs.is_dead and not self._was_dead
        combat_ended = not gs.in_combat and self._was_in_combat
        health_critical = gs.health < self._health_critical and gs.in_combat
        new_area = bool(gs.area_name and gs.area_name not in self._known_areas)

        if new_area and gs.area_name:
            self._known_areas.add(gs.area_name)

        return {
            "entered_combat": just_entered_combat,
            "health_critical": gs.health < self._health_critical,
            "health_critical_combat": health_critical,
            "combat_ended_safe": combat_ended and gs.is_safe,
            "died_angry": just_died and random.random() < self._death_rage_chance,
            "rage_cooldown": (
                self._mode_manager.mode == BehaviorMode.RAGE
                and self._mode_manager.mode_duration > 10.0
            ),
            "rage_timeout": (
                self._mode_manager.mode == BehaviorMode.RAGE
                and self._mode_manager.mode_duration > 20.0
            ),
            "clutch_resolved": (
                self._mode_manager.mode == BehaviorMode.CLUTCH
                and gs.is_safe
            ),
            "new_area": new_area and random.random() < self._exploration_curiosity_chance,
            "curiosity_satisfied": (
                self._mode_manager.mode == BehaviorMode.CURIOUS
                and self._mode_manager.mode_duration > 15.0
            ),
        }

    # --- Directive generation ---

    def _generate_directives(self, gs: GameState):
        """Generate directives based on current game state and behavior mode."""
        mode = self._mode_manager.mode

        # Critical health → heal
        if gs.is_critical and self._heal_cd.ready():
            self._directive_buffer.push(self._factory.heal(urgency=0.9))
            self._heal_cd.trigger()

        # In combat → mode-specific actions
        if gs.in_combat:
            if mode in (BehaviorMode.TRYHARD, BehaviorMode.CLUTCH):
                # More aggressive
                if self._dodge_cd.ready() and gs.health < 0.5:
                    direction = random.choice([self._factory.dodge_left, self._factory.dodge_right])
                    self._directive_buffer.push(direction(urgency=0.6))
                    self._dodge_cd.trigger()

        # Safe + curious → explore
        if gs.is_safe and mode == BehaviorMode.CURIOUS:
            if self._directive_buffer.empty:
                self._directive_buffer.push(self._factory.explore_forward(urgency=0.2))

    # --- Commentary generation ---

    def _generate_commentary(self, gs: GameState, emotion: str):
        """Generate commentary requests based on game events."""
        if not self._commentary_cd.ready():
            return

        mode = self._mode_manager.mode
        tts_instruct = self._emotion_bridge.get_tts_instruct(emotion)

        # Death reactions
        if gs.is_dead and not self._was_dead:
            if mode == BehaviorMode.RAGE:
                self._queue_commentary(
                    text=random.choice([
                        "Are you KIDDING me right now?!",
                        "NO! That was so unfair!",
                        "I literally pressed dodge! This game is broken!",
                    ]),
                    priority=CommentaryPriority.CRITICAL,
                    emotion="anger",
                    interrupt=True,
                )
            else:
                self._queue_commentary(
                    text=random.choice([
                        "Okay... that happened.",
                        "Well, that could have gone better.",
                        "Oof. Let's not do that again.",
                    ]),
                    priority=CommentaryPriority.REACTION,
                    emotion="disappointment",
                )
            return

        # Entering combat
        if gs.in_combat and not self._was_in_combat:
            if mode == BehaviorMode.ENTERTAINER:
                self._queue_commentary(
                    text=random.choice([
                        "Oh, here we go!",
                        "Fight time!",
                        "Let's see what you've got!",
                    ]),
                    priority=CommentaryPriority.REACTION,
                    emotion="excitement",
                )
            return

        # Critical health
        if gs.is_critical and not (self._prev_game_state and self._prev_game_state.is_critical):
            self._queue_commentary(
                text=random.choice([
                    "Okay okay okay, we need to heal!",
                    "Health is NOT looking good!",
                    "This is fine... this is fine...",
                ]),
                priority=CommentaryPriority.EXCITEMENT,
                emotion="nervousness",
                interrupt=True,
            )
            return

    def _queue_commentary(
        self,
        text: str,
        priority: CommentaryPriority = CommentaryPriority.NORMAL,
        emotion: str = "neutral",
        interrupt: bool = False,
    ):
        tts_instruct = self._emotion_bridge.get_tts_instruct(emotion)
        req = CommentaryRequest(
            text=text,
            priority=priority,
            emotion=emotion,
            tts_instruct=tts_instruct,
            interrupt=interrupt,
            source="director",
        )
        self._pending_commentary.append(req)
        self._commentary_cd.trigger()

    def pop_commentary(self) -> List[CommentaryRequest]:
        """Pop all pending commentary requests (consumed by stream.commentary)."""
        items = self._pending_commentary
        self._pending_commentary = []
        return items

    # --- Status ---

    def get_status(self) -> Dict:
        return {
            "mode": self._mode_manager.get_status(),
            "directive_buffer": self._directive_buffer.get_status(),
            "emotion_bridge": self._emotion_bridge.get_status(),
            "tick_count": self._tick_count,
            "pending_commentary": len(self._pending_commentary),
            "known_areas": len(self._known_areas),
        }
