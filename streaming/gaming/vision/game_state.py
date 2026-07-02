"""
Game State Extractor
====================

Extracts GameState from frames using simple heuristics
(health bar detection, menu detection, etc.).

In production, NitroGen or a separate vision model would provide richer
game state. This module provides baseline heuristics and a mock.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from streaming.gaming.types import CapturedFrame, GameState
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GameStateExtractor:
    """
    Extracts game state from frames.

    Currently uses simple heuristics. In production:
    - NitroGen provides features alongside actions
    - Dedicated classifiers for menus/combat/death
    """

    def __init__(self):
        self._frame_count = 0

    def extract(self, frame: CapturedFrame) -> GameState:
        """
        Extract game state from a captured frame.

        This is a heuristic stub — returns plausible defaults.
        """
        self._frame_count += 1

        return GameState(
            health=1.0,
            mana=1.0,
            stamina=1.0,
            in_combat=False,
            in_menu=False,
            in_cutscene=False,
            is_dead=False,
            area_name="",
            objective="",
            features={},
            timestamp=frame.timestamp,
            frame_number=frame.frame_number,
        )


class MockGameStateExtractor(GameStateExtractor):
    """
    Mock extractor that generates scripted game state sequences.

    Useful for testing director logic without real game frames.
    """

    def __init__(self):
        super().__init__()
        self._scenario_step = 0

    def extract(self, frame: CapturedFrame) -> GameState:
        self._frame_count += 1
        self._scenario_step += 1

        # Simple cycling scenario: safe → combat → critical → dead → safe
        cycle = self._scenario_step % 200
        if cycle < 80:
            return GameState(health=0.9, in_combat=False, timestamp=frame.timestamp, frame_number=frame.frame_number)
        elif cycle < 140:
            health = max(0.1, 0.9 - (cycle - 80) * 0.013)
            return GameState(health=health, in_combat=True, timestamp=frame.timestamp, frame_number=frame.frame_number)
        elif cycle < 160:
            return GameState(health=0.15, in_combat=True, timestamp=frame.timestamp, frame_number=frame.frame_number)
        elif cycle < 180:
            return GameState(health=0.0, is_dead=True, timestamp=frame.timestamp, frame_number=frame.frame_number)
        else:
            return GameState(health=0.8, in_combat=False, area_name="New Area", timestamp=frame.timestamp, frame_number=frame.frame_number)

    def reset(self):
        self._scenario_step = 0
