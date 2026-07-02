"""
Directive Factory + Buffer
==========================

Creates strategic directives and manages a time-decaying buffer
of active directives for the mixer to consume.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from streaming.gaming.types import (
    BehaviorMode,
    ButtonIndex,
    Directive,
    DirectivePriority,
    GamepadAction,
    GameState,
)
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DirectiveFactory:
    """
    Creates common directives from high-level strategic intents.

    These map strategic decisions (e.g. "dodge left", "use healing item")
    to concrete GamepadAction sequences.
    """

    @staticmethod
    def dodge_left(urgency: float = 0.7, duration: float = 0.3) -> Directive:
        action = GamepadAction.neutral()
        action.set(ButtonIndex.LSTICK_X, -1.0)
        action.set(ButtonIndex.B, 1.0)  # Typical dodge button
        return Directive(
            action=action,
            priority=DirectivePriority.HIGH,
            urgency=urgency,
            duration_sec=duration,
            description="dodge left",
        )

    @staticmethod
    def dodge_right(urgency: float = 0.7, duration: float = 0.3) -> Directive:
        action = GamepadAction.neutral()
        action.set(ButtonIndex.LSTICK_X, 1.0)
        action.set(ButtonIndex.B, 1.0)
        return Directive(
            action=action,
            priority=DirectivePriority.HIGH,
            urgency=urgency,
            duration_sec=duration,
            description="dodge right",
        )

    @staticmethod
    def heal(urgency: float = 0.8, duration: float = 0.5) -> Directive:
        action = GamepadAction.neutral()
        action.set(ButtonIndex.DPAD_DOWN, 1.0)  # Common heal shortcut
        return Directive(
            action=action,
            priority=DirectivePriority.CRITICAL,
            urgency=urgency,
            duration_sec=duration,
            description="use healing item",
        )

    @staticmethod
    def explore_forward(urgency: float = 0.3, duration: float = 2.0) -> Directive:
        action = GamepadAction.neutral()
        action.set(ButtonIndex.LSTICK_Y, 1.0)
        return Directive(
            action=action,
            priority=DirectivePriority.LOW,
            urgency=urgency,
            duration_sec=duration,
            description="move forward to explore",
        )

    @staticmethod
    def attack(urgency: float = 0.6, duration: float = 0.2) -> Directive:
        action = GamepadAction.neutral()
        action.set(ButtonIndex.RT, 1.0)
        return Directive(
            action=action,
            priority=DirectivePriority.NORMAL,
            urgency=urgency,
            duration_sec=duration,
            description="attack",
        )

    @staticmethod
    def custom(
        buttons: Dict[ButtonIndex, float],
        urgency: float = 0.5,
        priority: DirectivePriority = DirectivePriority.NORMAL,
        duration: float = 1.0,
        description: str = "custom directive",
    ) -> Directive:
        action = GamepadAction.neutral()
        for btn, val in buttons.items():
            action.set(btn, val)
        return Directive(
            action=action,
            priority=priority,
            urgency=urgency,
            duration_sec=duration,
            description=description,
        )


class DirectiveBuffer:
    """
    Holds active directives ordered by priority, expiring old ones.

    The mixer reads the top directive from this buffer each frame.
    """

    def __init__(self, max_size: int = 16):
        self._buffer: deque[Directive] = deque(maxlen=max_size)
        self._max_size = max_size

    def push(self, directive: Directive):
        """Add a directive to the buffer."""
        self._expire()
        self._buffer.append(directive)

    def peek(self) -> Optional[Directive]:
        """
        Get the highest-priority non-expired directive without removing it.
        """
        self._expire()
        if not self._buffer:
            return None
        # Return highest priority, then highest urgency
        return max(
            self._buffer,
            key=lambda d: (d.priority, d.urgency),
        )

    def pop(self) -> Optional[Directive]:
        """
        Remove and return the highest-priority non-expired directive.
        """
        d = self.peek()
        if d is not None:
            self._buffer.remove(d)
        return d

    def clear(self):
        self._buffer.clear()

    def _expire(self):
        """Remove expired directives."""
        self._buffer = deque(
            (d for d in self._buffer if not d.is_expired()),
            maxlen=self._max_size,
        )

    @property
    def size(self) -> int:
        self._expire()
        return len(self._buffer)

    @property
    def empty(self) -> bool:
        return self.size == 0

    def get_status(self) -> Dict:
        self._expire()
        return {
            "size": len(self._buffer),
            "max_size": self._max_size,
            "directives": [
                {"desc": d.description, "priority": d.priority.name, "urgency": d.urgency, "remaining": round(d.remaining_sec(), 2)}
                for d in self._buffer
            ],
        }
