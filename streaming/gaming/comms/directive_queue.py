"""
Directive Queue
===============

Thread-safe priority queue for passing directives between
the director (producer) and the mixer (consumer).
"""

from __future__ import annotations

import heapq
import threading
import time
from typing import Dict, Optional

from streaming.gaming.types import Directive
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DirectiveQueue:
    """
    Thread-safe priority queue for Directive objects.

    Producer: AliceDirector (runs on director tick thread)
    Consumer: DirectiveMixer (runs on main game loop thread)
    """

    def __init__(self, maxsize: int = 64):
        self._heap: list = []
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._counter = 0  # Tiebreaker for equal priorities

    def put(self, directive: Directive):
        """Add a directive (thread-safe). Drops lowest priority if full."""
        with self._lock:
            entry = (-directive.priority, self._counter, directive)
            self._counter += 1

            if len(self._heap) >= self._maxsize:
                heapq.heappushpop(self._heap, entry)
            else:
                heapq.heappush(self._heap, entry)

    def get(self) -> Optional[Directive]:
        """Pop highest-priority directive (thread-safe). Returns None if empty."""
        with self._lock:
            self._expire_locked()
            if not self._heap:
                return None
            _, _, directive = heapq.heappop(self._heap)
            return directive

    def peek(self) -> Optional[Directive]:
        """Peek at highest-priority directive without removing (thread-safe)."""
        with self._lock:
            self._expire_locked()
            if not self._heap:
                return None
            _, _, directive = self._heap[0]
            return directive

    def _expire_locked(self):
        """Remove expired directives. Must hold lock."""
        self._heap = [
            entry for entry in self._heap
            if not entry[2].is_expired()
        ]
        heapq.heapify(self._heap)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def empty(self) -> bool:
        return self.size == 0

    def clear(self):
        with self._lock:
            self._heap.clear()
            self._counter = 0

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "size": len(self._heap),
                "maxsize": self._maxsize,
            }
