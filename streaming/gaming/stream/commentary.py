"""
Commentary Pipeline
===================

Priority-queue based commentary system that controls when Alice speaks
during gaming. Handles timing, interrupts, and TTS parameter generation.
"""

from __future__ import annotations

import heapq
import time
from typing import Callable, Dict, List, Optional

from streaming.gaming.types import CommentaryPriority, CommentaryRequest
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TTS_INSTRUCT = "Speak naturally with a playful tone"


class CommentaryPipeline:
    """
    Manages a priority queue of commentary requests.

    Features:
    - Priority-based ordering (higher priority speaks first)
    - Minimum gap enforcement (prevents talking too fast)
    - TTL expiration (stale commentary is dropped)
    - Interrupt support (high-priority can cut current speech)
    - TTS instruct always populated (never empty)
    """

    def __init__(
        self,
        min_gap_sec: float = 3.0,
        max_queue_size: int = 10,
        ttl_sec: float = 15.0,
        interrupt_priority: int = 3,
        default_tts_instruct: str = DEFAULT_TTS_INSTRUCT,
    ):
        self._min_gap_sec = min_gap_sec
        self._max_queue_size = max_queue_size
        self._ttl_sec = ttl_sec
        self._interrupt_priority = interrupt_priority
        self._default_tts_instruct = default_tts_instruct

        # Priority queue: (-priority, counter, request)
        # Negative priority for max-heap behavior with heapq (min-heap)
        # Counter is tiebreaker so heapq never compares CommentaryRequest
        self._queue: List[tuple] = []
        self._counter: int = 0
        self._last_speak_time: float = 0.0
        self._is_speaking: bool = False
        self._current_request: Optional[CommentaryRequest] = None
        self._total_spoken: int = 0

        # Optional callback when speech should start
        self._on_speak: Optional[Callable[[CommentaryRequest], None]] = None

    # --- Queue management ---

    def submit(self, request: CommentaryRequest):
        """
        Submit a commentary request to the pipeline.

        May be dropped if queue is full (lowest priority gets evicted).
        """
        # Ensure TTS instruct is never empty
        if not request.tts_instruct or request.tts_instruct.strip() == "":
            request.tts_instruct = self._default_tts_instruct

        # Check for interrupt
        if request.interrupt and request.priority >= self._interrupt_priority:
            if self._is_speaking:
                logger.info(f"Interrupt: {request.text[:40]}...")
                self._is_speaking = False
                self._current_request = None

        # Add to queue (counter as tiebreaker — never compares request objects)
        self._counter += 1
        entry = (-request.priority, self._counter, request)
        if len(self._queue) >= self._max_queue_size:
            # Evict lowest priority
            heapq.heappush(self._queue, entry)
            heapq.heappop(self._queue)  # removes smallest = lowest priority
        else:
            heapq.heappush(self._queue, entry)

    def submit_many(self, requests: List[CommentaryRequest]):
        """Submit multiple requests."""
        for req in requests:
            self.submit(req)

    # --- Consumption ---

    def next(self) -> Optional[CommentaryRequest]:
        """
        Get the next commentary request that should be spoken.

        Returns None if:
        - Queue is empty
        - Min gap hasn't elapsed
        - Currently speaking

        Automatically drops expired requests.
        """
        if self._is_speaking:
            return None

        now = time.time()

        # Enforce min gap
        if (now - self._last_speak_time) < self._min_gap_sec:
            return None

        # Find next valid request
        while self._queue:
            neg_priority, cnt, request = heapq.heappop(self._queue)

            # Check TTL
            if request.age_sec > self._ttl_sec:
                continue  # Expired, skip

            # Check min_gap_sec on the request itself
            if request.min_gap_sec > 0:
                if (now - self._last_speak_time) < request.min_gap_sec:
                    # Re-add and wait
                    heapq.heappush(self._queue, (neg_priority, cnt, request))
                    return None

            # Valid request
            self._current_request = request
            self._is_speaking = True
            self._last_speak_time = now
            self._total_spoken += 1

            if self._on_speak:
                try:
                    self._on_speak(request)
                except Exception as e:
                    logger.error(f"on_speak callback error: {e}")

            return request

        return None

    def mark_done(self):
        """Mark current speech as finished (called when TTS completes)."""
        self._is_speaking = False
        self._current_request = None

    # --- Callbacks ---

    def on_speak(self, callback: Callable[[CommentaryRequest], None]):
        """Register callback for when speech should start."""
        self._on_speak = callback

    # --- Properties ---

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def current_request(self) -> Optional[CommentaryRequest]:
        return self._current_request

    def get_status(self) -> Dict:
        return {
            "queue_size": len(self._queue),
            "is_speaking": self._is_speaking,
            "total_spoken": self._total_spoken,
            "min_gap_sec": self._min_gap_sec,
            "current": self._current_request.text[:50] if self._current_request else None,
        }
