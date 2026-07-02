"""Tests for gaming.stream.commentary — pipeline timing and priority."""

import time

import pytest

from streaming.gaming.stream.commentary import CommentaryPipeline
from streaming.gaming.types import CommentaryPriority, CommentaryRequest


class TestCommentaryPipeline:
    def _make_pipeline(self, **kwargs) -> CommentaryPipeline:
        return CommentaryPipeline(
            min_gap_sec=kwargs.get("min_gap_sec", 0.0),
            max_queue_size=kwargs.get("max_queue_size", 10),
            ttl_sec=kwargs.get("ttl_sec", 60.0),
            interrupt_priority=kwargs.get("interrupt_priority", 3),
        )

    # --- Basic queue ---

    def test_submit_and_next(self):
        pipe = self._make_pipeline()
        pipe.submit(CommentaryRequest(text="hello", priority=CommentaryPriority.NORMAL))
        req = pipe.next()
        assert req is not None
        assert req.text == "hello"

    def test_empty_returns_none(self):
        pipe = self._make_pipeline()
        assert pipe.next() is None

    # --- Priority ordering ---

    def test_higher_priority_first(self):
        pipe = self._make_pipeline()
        pipe.submit(CommentaryRequest(text="low", priority=CommentaryPriority.FILLER))
        pipe.submit(CommentaryRequest(text="high", priority=CommentaryPriority.CRITICAL))
        req = pipe.next()
        assert req.text == "high"

    # --- Min gap enforcement ---

    def test_min_gap_blocks(self):
        pipe = self._make_pipeline(min_gap_sec=10.0)
        pipe.submit(CommentaryRequest(text="first", priority=CommentaryPriority.NORMAL))
        first = pipe.next()
        assert first is not None
        pipe.mark_done()

        pipe.submit(CommentaryRequest(text="second", priority=CommentaryPriority.NORMAL))
        second = pipe.next()
        assert second is None  # Blocked by min gap

    def test_min_gap_allows_after_elapsed(self):
        pipe = self._make_pipeline(min_gap_sec=0.0)
        pipe.submit(CommentaryRequest(text="first"))
        pipe.next()
        pipe.mark_done()
        pipe.submit(CommentaryRequest(text="second"))
        assert pipe.next() is not None

    # --- TTL expiration ---

    def test_expired_requests_dropped(self):
        pipe = self._make_pipeline(ttl_sec=0.0)
        pipe.submit(CommentaryRequest(
            text="stale",
            timestamp=time.time() - 10.0,
        ))
        assert pipe.next() is None

    # --- Speaking state ---

    def test_wont_dequeue_while_speaking(self):
        pipe = self._make_pipeline()
        pipe.submit(CommentaryRequest(text="test"))
        pipe.next()  # Sets is_speaking = True
        # Don't mark_done — still speaking
        pipe.submit(CommentaryRequest(text="queued"))
        assert pipe.next() is None  # Blocked by is_speaking

    def test_mark_done_allows_next(self):
        pipe = self._make_pipeline()
        pipe.submit(CommentaryRequest(text="first"))
        pipe.next()
        pipe.mark_done()
        pipe.submit(CommentaryRequest(text="second"))
        assert pipe.next() is not None

    # --- Interrupt ---

    def test_interrupt_clears_speaking(self):
        pipe = self._make_pipeline(interrupt_priority=3)
        pipe.submit(CommentaryRequest(text="normal"))
        pipe.next()  # Now speaking
        assert pipe.is_speaking

        # Submit an interrupt
        pipe.submit(CommentaryRequest(
            text="INTERRUPT!",
            priority=CommentaryPriority.CRITICAL,
            interrupt=True,
        ))
        assert not pipe.is_speaking  # Interrupt cleared it

    def test_low_priority_interrupt_ignored(self):
        pipe = self._make_pipeline(interrupt_priority=3)
        pipe.submit(CommentaryRequest(text="normal"))
        pipe.next()
        assert pipe.is_speaking

        # Low priority interrupt should NOT clear speaking
        pipe.submit(CommentaryRequest(
            text="weak interrupt",
            priority=CommentaryPriority.NORMAL,
            interrupt=True,
        ))
        assert pipe.is_speaking

    # --- Queue overflow ---

    def test_max_queue_size(self):
        pipe = self._make_pipeline(max_queue_size=3)
        for i in range(10):
            pipe.submit(CommentaryRequest(text=f"msg{i}", priority=CommentaryPriority.NORMAL))
        # Should have at most 3
        assert pipe.queue_size <= 3

    # --- TTS instruct never empty ---

    def test_tts_instruct_filled_on_submit(self):
        pipe = self._make_pipeline()
        req = CommentaryRequest(text="test", tts_instruct="")
        pipe.submit(req)
        fetched = pipe.next()
        assert fetched is not None
        assert len(fetched.tts_instruct) > 0

    # --- Status ---

    def test_status(self):
        pipe = self._make_pipeline()
        status = pipe.get_status()
        assert "queue_size" in status
        assert "is_speaking" in status
        assert "total_spoken" in status
