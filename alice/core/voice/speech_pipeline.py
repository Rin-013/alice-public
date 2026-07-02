"""
SpeechPipeline — fire-and-forget speech queue layered over TTSWorker.

Foundation for clause-streaming TTS. Today chat.py calls `submit()` once
per turn with the full LLM response. Tomorrow (Path B) chat.py will call
`submit()` once per clause boundary as the LLM streams tokens — same API,
no change to this class needed.

Why this exists:
- TTSWorker.speak() blocks the caller until the subprocess finishes
  generation. The subprocess accepts only one `speak` command at a time.
- chat.py wants to keep doing real work (tool dispatch, memory save,
  history maintenance) WHILE TTS is generating, then sync once before
  the next turn.
- SpeechPipeline owns a worker thread that feeds the subprocess
  serially. submit() returns immediately; wait_*() blocks at the end of
  the turn.

Threading model:
- Caller thread (chat.py): calls submit() and wait_*().
- Worker thread (here): consumes inbox, calls tts.speak().
- TTSWorker reader thread: drains subprocess JSON status replies.
- Subprocess playback thread: drains audio queue to sounddevice.

The subprocess's own audio queue handles back-to-back gapless playback,
so multiple submit() calls in a row produce continuous audio (modulo
the per-call prosody seam — that's a model limitation Path B will tune).
"""

import logging
import queue
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SpeechPipeline:
    def __init__(self, tts_worker):
        if tts_worker is None:
            raise ValueError("SpeechPipeline requires a TTSWorker")
        self._tts = tts_worker

        self._inbox: "queue.Queue[Optional[Tuple[str, str]]]" = queue.Queue()
        # `_busy` = items currently in the inbox + the one being spoken.
        # wait_generation_done() blocks until it hits 0. We use a Condition
        # so the wait is interrupt-aware instead of polling.
        self._busy = 0
        self._idle_cv = threading.Condition()
        self._shutdown = False

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="SpeechPipeline"
        )
        self._worker.start()

    def submit(self, text: str, emotion: str = "neutral") -> None:
        """Enqueue text for speech. Returns immediately.

        Empty/whitespace text is silently dropped. Multiple calls before
        the previous speak completes are serialized FIFO by the worker.
        """
        if not text or not text.strip():
            return
        with self._idle_cv:
            self._busy += 1
        self._inbox.put((text, emotion))

    def wait_generation_done(self, timeout: float = 180.0) -> bool:
        """Block until every queued speak() has finished generating in the
        subprocess. Playback may still be in progress on the subprocess
        side — see wait_playback_done() if you need that too.

        Returns True on drained, False on timeout.
        """
        deadline = time.monotonic() + timeout
        with self._idle_cv:
            while self._busy > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle_cv.wait(timeout=remaining)
        return True

    def wait_playback_done(self, timeout: float = 180.0) -> bool:
        """Block until generation completes AND the subprocess playback
        queue drains to silence. Use at end-of-turn so the next turn
        doesn't start mid-utterance."""
        if not self.wait_generation_done(timeout=timeout):
            return False
        try:
            self._tts.drain(timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"drain failed: {e}")
            return False

    def is_busy(self) -> bool:
        """True iff something is queued or currently speaking. Cheap check
        for the caller's diagnostics — not a synchronization primitive."""
        with self._idle_cv:
            return self._busy > 0

    def shutdown(self) -> None:
        self._shutdown = True
        self._inbox.put(None)
        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while not self._shutdown:
            item = self._inbox.get()
            if item is None:
                break
            text, emotion = item
            try:
                self._tts.speak(text, emotion)
            except Exception as e:
                logger.error(f"speak failed: {e}")
            finally:
                with self._idle_cv:
                    self._busy -= 1
                    if self._busy <= 0:
                        self._idle_cv.notify_all()
