"""
ClauseChunker — split a streaming text feed at clause boundaries.

Path B's text-side splitter. Sits in chat.py's LLM-token loop, accumulates
visible text, emits chunks at natural prosody boundaries (sentence-end,
clause break) according to a chunk-length schedule. Each emitted chunk
becomes one `SpeechPipeline.submit()` call, so Alice starts talking
mid-generation instead of waiting for the LLM to finish.

Schedule philosophy (mirrors ElevenLabs `chunk_length_schedule`):
- First chunk SMALL for low time-to-first-audio.
- Subsequent chunks LARGER so the talker has enough text to commit to
  good prosody — the TTS model adds end-of-utterance cadence per chunk, so
  fewer, longer chunks = fewer audible seams.

Boundary preference (high to low):
- HARD: sentence-end punctuation (. ! ?) followed by whitespace or EOL.
  This is the only break that sounds natural. Required at normal length.
- SOFT: ; : — – (and , as fallback). Only used past 1.5x the schedule
  threshold — model needs the larger window to handle the soft break.
- FORCED: at max_chars, cut on whitespace. Audible seam, last resort.

We deliberately do NOT break on raw "." without a following space —
catches "3.14", "Mr.", "U.S." cleanly without a dictionary.
"""

from dataclasses import dataclass
from typing import List, Tuple


HARD_TERMS: Tuple[str, ...] = (".", "!", "?")
SOFT_TERMS: Tuple[str, ...] = (";", ":", "—", "–")
COMMA = ","


@dataclass
class ClauseChunker:
    schedule: Tuple[int, ...] = (60, 140, 240, 240)
    max_chars: int = 400
    slack: int = 80

    def __post_init__(self) -> None:
        self._buffer: str = ""
        self._chunk_idx: int = 0

    # ---- public API ---------------------------------------------------

    def feed(self, text: str) -> List[str]:
        """Append `text`. Return list of newly-ready chunks (may be empty).

        Drains repeatedly so a single large input can emit multiple chunks.
        """
        if not text:
            return []
        self._buffer += text
        out: List[str] = []
        while True:
            chunk = self._try_extract()
            if chunk is None:
                break
            out.append(chunk)
            self._chunk_idx += 1
        return out

    def flush(self) -> str:
        """End-of-stream. Return whatever's left as one final chunk (or '').

        Used in chat.py after the LLM loop finishes. Anything still buffered
        — including a sentence with no terminator at all — gets spoken.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        if remaining:
            self._chunk_idx += 1
        return remaining

    def reset(self) -> None:
        self._buffer = ""
        self._chunk_idx = 0

    @property
    def buffered_chars(self) -> int:
        return len(self._buffer)

    @property
    def chunks_emitted(self) -> int:
        return self._chunk_idx

    # ---- internals ----------------------------------------------------

    def _threshold(self) -> int:
        return self.schedule[min(self._chunk_idx, len(self.schedule) - 1)]

    def _is_hard_break(self, i: int) -> bool:
        """Position i is a sentence-end iff it's a hard char AND the next
        char is whitespace (or i is the buffer end). Without the
        whitespace check, "3.14" and "U.S." get incorrectly split."""
        ch = self._buffer[i]
        if ch not in HARD_TERMS:
            return False
        nxt = i + 1
        if nxt >= len(self._buffer):
            # End of buffer mid-stream — wait for the next token to decide.
            # flush() handles this case explicitly.
            return False
        return self._buffer[nxt].isspace()

    def _is_soft_break(self, i: int) -> bool:
        """Soft break: semicolons/colons/em-dashes (always), or commas
        (caller's choice). Requires following whitespace to avoid splitting
        decimal-like patterns."""
        ch = self._buffer[i]
        if ch not in SOFT_TERMS and ch != COMMA:
            return False
        nxt = i + 1
        if nxt >= len(self._buffer):
            return False
        return self._buffer[nxt].isspace()

    def _consume_terminator_run(self, i: int) -> int:
        """If `i` is the start of a multi-char terminator run like "..."
        or "?!", advance to the last terminator in the run."""
        while (
            i + 1 < len(self._buffer)
            and self._buffer[i + 1] in HARD_TERMS
        ):
            i += 1
        return i

    def _try_extract(self) -> "str | None":
        """Return one chunk or None."""
        threshold = self._threshold()
        if len(self._buffer) < threshold:
            return None

        # 1. Hard terminator in [threshold-1, threshold+slack] —
        #    scan right-to-left so the LATEST clean boundary wins
        #    (gives the talker a longer first chunk if available).
        window_start = max(0, threshold - 1)
        window_end = min(len(self._buffer), threshold + self.slack)
        for i in range(window_end - 1, window_start - 1, -1):
            if self._is_hard_break(i):
                i = self._consume_terminator_run(i)
                return self._cut(i + 1)

        # 2. Buffer is past 1.5x threshold — soft break OK.
        soft_threshold = int(threshold * 1.5)
        if len(self._buffer) >= soft_threshold:
            for i in range(len(self._buffer) - 1, soft_threshold - 2, -1):
                if self._is_hard_break(i) or self._is_soft_break(i):
                    return self._cut(i + 1)

        # 3. Forced cut at max_chars — last whitespace before max_chars.
        if len(self._buffer) >= self.max_chars:
            ws = self._buffer.rfind(" ", 0, self.max_chars)
            cut = ws if ws > 0 else self.max_chars
            return self._cut(cut)

        return None

    def _cut(self, end: int) -> str:
        chunk = self._buffer[:end].strip()
        # Trim leading whitespace from what we keep so next chunk
        # doesn't start with the boundary's space.
        self._buffer = self._buffer[end:].lstrip()
        return chunk
