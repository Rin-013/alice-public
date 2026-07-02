"""
Proposals Buffer — Staging area for Mind's thoughts.

Thoughts aren't real until Alice sees them. This buffer stores Mind's proposals
with timestamps, lets the context builder grab recent ones, and tracks which
proposals Alice actually referenced in her response.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Proposal:
    """A single thought proposal from Mind."""
    content: str
    timestamp: float
    source: str = "mind"          # mind | voice:<name>
    used: Optional[bool] = None   # None=unseen, True=referenced, False=ignored
    similarity: float = 0.0       # How much Alice's response matched this


@dataclass
class MemoryHint:
    """
    A memory snippet Mind pre-fetched from IRIS.

    Unlike a Proposal (which is Mind's own thought), a hint is a concrete fact
    pulled from memory in response to one of Mind's IRIS_QUERIES. Kept separate
    so Alice's context renders it as "I remember: …" rather than "I'm thinking
    about: …" — different slot, different voice.
    """
    content: str           # the memory text
    query: str             # what Mind was curious about
    timestamp: float
    memory_id: str = ""    # originating memory id (for dedup / debug)


class ProposalsBuffer:
    """Thread-safe buffer of Mind's proposals for Alice's context."""

    def __init__(self, max_proposals: int = 20, ttl_seconds: float = 120.0):
        self._proposals: List[Proposal] = []
        self._memory_hints: List[MemoryHint] = []
        self._lock = threading.Lock()
        self.max_proposals = max_proposals
        self.ttl_seconds = ttl_seconds

        # Stats
        self.total_added = 0
        self.total_used = 0
        self.total_ignored = 0
        self.total_hints_added = 0

    def add(self, content: str, source: str = "mind"):
        """Add a proposal from Mind (thread-safe)."""
        with self._lock:
            self._proposals.append(Proposal(
                content=content,
                timestamp=time.time(),
                source=source,
            ))
            self.total_added += 1
            # Evict oldest if over limit
            if len(self._proposals) > self.max_proposals:
                self._proposals.pop(0)

    def get_recent(self, n: int = 5, max_age: float = None) -> List[Proposal]:
        """Get the N most recent proposals, optionally filtered by age."""
        max_age = max_age or self.ttl_seconds
        cutoff = time.time() - max_age
        with self._lock:
            recent = [p for p in self._proposals if p.timestamp > cutoff]
            return recent[-n:]

    def get_context_string(self, n: int = 5) -> str:
        """Format recent proposals as a string for Alice's system prompt."""
        recent = self.get_recent(n)
        if not recent:
            return ""
        lines = [f"- {p.content}" for p in recent]
        return "Your recent thoughts:\n" + "\n".join(lines)

    @staticmethod
    def _normalize_words(text: str) -> set:
        """Lowercase, strip punctuation, split into word set."""
        import re
        return set(re.findall(r"[a-z']+", text.lower()))

    def mark_usage(self, alice_response: str):
        """
        After Alice responds, check which proposals she referenced.
        Uses simple substring/keyword overlap — good enough for tracking signal.
        """
        if not alice_response:
            return

        response_words = self._normalize_words(alice_response)

        stop = {'the', 'a', 'an', 'is', 'was', 'are', 'i', 'you', 'and', 'or',
                'but', 'to', 'of', 'in', 'it', 'that', 'this', 'my', 'me', 'we',
                'so', 'do', 'if', 'at', 'be', 'on', 'no', 'not', 'what', 'how'}

        with self._lock:
            for p in self._proposals:
                if p.used is not None:
                    continue  # Already scored

                proposal_words = self._normalize_words(p.content)
                if len(proposal_words) < 2:
                    p.used = False
                    continue

                content_words = proposal_words - stop
                if not content_words:
                    p.used = False
                    continue

                overlap = content_words & response_words
                p.similarity = len(overlap) / len(content_words)
                p.used = p.similarity > 0.3  # 30%+ word overlap = "used"

                if p.used:
                    self.total_used += 1
                else:
                    self.total_ignored += 1

    def clear_old(self):
        """Remove proposals and memory hints older than TTL."""
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            self._proposals = [p for p in self._proposals if p.timestamp > cutoff]
            self._memory_hints = [h for h in self._memory_hints if h.timestamp > cutoff]

    # ---- Memory hints (Mind-initiated IRIS lookups) ----

    def add_memory_hint(self, content: str, query: str = "", memory_id: str = ""):
        """Stash a memory snippet Mind pulled from IRIS (thread-safe)."""
        if not content:
            return
        with self._lock:
            # Dedup by memory_id (or content when no id) so repeated queries
            # don't pile the same fact into context.
            key = memory_id or content
            existing = {(h.memory_id or h.content) for h in self._memory_hints}
            if key in existing:
                return
            self._memory_hints.append(MemoryHint(
                content=content,
                query=query,
                timestamp=time.time(),
                memory_id=memory_id,
            ))
            self.total_hints_added += 1
            if len(self._memory_hints) > self.max_proposals:
                self._memory_hints.pop(0)

    def get_memory_hints(self, n: int = 3, max_age: float = None) -> List[MemoryHint]:
        """Return the N most recent memory hints within TTL."""
        max_age = max_age or self.ttl_seconds
        cutoff = time.time() - max_age
        with self._lock:
            fresh = [h for h in self._memory_hints if h.timestamp > cutoff]
            return fresh[-n:]

    def get_memory_hints_string(self, n: int = 3) -> str:
        """Format hints as a context block for Alice's system prompt."""
        hints = self.get_memory_hints(n)
        if not hints:
            return ""
        lines = [f"- {h.content}" for h in hints]
        return "You recalled:\n" + "\n".join(lines)

    def consume_memory_hints(self, n: int = 3) -> List[MemoryHint]:
        """
        Return hints AND drop them from the buffer. Use when hints should
        appear exactly once (in the next turn's context) rather than lingering.
        """
        with self._lock:
            hints = list(self._memory_hints[-n:])
            self._memory_hints = self._memory_hints[:-n] if n < len(self._memory_hints) else []
            return hints

    def get_stats(self) -> dict:
        return {
            "total_added": self.total_added,
            "total_used": self.total_used,
            "total_ignored": self.total_ignored,
            "total_hints_added": self.total_hints_added,
            "current_count": len(self._proposals),
            "hint_count": len(self._memory_hints),
            "usage_rate": self.total_used / max(1, self.total_used + self.total_ignored),
        }
