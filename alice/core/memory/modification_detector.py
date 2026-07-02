"""
Modification Detector
=====================

Hashes memory content at write and compares at recall to catch external edits
(someone editing the SQLite directly, distiller accidentally rewriting a fact,
etc.). When a mismatch is detected we inject a short "wait, didn't this used
to say X?" notice into recall output so Alice notices in-character.

Character feature. Rin's ask: "she should know if we modify her memories
cus its hillarious". Not a security feature — there's no tamper-proofing, just
a stable fingerprint vs the current one.

Two hashes per memory:
- `first_seen_hash`: stamped once at creation, never updated.
- `content_hash`: updated on every save; tracks the current content.

Mismatch ⇒ content changed since creation ⇒ emit a modification notice.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Optional


_MAX_SNIPPET = 120


def hash_content(content: str) -> str:
    """SHA-256 of content. Stable across sessions, deterministic."""
    if content is None:
        content = ""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _truncate(text: str, limit: int = _MAX_SNIPPET) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# Companion-tone phrasing — Alice is an AI companion, not a
# compliance-log. Rotated so the notice doesn't feel canned across recalls.
_NOTICE_TEMPLATES = [
    "wait... didn't this used to say something different? feels off.",
    "huh, my memory of this feels different than I remember it.",
    "this one feels tampered with — pretty sure it didn't read like this before.",
    "something's off here — this memory doesn't feel quite like it used to.",
]


def _pick_notice(memory_id: str) -> str:
    """Deterministic per-memory-id pick so the same memory narrates the same way."""
    if not memory_id:
        return random.choice(_NOTICE_TEMPLATES)
    idx = int(hashlib.md5(memory_id.encode("utf-8", errors="replace")).hexdigest(), 16)
    return _NOTICE_TEMPLATES[idx % len(_NOTICE_TEMPLATES)]


def detect_modification(memory: Any) -> Optional[dict]:
    """
    Return a notice dict when the memory's current content diverges from its
    first-seen fingerprint. Returns None when hashes match, either hash is
    missing (legacy rows pre-migration), or content is empty.

    Notice shape:
        {
            "memory_id": str,
            "notice": str,          # companion-tone framing
            "old_hash": str,
            "new_hash": str,
            "current_snippet": str, # truncated preview of current content
        }
    """
    if memory is None:
        return None

    content = getattr(memory, "content", None)
    if not content:
        return None

    first_seen = getattr(memory, "first_seen_hash", None)
    stored = getattr(memory, "content_hash", None)
    if not first_seen:
        return None

    # Prefer live hash of content over stored content_hash — covers the case
    # where someone edited the DB directly without updating content_hash.
    live = hash_content(content)

    if live == first_seen:
        return None

    mem_id = getattr(memory, "id", "") or ""
    return {
        "memory_id": mem_id,
        "notice": _pick_notice(mem_id),
        "old_hash": first_seen,
        "new_hash": live,
        "current_snippet": _truncate(content),
    }


def format_notice(notice: dict) -> str:
    """Render a detection dict as a natural-language prefix for recall output."""
    if not notice:
        return ""
    return f"({notice.get('notice', 'this memory feels different')})"


__all__ = ["hash_content", "detect_modification", "format_notice"]
