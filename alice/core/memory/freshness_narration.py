"""
Freshness Narration
===================

Per-memory age framing in Alice's companion voice.

IRIS computes `freshness_score` and `decay_state` on every memory but that
signal never reaches Alice — it's just a scoring multiplier. This module
converts those values into short natural-language prefixes that ride along
with the memory content into Alice's prompt:

    "Rin's cat is named Pixel"              → unchanged (recent)
    "(vague memory) Rin used to game a lot" → older, less certain
    "(fuzzy — months ago) Rin had a dog"    → older still

Companion-tone wording by design. No `<system-reminder>` tags, no "verify
against current state." Alice isn't a code assistant; she's an AI companion.
The prefix reads as Alice's own subjective uncertainty, which is
what a human listener expects when someone says "I think...".
"""

from __future__ import annotations

import time
from typing import Any, Optional


def _memory_age_days(memory: Any) -> Optional[float]:
    """
    Days since the memory was last accessed (falls back to creation time).
    Returns None when no timestamp is available.
    """
    ts = getattr(memory, "last_accessed", None) or getattr(memory, "timestamp", None)
    if not ts:
        return None
    return max(0.0, (time.time() - ts) / 86400.0)


def _decay_state_value(memory: Any) -> Optional[str]:
    """
    Return decay state as a lowercase string (handles enum or raw str), or
    None when the memory carries no explicit decay state (caller should fall
    back to raw age).
    """
    state = getattr(memory, "decay_state", None)
    if state is None:
        return None
    value = getattr(state, "value", state)
    return str(value).lower()


# Thresholds chosen to match IRIS's ACT-R recency half-life (~2.8 days). WARM
# boundary sits around one half-life; COOL around 3 half-lives; COLD beyond.
# Real state values from storage/long_term.py authoritative when present.
_FRESH_DAYS_WARM = 3      # ≤3 days: no prefix
_FRESH_DAYS_COOL = 14     # 3-14 days: unchanged (still recent enough)
_FRESH_DAYS_DIM = 45      # 14-45 days: "vague"
# >45 days: "fuzzy — months ago"


def age_prefix(memory: Any) -> str:
    """
    Return a short companion-tone prefix for a memory's age, or "" when no
    framing is warranted.

    Prefers the computed `decay_state` when present (it factors in emotional
    weight, access patterns, etc.), falls back to raw day count when not.
    """
    if memory is None:
        return ""

    state = _decay_state_value(memory)

    if state == "cold":
        return "(fuzzy — months ago)"
    if state == "archived":
        return "(distant memory)"
    if state == "cool":
        return "(vague memory)"
    if state in ("active", "warm"):
        return ""

    # No explicit decay_state on this memory — use raw age.
    days = _memory_age_days(memory)
    if days is None:
        return ""
    if days <= _FRESH_DAYS_COOL:
        return ""
    if days <= _FRESH_DAYS_DIM:
        return "(vague memory)"
    return "(fuzzy — months ago)"


def age_phrase(memory: Any) -> str:
    """
    Human-readable age phrase for debug/telemetry ("3 days ago", "2 months ago").
    Not used in prompts — prompts use age_prefix(). Here for logging tools.
    """
    days = _memory_age_days(memory)
    if days is None:
        return "unknown"
    if days < 1:
        return "today"
    if days < 2:
        return "yesterday"
    if days < 14:
        return f"{int(days)} days ago"
    if days < 60:
        return f"{int(days / 7)} weeks ago"
    if days < 365:
        return f"{int(days / 30)} months ago"
    return f"{int(days / 365)} years ago"


__all__ = ["age_prefix", "age_phrase"]
