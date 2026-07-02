"""
Memory Wiring
=============

Single source of truth for how memory reaches Alice's LLM prompt.

Alice's base prompt (`base_chat_v4.txt`) has two trained memory slots:
  - {knowledge_narrative}  — stable core facts ("You know: ...")
  - {memory_narrative}     — per-turn recalled memories ("Relevant memories: ...")

Before this module, rendering was split between `chat.py` and
`script_integration.py`, each making its own formatting choices. That made it
impossible to uniformly attach per-memory metadata (age framing, modification
notices) at the point where content meets the prompt.

This module owns that attachment. Callers pass in raw memory objects;
this module returns the strings ready for template substitution.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


# Companion-tone age framing. Alice is an AI companion, not a
# code assistant — she wouldn't say "verify before asserting as fact." She'd
# say "fuzzy, haven't thought about this in months." Tone matches her voice.
#
# freshness_narration.py (P3) will own the full age→prefix table. For now
# wiring.py imports from it with a safe fallback so P7 ships even if P3 isn't
# merged yet.
try:
    from .freshness_narration import age_prefix as _age_prefix
    _HAS_FRESHNESS = True
except ImportError:
    _HAS_FRESHNESS = False

    def _age_prefix(memory: Any) -> str:  # type: ignore[misc]
        return ""


# Modification awareness (P1): if a memory's current content no longer matches
# the fingerprint stamped at creation, prepend a short in-character "wait...
# didn't this used to say X?" notice so Alice reacts to external edits.
try:
    from .modification_detector import detect_modification as _detect_mod
    from .modification_detector import format_notice as _format_mod_notice
    _HAS_MOD_DETECTION = True
except ImportError:
    _HAS_MOD_DETECTION = False

    def _detect_mod(memory: Any):  # type: ignore[misc]
        return None

    def _format_mod_notice(notice: dict) -> str:  # type: ignore[misc]
        return ""


def _extract_content(item: Any) -> Optional[str]:
    """
    Accept any of the shapes callers already hand us:
      - SearchResult (has .memory.content)
      - Memory (has .content)
      - dict (has 'content' or 'text')
      - raw string
    """
    if hasattr(item, "memory") and hasattr(item.memory, "content"):
        return item.memory.content
    if hasattr(item, "content"):
        return item.content
    if isinstance(item, dict):
        return item.get("content") or item.get("text")
    if isinstance(item, str):
        return item
    return None


def _extract_memory_obj(item: Any) -> Any:
    """Return the underlying Memory object (or the item itself if it IS one)."""
    if hasattr(item, "memory"):
        return item.memory
    return item


def _strip_conversation_echo(content: str) -> str:
    """
    CONVERSATION-type memories come back as 'User: X\\nAlice: Y' — strip
    Alice's half so the recalled string is about Rin, not about Alice's
    past responses.
    """
    if content.startswith("User:") and "\nAlice:" in content:
        return content.split("\nAlice:", 1)[0].removeprefix("User:").strip()
    return content.replace("\n", " ").strip()


def _truncate(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[: limit - 3] + "..."


def render_pinned(memory_system: Any, n: int = 10) -> str:
    """
    Render the stable pinned-context block. Plugs into {knowledge_narrative}.

    Returns "" when nothing to pin (fresh session, no facts yet).
    """
    if memory_system is None:
        return ""

    get_pinned = getattr(memory_system, "get_pinned_context", None)
    if get_pinned is not None:
        try:
            return get_pinned(n=n) or ""
        except Exception:
            return ""

    iris = getattr(memory_system, "iris", None)
    if iris is not None:
        try:
            return iris.get_pinned_context(n=n) or ""
        except Exception:
            return ""

    return ""


def render_identity(memory_system: Any, n: int = 16) -> str:
    """
    Render Alice's identity block (cartridge canon). Plugs into
    {identity_narrative}. Returns "" when no self-memories exist yet.
    """
    if memory_system is None:
        return ""
    get_identity = getattr(memory_system, "get_identity_context", None)
    if get_identity is None:
        return ""
    try:
        return get_identity(n=n) or ""
    except Exception:
        return ""


def render_recalled(memories: Iterable[Any], max_items: int = 3, char_limit: int = 180) -> str:
    """
    Render per-turn recalled memories with freshness framing. Plugs into
    {memory_narrative}.

    Companion-tone, not dev-tone. Example output:
      "Rin's cat Pixel has three legs; (fuzzy) Rin used to live in Dallas"
    """
    if not memories:
        return ""

    parts: list[str] = []
    for item in list(memories)[:max_items]:
        content = _extract_content(item)
        if not content or len(content) < 10:
            continue

        content = _strip_conversation_echo(content)
        content = _truncate(content, char_limit)

        mem_obj = _extract_memory_obj(item) if (_HAS_FRESHNESS or _HAS_MOD_DETECTION) else None

        # Modification notice takes precedence over age framing — if the
        # memory was edited, the surprise is the point, not the age.
        mod_prefix = ""
        if _HAS_MOD_DETECTION and mem_obj is not None:
            notice = _detect_mod(mem_obj)
            if notice:
                mod_prefix = _format_mod_notice(notice)

        if mod_prefix:
            content = f"{mod_prefix} {content}"
        elif _HAS_FRESHNESS and mem_obj is not None:
            prefix = _age_prefix(mem_obj)
            if prefix:
                content = f"{prefix} {content}"

        parts.append(content)

    return "; ".join(parts)


def build_memory_context_block(
    memory_system: Any,
    recalled: Optional[Iterable[Any]] = None,
    pinned_n: int = 10,
    recall_max: int = 3,
) -> dict:
    """
    One call, everything Alice's template needs for memory slots.

    Returns:
      {
        "identity_narrative":  str,  # → {identity_narrative} slot
        "knowledge_narrative": str,  # → {knowledge_narrative} slot
        "memory_narrative":    str,  # → {memory_narrative} slot
      }

    `recalled` is already-searched results (from MemoryRecallGate.fetch or
    memory_system.search_memories). Passed through `render_recalled` for
    per-memory freshness framing.
    """
    identity_raw = render_identity(memory_system)
    identity_narrative = ""
    if identity_raw:
        lines = [
            ln.lstrip("- ").strip()
            for ln in identity_raw.splitlines()
            if ln.strip() and not ln.startswith("[")
        ]
        if lines:
            identity_narrative = "; ".join(lines)

    pinned_raw = render_pinned(memory_system, n=pinned_n)

    # Reformat pinned output into "You know: ..." shape so it matches the
    # trained distribution of {knowledge_narrative}.
    pinned_narrative = ""
    if pinned_raw:
        lines = [
            ln.lstrip("- ").strip()
            for ln in pinned_raw.splitlines()
            if ln.strip() and not ln.startswith("[")
        ]
        if lines:
            pinned_narrative = "You know: " + "; ".join(lines[: pinned_n])

    recall_narrative = render_recalled(recalled or [], max_items=recall_max)

    return {
        "identity_narrative": identity_narrative,
        "knowledge_narrative": pinned_narrative,
        "memory_narrative": recall_narrative,
    }


__all__ = [
    "build_memory_context_block",
    "render_identity",
    "render_pinned",
    "render_recalled",
]
