"""
ScriptIntegration — minimal prompt builder.

Loads `base_chat.txt`, fills 5 narrative slots from Mind proposals + IRIS
memory, returns the rendered prompt to chat.py via `context_data["script_prompt"]`.

The full StateStorage/ScriptEngine/StateTranslator stack (1402 + 370 + 392 LOC)
this replaced has been archived to master_archive/simplify_2026_05_05/scripting/.
With Mind owning emotion and IRIS owning memory, the only thing scripting
still needs to do is template substitution.
"""

import re
from pathlib import Path
from typing import Any, Dict


_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "base_chat.txt"
_TEMPLATE_CACHE: str | None = None

# Multi-blank-line collapse pattern. Empty template variables ({foo}) sitting
# on their own line between \n\n separators produce 3-4 consecutive newlines
# after substitution. Collapse anything >2 down to one blank line.
_BLANK_RUN_RE = re.compile(r"\n{3,}")

# Stripped from narrative entries before display. wiring.py's pinned-context
# filter at line 198 tries to drop `[`-prefixed lines but checks BEFORE its
# lstrip("- "), so "- [FACT] X" slips through. Fixing it consumer-side keeps
# the memory subsystem untouched.
_FACT_PREFIX = "[FACT] "


def _load_template() -> str:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return _TEMPLATE_CACHE


def _user_name(user_id: str, memory_system: Any) -> str:
    """Display name from IRIS pinned facts, else titlecased user_id."""
    if memory_system is not None:
        get_pinned = getattr(memory_system, "get_pinned_context", None)
        if get_pinned is not None:
            try:
                pinned = get_pinned(n=3) or ""
            except Exception:
                pinned = ""
            for line in pinned.splitlines():
                low = line.lower()
                if "name is " in low or "called " in low:
                    return line.split(":", 1)[-1].strip().lstrip("- ")
    return user_id.title()


def _recent_thought(mind: Any) -> str:
    if mind is None:
        return ""
    try:
        recent = mind.proposals.get_recent(n=1)
    except Exception:
        return ""
    if not recent:
        return ""
    return f"You just thought: {recent[0].content}"


def _mood_line() -> str:
    """Persistent mood-with-cause from Mind (decays over minutes). Empty
    string when she's neutral — the template line collapses."""
    try:
        from alice.core.mind.mood import render_mood_line
        return render_mood_line()
    except Exception:
        return ""


def _clean_narrative(narrative: str, prefix: str = "") -> str:
    """
    Dedupe entries (case-insensitive), strip [FACT] markers, drop empty.
    Input: "Prefix: A; B; A; [FACT] C" → "Prefix: A; B; C"
    If `prefix` is non-empty and the narrative starts with it, the prefix is
    preserved on the output. Empty cleaned body → empty string (lets the
    template caller decide whether to keep surrounding scaffolding).
    """
    if not narrative:
        return ""
    body = narrative
    if prefix and body.startswith(prefix):
        body = body[len(prefix):]
    parts = [p.strip() for p in body.split(";") if p.strip()]
    cleaned: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p.startswith(_FACT_PREFIX):
            p = p[len(_FACT_PREFIX):].strip()
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)
    if not cleaned:
        return ""
    return prefix + "; ".join(cleaned)


class ScriptIntegration:
    """Drop-in replacement for the old ScriptIntegration. Same public surface
    chat.py uses (`process_pre_llm` setting `context_data['script_prompt']`),
    no SQLite, no narrative variables Alice never sees."""

    def __init__(self, _hive_unused=None):
        self.initialized = True

    async def process_pre_llm(
        self,
        user_input: str,
        user_id: str,
        context_data: Dict[str, Any],
    ) -> None:
        from alice.core.system import get_registry
        registry = get_registry()
        mind = registry.get("mind") if registry and registry.has("mind") else None
        memory_system = registry.get("memory") if registry and registry.has("memory") else None

        memory_context = context_data.get("memory_context") or {}
        recalled = memory_context.get("relevant_memories") or []
        if memory_context.get("knowledge"):
            recalled = list(recalled) + list(memory_context["knowledge"])

        from alice.core.memory.wiring import build_memory_context_block
        block = build_memory_context_block(memory_system, recalled=recalled, pinned_n=5)

        filled = _load_template().format_map(_SafeDict(
            recent_thought_narrative=_recent_thought(mind),
            mood_narrative=_mood_line(),
            identity_narrative=_clean_narrative(block["identity_narrative"], prefix="About you: "),
            knowledge_narrative=_clean_narrative(block["knowledge_narrative"], prefix="You know: "),
            memory_narrative=_clean_narrative(block["memory_narrative"]),
            past_learnings_narrative="",
            user_name=_user_name(user_id, memory_system),
            relationship_description="",
        ))
        # Collapse blank-line scars from any empty template variables.
        filled = _BLANK_RUN_RE.sub("\n\n", filled)
        context_data["script_prompt"] = filled


class _SafeDict(dict):
    """format_map() helper — leaves unknown {placeholders} intact instead
    of raising KeyError, so a stale template slot won't crash the turn."""
    def __missing__(self, key: str) -> str:
        return ""


__all__ = ["ScriptIntegration"]
