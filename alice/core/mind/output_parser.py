"""
Output Parser — Extract structured data from Mind's [MIND_OUTPUT] blocks.

Mind generates free-form text that may contain a [MIND_OUTPUT] YAML block.
This parser extracts thoughts, curiosities, memory candidates, etc.
Falls back to treating the entire output as a single thought ONLY if it looks
like a real thought (short, no template tells, no <think> reasoning).
"""

import re
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# <think>...</think> blocks leak into Mind's output even with
# enable_thinking=False in the chat template (base model artifact). Strip
# them before any other parsing — they're internal reasoning, not content.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*?$", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)

# Substrings that indicate Mind is parroting its own prompt template back at
# us instead of generating real content. These came verbatim from the
# MIND_SYSTEM_PROMPT and POST_PROCESS_PROMPT in mind.py — if Mind echoes any
# of them in a "thought," drop the thought entirely.
_TEMPLATE_TELLS = (
    "only include memory_candidates",
    "only include iris_queries",
    "avatar_intent options",
    "reflection on how that response went",
    "what happened worth remembering",
    "relevant memory search for next turn",
    "first thought here",
    "second thought here",
    "something i'm wondering about",
    "first person as alice",
    "alice's inner mind",
    "[mind_output]",  # the literal block markers as plain text
    "[/mind_output]",
)

# A real Mind thought is 1-2 sentences. Anything dramatically longer is
# almost always a runaway reasoning dump or template parroting.
_MAX_THOUGHT_CHARS = 280


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (matched OR unclosed)."""
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub(" ", text)
    # Unclosed: kill from <think> to EOS, or from start to </think>
    text = _THINK_OPEN_RE.sub(" ", text)
    text = _THINK_CLOSE_RE.sub(" ", text)
    return text.strip()


def _looks_like_template_tell(s: str) -> bool:
    """True iff the string contains an obvious template-parroting marker."""
    if not s:
        return False
    low = s.lower()
    return any(tell in low for tell in _TEMPLATE_TELLS)


def _filter_thought(s: str) -> Optional[str]:
    """
    Sanitize and validate a single thought. Returns the cleaned string if
    it survives, None if it should be dropped.
    """
    if not s:
        return None
    s = _strip_think_blocks(str(s)).strip()
    if not s:
        return None
    if len(s) > _MAX_THOUGHT_CHARS:
        return None  # Runaway dump
    if _looks_like_template_tell(s):
        return None  # Template parroting
    return s


@dataclass
class MindOutput:
    """Parsed output from Mind's generation."""
    thoughts: List[str] = field(default_factory=list)
    curiosities: List[str] = field(default_factory=list)
    memory_candidates: List[Dict[str, Any]] = field(default_factory=list)
    iris_queries: List[str] = field(default_factory=list)
    goal_suggestions: List[str] = field(default_factory=list)
    avatar_intent: Optional[str] = None
    # Structured emotion tag — feeds memory.tag_memory_emotion when Mind
    # bothers to score the turn. Falls back to AVATAR_INTENT lookup when
    # this field is missing. dict shape: valence, arousal, curiosity,
    # connection, safety, agency, play (all floats; missing keys → 0.0).
    emotion_tag: Optional[Dict[str, Any]] = None
    # Short phrase naming WHY Alice feels the AVATAR_INTENT emotion.
    # Feeds mood.py (persistent mood-with-cause).
    mood_cause: Optional[str] = None
    raw: str = ""


def parse_mind_output(raw_text: str) -> MindOutput:
    """
    Parse Mind's raw text into structured MindOutput.

    Looks for [MIND_OUTPUT]...[/MIND_OUTPUT] block. If not found, falls back
    to treating the text as a single thought ONLY if it survives sanitization
    (think-block stripping, length cap, no template-parroting tells).
    """
    result = MindOutput(raw=raw_text)

    if not raw_text or not raw_text.strip():
        return result

    # Strip <think> reasoning blocks before any other parsing — they're
    # never content, regardless of YAML vs plaintext path.
    cleaned_raw = _strip_think_blocks(raw_text)

    # Try to extract YAML block
    match = re.search(
        r'\[MIND_OUTPUT\]\s*\n(.*?)\n\s*\[/MIND_OUTPUT\]',
        cleaned_raw,
        re.DOTALL
    )

    if match:
        try:
            data = yaml.safe_load(match.group(1))
            if isinstance(data, dict):
                # Filter thoughts and curiosities through _filter_thought —
                # catches template-tell parroting even inside valid YAML.
                raw_thoughts = _as_list(data.get('THOUGHTS', []))
                result.thoughts = [t for t in (_filter_thought(x) for x in raw_thoughts) if t]
                raw_curiosities = _as_list(data.get('CURIOSITIES', []))
                result.curiosities = [c for c in (_filter_thought(x) for x in raw_curiosities) if c]
                result.memory_candidates = data.get('MEMORY_CANDIDATES', []) or []
                result.iris_queries = _as_list(data.get('IRIS_QUERIES', []))
                result.goal_suggestions = _as_list(data.get('GOAL_SUGGESTIONS', []))
                result.avatar_intent = data.get('AVATAR_INTENT')
                cause = data.get('MOOD_CAUSE')
                if isinstance(cause, str) and cause.strip():
                    result.mood_cause = cause.strip()[:120]
                emotion_tag = data.get('EMOTION_TAG')
                if isinstance(emotion_tag, dict):
                    result.emotion_tag = emotion_tag
                return result
        except yaml.YAMLError:
            pass  # Fall through to plain text

    # No YAML block — treat whole text as a single thought, but ONLY if it
    # passes sanitization. Mind frequently produces long reasoning dumps or
    # template-parroting when it fails to format the YAML; those should NOT
    # become "thoughts" injected into Alice's prompt.
    cleaned = re.sub(r'\[/?MIND_OUTPUT\]', '', cleaned_raw).strip()
    survived = _filter_thought(cleaned)
    if survived:
        result.thoughts = [survived]

    return result


def _as_list(val) -> list:
    """Ensure value is a list of strings."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return []
