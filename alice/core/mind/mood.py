"""
Mood with a cause — Alice's persistent emotional state.

The avatar emotion (emotion_state.json) is per-turn decoration: it drives
animation and TTS tone, then gets overwritten. This module is the part that
makes emotion read as a *person*: one mood at a time, with the reason
attached, decaying over minutes instead of resetting every turn. Get teased,
stay a little salty for ten minutes, and be able to say why.

Flow:
  Mind post-process  -> update_mood(avatar_intent, intensity, cause)
  Alice's prompt     <- render_mood_line()  ("Right now you're sassy — Rin
                        called your take mid.")  via {mood_narrative}

Decay happens at read time (no background thread): intensity halves every
MOOD_HALF_LIFE_S seconds; below MOOD_FLOOR the mood expires to nothing.
A 'neutral' update clears the mood — Mind saying "she's fine now" wins.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Lives next to the memory DB, not in motor/ — mood is mind-state, the
# avatar file stays the animation protocol.
MOOD_STATE_FILE = str(Path(__file__).resolve().parents[2] / "data" / "mood_state.json")

MOOD_HALF_LIFE_S = 480.0   # ~8 min: teasing wears off over a segment, not a turn
MOOD_FLOOR = 0.15          # below this, the mood has faded to nothing


@dataclass
class MoodState:
    mood: str = "neutral"
    intensity: float = 0.0     # value at the moment it was set
    cause: str = ""
    timestamp: float = 0.0


def update_mood(mood: str, intensity: float = 0.6, cause: str = "") -> None:
    """Write-through mood update (Mind calls this post-turn).

    'neutral' clears the mood entirely — no lingering ghost state."""
    mood = (mood or "").strip().lower()
    if not mood:
        return
    if mood == "neutral":
        state = MoodState()
    else:
        state = MoodState(
            mood=mood,
            intensity=max(0.0, min(1.0, float(intensity))),
            cause=(cause or "").strip()[:120],
            timestamp=time.time(),
        )
    try:
        os.makedirs(os.path.dirname(MOOD_STATE_FILE), exist_ok=True)
        with open(MOOD_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, indent=2)
    except OSError:
        pass


def get_mood() -> Optional[MoodState]:
    """Current mood with decay applied. None when neutral/expired/no file."""
    try:
        with open(MOOD_STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        state = MoodState(**{k: data[k] for k in ("mood", "intensity", "cause", "timestamp")})
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if state.mood in ("", "neutral") or state.intensity <= 0:
        return None
    elapsed = max(0.0, time.time() - state.timestamp)
    state.intensity = state.intensity * 0.5 ** (elapsed / MOOD_HALF_LIFE_S)
    if state.intensity < MOOD_FLOOR:
        return None
    return state


def render_mood_line() -> str:
    """One natural-language line for Alice's prompt, or "" when neutral.

    The cause traveling with the mood is the whole point — it's what lets
    her stay salty about a specific thing instead of being vaguely moody."""
    state = get_mood()
    if state is None:
        return ""
    if state.intensity >= 0.65:
        feel = f"Right now you're really {state.mood}"
    elif state.intensity >= 0.35:
        feel = f"Right now you're {state.mood}"
    else:
        feel = f"You're still a little {state.mood}"
    if state.cause:
        return f"{feel} — {state.cause}."
    return f"{feel}."
