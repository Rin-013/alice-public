"""
EmotionTag — duck-typed adapter so emotional_tagging.tag_memory_emotion
can consume Mind's emotion output the same way it used to consume drive
snapshots.

`tag_memory_emotion(drive_snapshot=...)` reads:
    getattr(drive_snapshot, "valence", 0.0)
    getattr(drive_snapshot, "arousal", 0.0)
    getattr(drive_snapshot, k, 0.0)  for k in ("curiosity", "connection",
                                                "safety", "agency", "play")

EmotionTag exposes those attributes. Two construction paths:
  - `EmotionTag(valence=..., arousal=..., curiosity=..., ...)` — preferred,
    used when Mind emits a structured EMOTION_TAG block in post-process YAML.
  - `EmotionTag.from_avatar_intent("happy")` — fallback, used when Mind
    only emits AVATAR_INTENT (a single word). Maps the word through a
    lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# AVATAR_INTENT word -> (valence, arousal, primary_drive, primary_drive_level).
# Valence is [-1, 1], arousal is [0, 1]. Primary drive carries the
# emotional weight so `weight = max(drives.values())` in tag_memory_emotion
# returns a non-zero magnitude that reflects how strong the moment was.
_INTENT_TABLE = {
    "neutral":   (0.0,  0.2, "agency",     0.3),
    "happy":     (0.7,  0.5, "play",       0.7),
    "excited":   (0.7,  0.9, "play",       0.9),
    "sassy":     (0.4,  0.6, "agency",     0.7),
    "sad":       (-0.7, 0.3, "connection", 0.4),
    "angry":     (-0.7, 0.8, "agency",     0.6),
    "surprised": (0.1,  0.8, "curiosity",  0.8),
    "thinking":  (0.0,  0.4, "curiosity",  0.6),
    "tired":     (-0.3, 0.2, "safety",     0.3),
}

_DRIVE_KEYS = ("curiosity", "connection", "safety", "agency", "play")


@dataclass
class EmotionTag:
    """Duck-typed snapshot consumed by tag_memory_emotion."""
    valence: float = 0.0
    arousal: float = 0.0
    curiosity: float = 0.0
    connection: float = 0.0
    safety: float = 0.0
    agency: float = 0.0
    play: float = 0.0

    @classmethod
    def from_avatar_intent(cls, intent: Optional[str]) -> Optional["EmotionTag"]:
        """Map a single AVATAR_INTENT word to an EmotionTag. Unknown words
        return None so the caller can decide whether to skip tagging."""
        if not intent:
            return None
        row = _INTENT_TABLE.get(intent.strip().lower())
        if row is None:
            return None
        valence, arousal, primary_drive, level = row
        kwargs = {"valence": valence, "arousal": arousal}
        kwargs[primary_drive] = level
        return cls(**kwargs)

    @classmethod
    def from_yaml_dict(cls, data: dict) -> Optional["EmotionTag"]:
        """Parse Mind's structured EMOTION_TAG YAML block. Lenient: missing
        keys default to 0.0; non-numeric values silently dropped."""
        if not isinstance(data, dict):
            return None

        def _f(key: str) -> float:
            v = data.get(key, 0.0)
            try:
                return max(-1.0, min(1.0, float(v))) if key == "valence" \
                    else max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return 0.0

        return cls(
            valence=_f("valence"),
            arousal=_f("arousal"),
            curiosity=_f("curiosity"),
            connection=_f("connection"),
            safety=_f("safety"),
            agency=_f("agency"),
            play=_f("play"),
        )


__all__ = ["EmotionTag"]
