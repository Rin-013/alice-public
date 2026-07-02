# Copyright 2025 Rin - Alice AI System
"""
Emotional Tagging — tag memories with emotion at write time.

Honest substrate: the legacy classifier-on-text path (EmotionBERT averaged
over user + Alice text) is retired. Memories are tagged from Alice's actual
internal drive state at write time when a DriveSnapshot is provided.

Without a snapshot, returns None — memories are written with no emotion
context rather than synthesizing one from text classification.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DRIVE_KEYS = ("curiosity", "connection", "safety", "agency", "play")


def tag_memory_emotion(
    user_message: str,
    alice_response: str,
    drive_snapshot: Optional[Any] = None,
) -> Optional[Dict]:
    """
    Tag a conversation exchange with Alice's internal affect at write time.

    Reads the drive snapshot (5 scalars + 2D valence/arousal projection) and
    flattens it into the IRIS Memory.emotional_* fields.

    Returns:
        Dict with valence, arousal, weight, markers, and per-drive scalars,
        or None if no snapshot was supplied.
    """
    if drive_snapshot is None:
        return None

    try:
        valence = float(getattr(drive_snapshot, "valence", 0.0))
        arousal = float(getattr(drive_snapshot, "arousal", 0.0))
        drives = {k: float(getattr(drive_snapshot, k, 0.0)) for k in _DRIVE_KEYS}
    except (TypeError, ValueError) as e:
        logger.debug("emotional_tagging: bad drive snapshot: %s", e)
        return None

    # Weight = strongest active drive — emotional significance of the moment.
    weight = max(drives.values()) if drives else 0.0

    # Markers = top-2 drives by activation (e.g. ["curiosity", "play"]).
    markers = [name for name, _ in sorted(drives.items(), key=lambda kv: -kv[1])[:2]]

    return {
        "valence": valence,
        "arousal": arousal,
        "weight": weight,
        "markers": markers,
        "drives": drives,
    }
