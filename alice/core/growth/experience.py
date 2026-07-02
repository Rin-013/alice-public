import dataclasses
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GROWTH_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "growth"
STATE_FILE = GROWTH_DATA_DIR / "state.json"
EXPERIENCE_FILE = GROWTH_DATA_DIR / "experiences.jsonl"


@dataclasses.dataclass
class ExperienceRecord:
    """Per-turn experience record with computed metadata."""
    turn_id: str
    timestamp: str
    context: List[Dict[str, Any]]  # list of message dicts {"role": ..., "content": ...}
    response: str
    emotion: Dict[str, float]
    mood: Optional[Dict[str, Any]]  # {"mood": str, "intensity": float, "cause": str} or None
    salience: float
    surprise: float
    outcome: float
    identity_anchor: bool
    priority: float


def compute_priority(emotion: Dict[str, float], salience: float, surprise: float) -> float:
    """
    Compute sampling weight / XP value for a turn.

    0.3 * emotional_intensity + 0.25 * salience + 0.25 * surprise + 0.2 recency term.
    emotional_intensity = (abs(valence) + arousal) / 2
    """
    if not emotion:
        emo_valence, emo_arousal = 0.0, 0.0
    else:
        emo_valence = emotion.get("valence", 0.0)
        emo_arousal = emotion.get("arousal", 0.0)
    emotional_intensity = (abs(emo_valence) + emo_arousal) / 2.0
    return 0.3 * emotional_intensity + 0.25 * salience + 0.25 * surprise + 0.2


def _load_state() -> Dict[str, Any]:
    """Load growth state from STATE_FILE, creating default if missing."""
    default = {"level": 0, "xp": 0.0, "xp_next": 200.0, "total_turns": 0}
    try:
        GROWTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if STATE_FILE.exists():
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all required keys are present
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            return data
        else:
            _save_state(default)
            return default.copy()
    except Exception as e:
        logger.error(f"Failed to load growth state: {e}")
        return default.copy()


def _save_state(state: Dict[str, Any]) -> None:
    """Save growth state to STATE_FILE."""
    try:
        GROWTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save growth state: {e}")


def get_growth_state() -> Dict[str, Any]:
    """Return a copy of the current growth state (level, xp, etc.)."""
    return _load_state().copy()


def capture_experience(
    user_input: str,
    response: str,
    emotion_tag: Optional[Any] = None,
    mood_state: Optional[Any] = None,
    usefulness_scores: Optional[Dict[str, float]] = None,
    context_messages: Optional[List[Dict[str, Any]]] = None,
    memory_context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Capture a single turn's experience, append to experiences.jsonl, and update XP.

    All parameters after user_input/response can be None; sensible defaults are used.
    Never raises – errors are logged.
    """
    try:
        # ---- timestamp and unique ID ----
        now = datetime.now(timezone.utc).isoformat()
        turn_id = str(uuid.uuid4())

        # ---- emotion ----
        if emotion_tag is not None:
            emotion = {
                "valence": getattr(emotion_tag, "valence", 0.0),
                "arousal": getattr(emotion_tag, "arousal", 0.0),
                "curiosity": getattr(emotion_tag, "curiosity", 0.0),
                "connection": getattr(emotion_tag, "connection", 0.0),
                "safety": getattr(emotion_tag, "safety", 0.0),
                "agency": getattr(emotion_tag, "agency", 0.0),
                "play": getattr(emotion_tag, "play", 0.0),
            }
        else:
            emotion = {
                "valence": 0.0,
                "arousal": 0.0,
                "curiosity": 0.0,
                "connection": 0.0,
                "safety": 0.0,
                "agency": 0.0,
                "play": 0.0,
            }

        # ---- mood ----
        if mood_state is not None:
            mood = {
                "mood": getattr(mood_state, "mood", "neutral"),
                "intensity": getattr(mood_state, "intensity", 0.5),
                "cause": getattr(mood_state, "cause", "unknown"),
            }
        else:
            mood = None

        # ---- usefulness scores ----
        scores = usefulness_scores or {}
        if scores:
            mean_usefulness = sum(scores.values()) / len(scores)
        else:
            mean_usefulness = 0.5  # default

        # ---- memory context ----
        memories = []
        if memory_context and "relevant_memories" in memory_context:
            memories = memory_context["relevant_memories"]

        importances = []
        for mem in memories:
            # mem is an object with .memory.importance
            if hasattr(mem, "memory") and hasattr(mem.memory, "importance"):
                importances.append(mem.memory.importance)

        salience = sum(importances) / len(importances) if importances else 0.5
        identity_anchor = any(imp >= 0.9 for imp in importances)

        surprise = 1.0 - mean_usefulness
        outcome = mean_usefulness

        # ---- compute priority ----
        priority = compute_priority(emotion, salience, surprise)

        # ---- build record ----
        record = ExperienceRecord(
            turn_id=turn_id,
            timestamp=now,
            context=context_messages or [],
            response=response,
            emotion=emotion,
            mood=mood,
            salience=salience,
            surprise=surprise,
            outcome=outcome,
            identity_anchor=identity_anchor,
            priority=priority,
        )

        # ---- persist experience ----
        GROWTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EXPERIENCE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")

        # ---- update XP ----
        state = _load_state()
        state["xp"] = state.get("xp", 0.0) + priority
        state["total_turns"] = state.get("total_turns", 0) + 1
        _save_state(state)

    except Exception as e:
        logger.error(f"capture_experience failed: {e}", exc_info=True)
