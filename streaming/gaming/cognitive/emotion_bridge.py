"""
Emotion Bridge - Maps EmotionBERT 43-class output to gaming behavior.
=====================================================================

Connects EmotionBERT's EmotionResult to:
1. Behavior mode biases (which mode to lean toward)
2. TTS instruct strings (how Alice should sound)
3. Avatar emotion (via streaming/animation/emotion_state.py IPC)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from streaming.gaming.types import BehaviorMode
from streaming.gaming.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Optional imports — gaming works without these loaded
try:
    from alice.core.spark.fallback.emotion_bert import EmotionResult
    EMOTIONBERT_AVAILABLE = True
except ImportError as e:
    EMOTIONBERT_AVAILABLE = False
    logger.debug(f"EmotionBERT not available: {e}")

try:
    from streaming.animation.emotion_state import set_emotion as _set_avatar_emotion
    AVATAR_IPC_AVAILABLE = True
except ImportError as e:
    AVATAR_IPC_AVAILABLE = False
    logger.debug(f"Avatar IPC not available: {e}")


# ---------------------------------------------------------------------------
# Full 43-emotion mapping tables
# ---------------------------------------------------------------------------

# Mode bias weights for each emotion.
# Dict maps BehaviorMode → additive weight (higher = more likely to transition)
EMOTION_MODE_BIAS: Dict[str, Dict[BehaviorMode, float]] = {
    # --- Positive / high-energy ---
    "admiration":   {BehaviorMode.ENTERTAINER: 0.3, BehaviorMode.CHILL: 0.1},
    "amusement":    {BehaviorMode.ENTERTAINER: 0.4, BehaviorMode.CHILL: 0.2},
    "approval":     {BehaviorMode.ENTERTAINER: 0.2, BehaviorMode.CHILL: 0.1},
    "caring":       {BehaviorMode.CHILL: 0.3, BehaviorMode.ENTERTAINER: 0.1},
    "curiosity":    {BehaviorMode.CURIOUS: 0.5, BehaviorMode.ENTERTAINER: 0.1},
    "desire":       {BehaviorMode.TRYHARD: 0.2, BehaviorMode.ENTERTAINER: 0.2},
    "excitement":   {BehaviorMode.ENTERTAINER: 0.4, BehaviorMode.CLUTCH: 0.2},
    "gratitude":    {BehaviorMode.CHILL: 0.3, BehaviorMode.ENTERTAINER: 0.1},
    "joy":          {BehaviorMode.ENTERTAINER: 0.4, BehaviorMode.CHILL: 0.2},
    "love":         {BehaviorMode.CHILL: 0.3, BehaviorMode.ENTERTAINER: 0.2},
    "optimism":     {BehaviorMode.ENTERTAINER: 0.3, BehaviorMode.TRYHARD: 0.1},
    "pride":        {BehaviorMode.ENTERTAINER: 0.3, BehaviorMode.TRYHARD: 0.2},
    "relief":       {BehaviorMode.CHILL: 0.4, BehaviorMode.ENTERTAINER: 0.1},
    "happiness":    {BehaviorMode.ENTERTAINER: 0.3, BehaviorMode.CHILL: 0.2},
    "fun":          {BehaviorMode.ENTERTAINER: 0.4, BehaviorMode.CHILL: 0.2},
    "enthusiasm":   {BehaviorMode.ENTERTAINER: 0.4, BehaviorMode.TRYHARD: 0.2},
    "creativity":   {BehaviorMode.CURIOUS: 0.3, BehaviorMode.ENTERTAINER: 0.2},

    # --- Negative / high-arousal ---
    "anger":        {BehaviorMode.RAGE: 0.5, BehaviorMode.TRYHARD: 0.2},
    "annoyance":    {BehaviorMode.RAGE: 0.2, BehaviorMode.TRYHARD: 0.1},
    "disappointment": {BehaviorMode.CHILL: 0.2, BehaviorMode.RAGE: 0.1},
    "disapproval":  {BehaviorMode.RAGE: 0.1, BehaviorMode.TRYHARD: 0.1},
    "disgust":      {BehaviorMode.RAGE: 0.3, BehaviorMode.ENTERTAINER: 0.1},
    "fear":         {BehaviorMode.CLUTCH: 0.3, BehaviorMode.TRYHARD: 0.2},
    "grief":        {BehaviorMode.CHILL: 0.3, BehaviorMode.RAGE: 0.1},
    "nervousness":  {BehaviorMode.CLUTCH: 0.2, BehaviorMode.TRYHARD: 0.2},
    "remorse":      {BehaviorMode.CHILL: 0.3},
    "sadness":      {BehaviorMode.CHILL: 0.3, BehaviorMode.RAGE: 0.1},
    "hate":         {BehaviorMode.RAGE: 0.5, BehaviorMode.TRYHARD: 0.2},
    "empty":        {BehaviorMode.CHILL: 0.3},
    "worry":        {BehaviorMode.TRYHARD: 0.2, BehaviorMode.CLUTCH: 0.2},

    # --- Neutral / ambiguous ---
    "confusion":    {BehaviorMode.CURIOUS: 0.3, BehaviorMode.CHILL: 0.1},
    "surprise":     {BehaviorMode.ENTERTAINER: 0.3, BehaviorMode.CLUTCH: 0.2},
    "realization":  {BehaviorMode.CURIOUS: 0.3, BehaviorMode.ENTERTAINER: 0.2},
    "embarrassment": {BehaviorMode.CHILL: 0.2, BehaviorMode.ENTERTAINER: 0.2},
    "neutral":      {BehaviorMode.ENTERTAINER: 0.1, BehaviorMode.CHILL: 0.1},
    "boredom":      {BehaviorMode.CHILL: 0.3, BehaviorMode.CURIOUS: 0.2},

    # --- Abstract ---
    "autonomy":     {BehaviorMode.TRYHARD: 0.2, BehaviorMode.ENTERTAINER: 0.1},
    "safety":       {BehaviorMode.CHILL: 0.3},
    "understanding": {BehaviorMode.CURIOUS: 0.3, BehaviorMode.CHILL: 0.1},
    "recreation":   {BehaviorMode.ENTERTAINER: 0.3, BehaviorMode.CHILL: 0.2},
    "sense of belonging": {BehaviorMode.CHILL: 0.3, BehaviorMode.ENTERTAINER: 0.1},
    "meaning":      {BehaviorMode.CURIOUS: 0.3, BehaviorMode.CHILL: 0.1},
    "sustenance":   {BehaviorMode.CHILL: 0.2},
}

# TTS instruct string for each emotion.
# ALWAYS pass instruct — without it the model freestyles.
EMOTION_TTS_INSTRUCT: Dict[str, str] = {
    "admiration":       "Speak with genuine admiration and awe",
    "amusement":        "Speak in an amused, lightly laughing tone",
    "anger":            "Speak with sharp, frustrated anger",
    "annoyance":        "Speak with mild irritation and sarcasm",
    "approval":         "Speak warmly with approval and encouragement",
    "caring":           "Speak gently and warmly, with care",
    "confusion":        "Speak with a confused, questioning tone",
    "curiosity":        "Speak with bright curiosity and interest",
    "desire":           "Speak with eager anticipation and wanting",
    "disappointment":   "Speak with a disappointed, let-down tone",
    "disapproval":      "Speak with disapproval, slightly stern",
    "disgust":          "Speak with disgust and revulsion",
    "embarrassment":    "Speak shyly, flustered and embarrassed",
    "excitement":       "Speak with high energy and excitement",
    "fear":             "Speak with nervous fear and anxiety",
    "gratitude":        "Speak warmly with heartfelt gratitude",
    "grief":            "Speak softly, with deep sadness",
    "joy":              "Speak with bright, happy joy",
    "love":             "Speak warmly with affection and love",
    "nervousness":      "Speak nervously with slight trembling",
    "optimism":         "Speak with hopeful, upbeat optimism",
    "pride":            "Speak proudly with confidence",
    "realization":      "Speak with dawning realization, surprised insight",
    "relief":           "Speak with a relieved, exhaling tone",
    "remorse":          "Speak softly with regret and remorse",
    "sadness":          "Speak quietly with sadness",
    "surprise":         "Speak with sudden surprise",
    "neutral":          "Speak naturally with a playful tone",
    "worry":            "Speak with anxious worry",
    "happiness":        "Speak with bright happiness",
    "fun":              "Speak playfully with a fun, teasing tone",
    "hate":             "Speak with intense frustration and anger",
    "autonomy":         "Speak confidently and independently",
    "safety":           "Speak calmly and reassuringly",
    "understanding":    "Speak thoughtfully with understanding",
    "empty":            "Speak flatly, emotionally drained",
    "enthusiasm":       "Speak with bubbly enthusiasm and energy",
    "recreation":       "Speak playfully and casually",
    "sense of belonging": "Speak warmly, feeling connected",
    "meaning":          "Speak thoughtfully, contemplatively",
    "sustenance":       "Speak naturally with a playful tone",
    "creativity":       "Speak with inspired, creative energy",
    "boredom":          "Speak flatly with obvious boredom",
}

DEFAULT_TTS_INSTRUCT = "Speak naturally with a playful tone"


class EmotionBridge:
    """
    Maps EmotionBERT output to gaming-relevant signals.

    Responsibilities:
    - Convert 43-class emotion → behavior mode bias weights
    - Generate TTS instruct string
    - Update avatar emotion via IPC (emotion_state.json)
    """

    def __init__(self):
        self._last_emotion: Optional[str] = None
        self._last_tts_instruct: str = DEFAULT_TTS_INSTRUCT

    def get_mode_bias(self, emotion_primary: str) -> Dict[BehaviorMode, float]:
        """
        Get behavior mode bias weights for a given primary emotion.

        Returns dict of BehaviorMode → weight (0-1).
        Missing modes have 0 weight.
        """
        return EMOTION_MODE_BIAS.get(emotion_primary, {})

    def get_tts_instruct(self, emotion_primary: str) -> str:
        """
        Get TTS instruct string for an emotion.

        Always returns a non-empty string (falls back to default).
        """
        instruct = EMOTION_TTS_INSTRUCT.get(emotion_primary, DEFAULT_TTS_INSTRUCT)
        self._last_tts_instruct = instruct
        return instruct

    def update_avatar(self, emotion_primary: str, confidence: float = 1.0):
        """
        Push emotion to the avatar via file-based IPC.

        Calls motor.animation.emotion_state.set_emotion() which writes
        to emotion_state.json for the motion engine to read.
        """
        if AVATAR_IPC_AVAILABLE:
            try:
                _set_avatar_emotion(emotion_primary, confidence=confidence)
                self._last_emotion = emotion_primary
            except Exception as e:
                logger.error(f"Failed to update avatar emotion: {e}")
        else:
            self._last_emotion = emotion_primary

    def process_emotion_result(
        self, emotion_result: object
    ) -> Tuple[Dict[BehaviorMode, float], str]:
        """
        Process a full EmotionResult from EmotionBERT.

        Args:
            emotion_result: EmotionResult dataclass (or any object with
                            .primary and .primary_confidence attributes).

        Returns:
            (mode_bias_dict, tts_instruct_string)
        """
        primary = getattr(emotion_result, "primary", "neutral")
        confidence = getattr(emotion_result, "primary_confidence", 1.0)

        # Update avatar
        self.update_avatar(primary, confidence)

        # Get mode bias
        mode_bias = self.get_mode_bias(primary)

        # Get TTS instruct
        tts_instruct = self.get_tts_instruct(primary)

        return mode_bias, tts_instruct

    def get_status(self) -> Dict:
        return {
            "last_emotion": self._last_emotion,
            "last_tts_instruct": self._last_tts_instruct,
            "emotionbert_available": EMOTIONBERT_AVAILABLE,
            "avatar_ipc_available": AVATAR_IPC_AVAILABLE,
            "mapped_emotions": len(EMOTION_MODE_BIAS),
        }
