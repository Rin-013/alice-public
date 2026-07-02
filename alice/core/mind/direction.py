"""
Direction tag — steers Alice's tone at inference time.

Every training example prepends
`<alice_direction>response_type / tone / energy / length</alice_direction>`
to the user message. Live inference sent no tag, leaving Alice
out-of-distribution every turn — she collapsed to her loudest register
("sounds like a character, not a person"). This module produces a tag
per turn so inference matches training. Zero retraining.

v1 (this file): tag derives from mood state (mood.py); quiet baselines
when neutral. v2 (planned): Mind picks the direction per turn.

Vocabulary:
  response_type: quip deadpan bit roast flex tease react rant unhinged dodge hype gush
  tone:   smug sassy dramatic bored chaotic "fired up" manic smug-sweet shocked flustered
  energy: neutral hyped chaotic relaxed unhinged
  length: micro short medium long
"""

import random

try:
    from .mood import get_mood
    MOOD_AVAILABLE = True
except ImportError as e:
    MOOD_AVAILABLE = False
    print(f"   Failed in {__file__}: {e}")

# mood (AVATAR_INTENT vocabulary, emotion_tag.py) ->
#   (response_type, tone, energy at low intensity, energy at high intensity)
_MOOD_DIRECTIONS = {
    "happy":     ("tease",   "smug-sweet", "relaxed", "hyped"),
    "excited":   ("hype",    "fired up",   "hyped",   "hyped"),
    "sassy":     ("tease",   "sassy",      "neutral", "chaotic"),
    "sad":       ("deadpan", "bored",      "relaxed", "relaxed"),
    "angry":     ("roast",   "fired up",   "neutral", "chaotic"),
    "surprised": ("react",   "shocked",    "neutral", "hyped"),
    "thinking":  ("quip",    "smug",       "relaxed", "neutral"),
    "tired":     ("deadpan", "bored",      "relaxed", "relaxed"),
}

_HIGH_INTENSITY = 0.65  # same knee as render_mood_line's "really X"

# Neutral/no-mood baselines, sampled per turn for register variety.
# All from the dataset's biggest quiet pockets (deadpan/quip x
# bored/smug/sassy x neutral/relaxed dominate the tag counts).
_BASELINES = [
    ("quip",    "sassy", "neutral"),
    ("deadpan", "smug",  "relaxed"),
    ("quip",    "bored", "relaxed"),
    ("tease",   "smug",  "neutral"),
]

# micro (4211) and short (3073) dominate training; medium/long would
# truncate against chat.py's 50-token generation cap anyway.
_LENGTHS = ("micro", "short")
_LENGTH_WEIGHTS = (0.45, 0.55)


def get_direction_tag(rng: random.Random = None) -> str:
    """One direction tag for this turn, e.g. "quip / sassy / neutral / short".

    Format matches format_for_training.py exactly: four fields joined
    with " / ". Mood (with decay already applied by get_mood) picks the
    register; intensity picks the energy; neutral falls back to a
    sampled quiet baseline."""
    rng = rng or random
    state = None
    if MOOD_AVAILABLE:
        try:
            state = get_mood()
        except Exception:
            state = None

    if state is not None and state.mood in _MOOD_DIRECTIONS:
        rtype, tone, energy_low, energy_high = _MOOD_DIRECTIONS[state.mood]
        energy = energy_high if state.intensity >= _HIGH_INTENSITY else energy_low
    else:
        rtype, tone, energy = rng.choice(_BASELINES)

    length = rng.choices(_LENGTHS, weights=_LENGTH_WEIGHTS, k=1)[0]
    return f"{rtype} / {tone} / {energy} / {length}"


def wrap_user_input(user_input: str, rng: random.Random = None) -> str:
    """Prepend the direction tag in the exact training shape."""
    return f"<alice_direction>{get_direction_tag(rng)}</alice_direction>\n{user_input}"


__all__ = ["get_direction_tag", "wrap_user_input"]
