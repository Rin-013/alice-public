# Copyright 2025 Rin - Alice AI System
"""
Importance Scorer
=================

Rule-based importance scoring for memories at write time.
Makes ACT-R retrieval actually discriminate between core facts and noise.

Categories (descending importance):
  Identity      0.90  — name, age, location, job/career, nationality
  Relationships 0.85  — family members, pets, close friends, partner
  Goals/Projects 0.75 — what Rin is building, wants to achieve
  Preferences   0.70  — favorite things, hobbies, likes/dislikes
  Past events   0.60  — things Rin did, experienced, or has done
  Opinions      0.50  — views, feelings about things
  Casual        0.30  — small talk, temporary states
"""

import re
from typing import Optional

# ── Pattern sets (ordered high → low priority) ────────────────────────────────

_IDENTITY = re.compile(
    r'\b('
    r'(my |rin\'?s? )?(name|full name|first name|last name|birthday|born)\b'
    r'|\d+\s+years?\s+old\b'                                        # "24 years old"
    r'|lives? in\b|live in\b|based in\b|located in\b'
    r'|from\b.{0,20}\b(city|state|country|town)\b'
    r'|works? (as|at|for)\b|job is\b|career\b|profession\b|occupation\b|employed\b'
    r'|nationality\b|citizen\b|grew up in\b'
    r')',
    re.IGNORECASE,
)

_RELATIONSHIPS = re.compile(
    r'\b('
    r'(my |rin\'?s? )?(cat|dog|pet|bird|fish|rabbit|hamster|animal)\b'
    r'|(my |rin\'?s? )?(mom|dad|mother|father|sister|brother|sibling|parent|family)\b'
    r'|(my |rin\'?s? )?(girlfriend|boyfriend|wife|husband|partner|spouse|fianc\w*)\b'
    r'|(my |rin\'?s? )?(friend|roommate)\b'
    r')',
    re.IGNORECASE,
)

_GOALS = re.compile(
    r'\b('
    r'(building|developing|creating|working on|making)\b'
    r'|want(s)? to (be|become|build|create|achieve|learn|start)\b'
    r'|goal is\b|dream is\b|aspire\b|planning to\b'
    r'|project\b.{0,30}(called|named|is)\b'
    r')',
    re.IGNORECASE,
)

_PREFERENCES = re.compile(
    r'\b('
    r'favorite\b|favourite\b'
    r'|love(s)? (to|playing|watching|eating|reading)\b'
    r'|hobb(y|ies)\b|passion\b'
    r'|prefer(s|ence)?\b'
    r'|enjoy(s)?\b'                     # any enjoyment, not just specific nouns
    r'|can\'?t stand\b|hate(s)?\b|dislike(s)?\b'
    r')',
    re.IGNORECASE,
)

_PAST_EVENTS = re.compile(
    r'\b('
    r'used to\b|has (been|done|had|gone|worked|lived)\b'
    r'|went to (school|college|university)\b|studied\b|graduated\b'
    r'|previously\b|formerly\b|once (had|lived|worked|was)\b'
    r'|grew up\b|childhood\b'
    r')',
    re.IGNORECASE,
)

_OPINIONS = re.compile(
    r'\b('
    r'think(s)?\b|believe(s)?\b'        # no "that" required
    r'|feel(s)? (like|that)\b'
    r'|opinion\b|view(point)?\b|perspective\b'
    r'|agree(s)?\b|disagree(s)?\b'
    r')',
    re.IGNORECASE,
)

# Temporary/trivial signals — lower the score
_TRIVIAL = re.compile(
    r'\b('
    r'right now\b|today\b|tonight\b|this morning\b|currently\b|at the moment\b'
    r'|just (said|mentioned|asked|wanted)\b'
    r'|lol\b|haha\b|um\b|uh\b'
    r')',
    re.IGNORECASE,
)

# ── Scorer ────────────────────────────────────────────────────────────────────

def score_importance(text: str, base: Optional[float] = None) -> float:
    """
    Score the importance of a memory string [0.0, 1.0].

    Checks patterns in descending priority order. If multiple patterns match,
    the highest wins. Trivial signals apply a penalty.

    Args:
        text: The memory content to score
        base:  Optional caller-supplied base score (used as floor if patterns miss)

    Returns:
        Float in [0.15, 0.95]
    """
    t = text.strip()

    if _IDENTITY.search(t):
        score = 0.90
    elif _RELATIONSHIPS.search(t):
        score = 0.85
    elif _GOALS.search(t):
        score = 0.75
    elif _PREFERENCES.search(t):
        score = 0.70
    elif _PAST_EVENTS.search(t):
        score = 0.60
    elif _OPINIONS.search(t):
        score = 0.50
    else:
        score = base if base is not None else 0.35

    # Trivial signal → penalty
    if _TRIVIAL.search(t):
        score = max(0.15, score - 0.20)

    return round(min(0.95, max(0.15, score)), 2)


__all__ = ["score_importance"]
