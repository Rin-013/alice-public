# Copyright 2025 Rin - Alice AI System
"""
Contradiction Detector
======================

Detects when a new FACT memory conflicts with an existing one and returns
the IDs of facts that should be superseded.

Strategy: topic-key extraction
  Each fact is parsed for a (topic, value) pair using regex patterns.
  Two facts with the same topic but different values are contradictory.

Examples of detected contradictions:
  "Rin lives in Austin."   ← superseded by →  "Rin lives in Seattle."
  "Rin works as a data scientist."  ← →  "Rin works as a software engineer."
  "Rin's girlfriend is named Amy."  ← →  "Rin's girlfriend is named Sarah."
  "Rin is 23 years old."  ← →  "Rin is 24 years old."

Examples NOT flagged (compatible facts):
  "Rin used to live in Austin."  +  "Rin lives in Seattle."   (past ≠ current)
  "Rin has a cat."               +  "Rin has a dog."          (different pets)
  "Rin works as an engineer."    +  "Rin previously worked at a startup." (past)

Design rules:
  - Conservative: only flag when confident. False negatives are fine;
    false positives (wrongly superseding a true fact) are harmful.
  - Fast: pure regex, no model calls, <1ms per check.
  - Only applies to MemoryType.FACT — not conversations.
  - Past-tense facts are never treated as current-state facts.
"""

import re
from typing import List, Optional, Tuple


# ── Past-tense guard ───────────────────────────────────────────────────────────
# If a fact matches this, it describes a past state and should never be used
# as the "current" value when checking contradictions.
_PAST_RE = re.compile(
    r'\b(used to|previously|formerly|once (was|had|lived|worked)|'
    r'has been|used to be|grew up|childhood|back when|'
    r'at the time|in the past|years ago|before (he|she|they|moving))\b',
    re.IGNORECASE,
)


def _is_past(text: str) -> bool:
    return bool(_PAST_RE.search(text))


# ── Topic key extractors ───────────────────────────────────────────────────────
# Each extractor is (topic_name, compiled_regex).
# The regex must have exactly one capture group — the "value" for that topic.
# Two facts with the same topic_name but different values are contradictions.
#
# Ordered from most specific to least specific.
_EXTRACTORS: List[Tuple[str, re.Pattern]] = [
    # Current location — "lives in X", "living in X", "based in X", "located in X"
    ("location_current",
     re.compile(
         r'\b(?:rin\s+)?(?:lives?|living|based|located)\s+in\s+([A-Za-z][^\.,;]{2,40})',
         re.IGNORECASE,
     )),

    # Job title — "works as X", "job is X", "is a(n) X" (occupation only)
    ("job_title",
     re.compile(
         r'\b(?:rin\s+)?(?:works?\s+as\s+(?:a\s+|an\s+)?|job\s+is\s+(?:a\s+|an\s+)?|is\s+(?:a|an)\s+)'
         r'([A-Za-z][^\.,;]{2,40})',
         re.IGNORECASE,
     )),

    # Employer — "works at X", "works for X"
    ("employer",
     re.compile(
         r'\b(?:rin\s+)?works?\s+(?:at|for)\s+([A-Za-z][^\.,;]{2,40})',
         re.IGNORECASE,
     )),

    # Age — "is N years old", "is N years of age"
    ("age",
     re.compile(
         r'\b(?:rin\s+)?is\s+(\d{1,3})\s+years?\s+(?:old|of\s+age)',
         re.IGNORECASE,
     )),

    # Romantic partner — "girlfriend is/named X", "dating X", etc.
    ("romantic_partner",
     re.compile(
         r'\b(?:rin\s+)?(?:girlfriend|boyfriend|partner|wife|husband|fianc\w*)'
         r'(?:\s+is(?:\s+named)?|\s+named|\s+called)?\s+([A-Za-z][^\.,;]{1,30})',
         re.IGNORECASE,
     )),

    # Single/not dating — captures the status as a fixed sentinel value
    ("romantic_status",
     re.compile(
         r'\b(?:rin\s+)?(?:is\s+)?(single|not\s+dating|no\s+(?:girlfriend|boyfriend|partner))\b',
         re.IGNORECASE,
     )),

    # Major (studying) — "studying X", "majoring in X"
    ("major",
     re.compile(
         r'\b(?:rin\s+)?(?:studying|majoring\s+in|majors?\s+in)\s+([A-Za-z][^\.,;]{2,40})',
         re.IGNORECASE,
     )),
]


def extract_topic_key(text: str) -> Optional[Tuple[str, str]]:
    """
    Extract a (topic_name, normalised_value) pair from a fact string.

    Returns None if the fact doesn't match any mutable topic pattern,
    or if the fact is past-tense (past facts are never current values).

    The returned value is lowercased and stripped for reliable comparison.
    """
    if _is_past(text):
        return None

    for topic, pattern in _EXTRACTORS:
        m = pattern.search(text)
        if m:
            value = m.group(1).strip().lower().rstrip('.,;!')
            # Reject very short or clearly incomplete captures
            if len(value) < 2:
                continue
            return (topic, value)

    return None


# ── Detector ───────────────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Detects contradicting FACT memories using topic-key comparison.

    Usage (called inside LongTermMemory.add_memory for FACT type):

        detector = ContradictionDetector()
        old_ids = detector.find_contradictions(new_content, existing_facts)
        for memory_id in old_ids:
            long_term.supersede_memory(memory_id)
    """

    def find_contradictions(
        self,
        new_content: str,
        existing_facts: list,          # List[Memory]
    ) -> List[str]:
        """
        Return IDs of existing facts contradicted by new_content.

        Args:
            new_content:    Content of the new fact being added.
            existing_facts: List of Memory objects to check against.

        Returns:
            List of memory IDs that should be superseded.
        """
        new_key = extract_topic_key(new_content)
        if new_key is None:
            return []

        new_topic, new_value = new_key
        contradicted = []

        for fact in existing_facts:
            old_key = extract_topic_key(fact.content)
            if old_key is None:
                continue
            old_topic, old_value = old_key

            if old_topic == new_topic and old_value != new_value:
                contradicted.append(fact.id)

        return contradicted


__all__ = ["ContradictionDetector", "extract_topic_key"]
