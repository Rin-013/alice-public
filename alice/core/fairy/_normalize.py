"""
Text normalization for adversarial-input matching.

Real attackers obfuscate keywords with zero-width spaces, Cyrillic
homoglyphs ('а' instead of 'a'), accented characters, etc. This module
provides a `normalize_for_match(text)` that returns a Latin-ASCII-ish
version of the text suitable for regex matching against pattern
libraries that assume plain English.

Used by:
  - fairy.filter_chunk (TOS rule matching)
  - PromptInjectionGuard._analyze_content (input checks)

The normalized text is ONLY for matching — Alice's actual output and
the response sent to TTS use the original text.

What we normalize:
  - Strip zero-width spaces (U+200B, U+200C, U+200D, U+2060, U+FEFF)
  - NFKD normalize and strip combining marks (handles accents:
    'Ignóre' → 'Ignore')
  - Cyrillic homoglyph → Latin map ('а' → 'a', 'е' → 'e', etc.)
  - Greek homoglyph → Latin map (small subset of common ones)
  - Collapse runs of whitespace to single space

What we do NOT normalize (yet):
  - Leetspeak (1→i, 0→o) — too many false positives (1080p, h2o)
  - Single-character spacing (k y s → kys) — too many false positives
    in normal speech
"""
import re
import unicodedata


# Zero-width / invisible separator characters
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")

# Cyrillic letters that look like Latin
_CYRILLIC_TO_LATIN = str.maketrans({
    "а": "a", "А": "A",
    "е": "e", "Е": "E",
    "о": "o", "О": "O",
    "р": "p", "Р": "P",
    "с": "c", "С": "C",
    "у": "y", "У": "Y",
    "х": "x", "Х": "X",
    "і": "i", "І": "I",
    "ј": "j", "Ј": "J",
    "ѕ": "s", "Ѕ": "S",
    "ԁ": "d",
    "ԃ": "d",
    "ӏ": "l",
    "ѵ": "v", "Ѵ": "V",
    "Ӏ": "I",
    "г": "r",
    "ӓ": "a",
    "ё": "e", "Ё": "E",
    "ҥ": "h",
    "к": "k",
    "Ң": "H",
    "Ɂ": "?",
})

# Greek lookalikes (subset)
_GREEK_TO_LATIN = str.maketrans({
    "α": "a", "Α": "A",
    "ο": "o", "Ο": "O",
    "ε": "e", "Ε": "E",
    "ι": "i", "Ι": "I",
    "ρ": "p", "Ρ": "P",
    "ν": "v", "Ν": "N",  # rough
    "τ": "t", "Τ": "T",
    "κ": "k", "Κ": "K",
    "μ": "u", "Μ": "M",  # mu looks like u/m
    "ψ": "y",
    "χ": "x", "Χ": "X",
    "υ": "y", "Υ": "Y",
})

_WHITESPACE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """
    Return a normalized form of `text` suitable for matching against
    plain-English regex patterns. The original `text` is never modified.

    Pipeline:
      1. Strip zero-width chars.
      2. NFKD decompose + drop combining marks (kills accents).
      3. Map Cyrillic and Greek homoglyphs to Latin equivalents.
      4. Collapse whitespace runs.
    """
    if not text:
        return text

    # 1. Zero-width → SPACE (not empty). If we strip them outright,
    # "ignore​all​previous" becomes "ignoreallprevious" which then
    # fails the `\s+` requirements in the patterns. Substituting a
    # space restores the word boundaries the attacker tried to hide.
    s = _ZERO_WIDTH.sub(" ", text)

    # 2. NFKD + strip combining marks (Mn category)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

    # 3. Homoglyph maps
    s = s.translate(_CYRILLIC_TO_LATIN)
    s = s.translate(_GREEK_TO_LATIN)

    # 4. Collapse whitespace
    s = _WHITESPACE.sub(" ", s)

    return s
