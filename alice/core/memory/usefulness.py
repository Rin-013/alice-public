"""
Memory Usefulness Signal
========================

Measures whether a retrieved memory was actually *useful* to the response, not
just whether its words happened to overlap with Alice's reply. Replaces the
bag-of-words heuristic in chat.py's IRIS-feedback block.

Two levels of judgement:

  1. **Cosine (cheap, always-on)** — MiniLM embedding of (memory, response)
     via the shared embedding singleton. Threshold 0.55 = "used".

  2. **LLM judge (optional, ambiguous band only)** — when cosine sits in the
     mushy middle [0.35, 0.65] we can fire a Ghost yes/partial/no prompt on
     Mind's idle cycle. Flag-gated via `ALICE_IRIS_LLM_USEFULNESS=1` and
     only invoked when a Mind handle is passed in.

Both functions return a score in [0.0, 1.0] or None on failure. Fail-open
everywhere — Alice never blocks on a usefulness judgement, and downstream
EMA updates simply skip None values.

Guardrails:
  - `len(response) > 30` required. Short acks ("sure", "yeah") produce
    misleadingly high cosines against anything; we refuse to score them.
  - Empty memory content → None.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Cosine thresholds — tuned against MiniLM embedding distributions observed on
# actual Alice turns. Above HIGH = definitely useful; below LOW = not useful;
# the band between is where the LLM judge adds value.
COSINE_USED_THRESHOLD = 0.55
COSINE_AMBIGUOUS_LOW = 0.35
COSINE_AMBIGUOUS_HIGH = 0.65

# Below this many characters, Alice's response is too short to score reliably.
_MIN_RESPONSE_CHARS = 30

_LLM_FLAG = "ALICE_IRIS_LLM_USEFULNESS"


def _cosine(a, b) -> float:
    """Cosine similarity between two 1-D numpy vectors. Safe against zero-norms."""
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_usefulness(memory_content: str, response: str) -> Optional[float]:
    """
    Cosine-based usefulness score. Returns a value in [0.0, 1.0] (rescaled
    from the raw [-1, 1] cosine so downstream EMA stays in-range) or None
    if the inputs don't meet the minimum bar.

    None is meaningfully different from 0.0: None = "can't judge, don't
    update"; 0.0 = "judged and not useful".
    """
    if not memory_content or not memory_content.strip():
        return None
    if not response or len(response.strip()) < _MIN_RESPONSE_CHARS:
        return None

    try:
        from ..utils.embedding_utils import get_shared_embedding_model
        model = get_shared_embedding_model()
        vecs = model.encode([memory_content.strip(), response.strip()],
                            show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        logger.debug(f"usefulness: embed failed: {e}")
        return None

    try:
        cos = _cosine(vecs[0], vecs[1])
    except Exception as e:
        logger.debug(f"usefulness: cosine failed: {e}")
        return None

    # Rescale [-1, 1] → [0, 1]. Most Alice/memory pairs sit in [0.1, 0.8] so
    # the rescale preserves the dynamic range our thresholds were chosen for.
    score = max(0.0, min(1.0, (cos + 1.0) / 2.0))
    return round(score, 4)


def is_ambiguous(score: Optional[float]) -> bool:
    """True when cosine is neither clearly useful nor clearly not."""
    if score is None:
        return False
    return COSINE_AMBIGUOUS_LOW <= score <= COSINE_AMBIGUOUS_HIGH


def score_usefulness_llm(memory_content: str,
                         response: str,
                         mind: Any) -> Optional[float]:
    """
    Ask Ghost to arbitrate ambiguous cosines. Only runs when:
      - `ALICE_IRIS_LLM_USEFULNESS=1` is set
      - a Mind handle with a callable `_generate` is provided
      - the cosine score was in the ambiguous band

    Returns 0.0 / 0.5 / 1.0 for no / partial / yes, or None on any failure.
    Callers are expected to schedule this *off the hot path* (e.g. on Mind's
    idle cycle post-TTS), not during response generation.
    """
    if os.environ.get(_LLM_FLAG, "0").strip().lower() in ("0", "false", "no", "off"):
        return None
    if mind is None:
        return None

    generate = getattr(mind, "_generate", None)
    if not callable(generate):
        return None

    prompt = (
        "Did the memory below actually help shape the response? Answer one word.\n\n"
        f"MEMORY: {memory_content.strip()[:400]}\n\n"
        f"RESPONSE: {response.strip()[:400]}\n\n"
        "Answer 'yes', 'partial', or 'no'."
    )

    try:
        out = generate(prompt, max_tokens=8, temperature=0.1) or ""
    except Exception as e:
        logger.debug(f"usefulness: LLM judge failed: {e}")
        return None

    token = out.strip().lower().split()[0] if out.strip() else ""
    if token.startswith("yes"):
        return 1.0
    if token.startswith("partial") or token.startswith("part"):
        return 0.5
    if token.startswith("no"):
        return 0.0
    return None


def blend(cosine_score: Optional[float],
          llm_score: Optional[float]) -> Optional[float]:
    """Blend cosine + LLM judgements. LLM is authoritative when present."""
    if llm_score is not None and cosine_score is not None:
        return round(0.4 * cosine_score + 0.6 * llm_score, 4)
    if llm_score is not None:
        return llm_score
    return cosine_score


__all__ = [
    "score_usefulness",
    "score_usefulness_llm",
    "is_ambiguous",
    "blend",
    "COSINE_USED_THRESHOLD",
    "COSINE_AMBIGUOUS_LOW",
    "COSINE_AMBIGUOUS_HIGH",
]
