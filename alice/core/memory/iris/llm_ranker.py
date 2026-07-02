"""
LLM Re-ranker for IRIS
======================

Takes FAISS top-K candidates and asks Mind's Ghost model to pick the
semantically relevant ones. Plugs in AFTER cosine/ACT-R scoring so the
cheap signals still do the heavy lifting; the LLM only arbitrates the
shortlist.

Feature-flagged via `ALICE_IRIS_LLM_RERANK=1`. Fails open: on any error
or timeout, returns the input candidates unchanged.

Implementation notes:
- Uses Mind's already-loaded Ghost 1.5B (llama-cpp Q4_K_M). No extra RAM.
- Locks Mind's _gen_lock so we don't collide with Mind's own thinking loop.
- Ghost is paused during Alice generation anyway, so reranker calls
  happen during the pre-generation window.
- Manifest format mirrors Claude Code's `findRelevantMemories.ts`:
  `[idx] content (age)` — keeps the prompt tiny and the parse boring.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_FLAG_ENV = "ALICE_IRIS_LLM_RERANK"
_MAX_CANDIDATES = 20
_TARGET_K = 5
_MAX_TOKENS = 64

_RERANK_PROMPT = """\
<|im_start|>system
You pick memories that are actually relevant to a user's query. Given a query \
and a numbered list of memories, return the numbers of the up-to-{k} most \
relevant ones, comma-separated. No explanation. If none are relevant, reply \
with "none".

Examples:
Query: what's my cat's name
Memories:
[1] Rin has a cat named Pixel with three legs.
[2] Rin lives in Austin.
[3] Rin's favorite game is Hollow Knight.
Answer: 1

Query: games I like
Memories:
[1] Rin works as a software engineer.
[2] Rin played Hollow Knight last year.
[3] Rin beat Elden Ring.
[4] Rin has a cat.
Answer: 2, 3
<|im_end|>
<|im_start|>user
Query: {query}
Memories:
{manifest}
Answer:<|im_end|>
<|im_start|>assistant
"""


def is_enabled() -> bool:
    """Check the feature flag. Off unless explicitly enabled."""
    val = os.environ.get(_FLAG_ENV, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _get_mind():
    """Locate Mind via the registry without hard-importing it (avoids cycles)."""
    try:
        from alice.core.system import get_registry
        registry = get_registry()
        if registry is None:
            return None
        return registry.get("mind")
    except Exception:
        return None


def _memory_content(item: Any) -> str:
    """Extract content string from a SearchResult / Memory / dict / str."""
    if hasattr(item, "memory") and hasattr(item.memory, "content"):
        return item.memory.content
    if hasattr(item, "content"):
        return item.content
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text") or "")
    return str(item)


def _build_manifest(candidates: List[Any]) -> str:
    lines = []
    for i, cand in enumerate(candidates, 1):
        content = _memory_content(cand).replace("\n", " ").strip()
        if len(content) > 180:
            content = content[:177] + "..."
        lines.append(f"[{i}] {content}")
    return "\n".join(lines)


def _parse_answer(answer: str, max_idx: int) -> List[int]:
    """
    Pull 1-based indices from the model's answer. Tolerates junk tokens around
    numbers ('1, 3', '[1], [3]', '1 and 3'). Returns indices in the order the
    model listed them — preserves its ranking.
    """
    answer = answer.strip().lower()
    if not answer or answer.startswith("none"):
        return []
    nums = re.findall(r"\d+", answer)
    seen: set[int] = set()
    out: List[int] = []
    for n in nums:
        try:
            idx = int(n)
        except ValueError:
            continue
        if 1 <= idx <= max_idx and idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def rerank(query: str, candidates: List[Any], k: int = _TARGET_K) -> List[Any]:
    """
    Filter `candidates` down to the ones the LLM judges relevant to `query`.

    Preserves the LLM's ordering for selected items. Items not selected are
    dropped entirely (pure filter, not re-rank — callers who want fallback
    can concat the remainder themselves).

    On any failure, returns `candidates[:k]` unchanged.
    """
    if not candidates:
        return []

    # Trim oversized candidate lists before building the manifest — Ghost is
    # small and long manifests bleed context.
    candidates = list(candidates)[:_MAX_CANDIDATES]

    mind = _get_mind()
    if mind is None or not getattr(mind, "_initialized", False):
        return candidates[:k]

    prompt = _RERANK_PROMPT.format(
        k=k,
        query=query.strip()[:200],
        manifest=_build_manifest(candidates),
    )

    start = time.perf_counter()
    try:
        # Mind's _generate() already holds _gen_lock internally.
        raw = mind._generate(prompt, max_tokens=_MAX_TOKENS)
    except Exception as e:
        logger.warning(f"LLM rerank failed in _generate: {e}")
        return candidates[:k]

    elapsed_ms = (time.perf_counter() - start) * 1000

    indices = _parse_answer(raw, max_idx=len(candidates))
    if not indices:
        # Model said "none" or emitted nothing parseable — fall back to
        # cosine/ACT-R ordering rather than returning empty.
        logger.debug(
            f"LLM rerank returned no selections in {elapsed_ms:.0f}ms "
            f"(raw={raw!r}); falling back to candidate order"
        )
        return candidates[:k]

    reranked = [candidates[i - 1] for i in indices[:k]]
    logger.debug(
        f"LLM rerank: {len(candidates)} → {len(reranked)} in {elapsed_ms:.0f}ms "
        f"(picked={indices[:k]})"
    )
    return reranked


__all__ = ["rerank", "is_enabled"]
