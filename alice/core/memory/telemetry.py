"""
IRIS Recall Telemetry
=====================

Per-turn append-only JSONL log of IRIS recalls and their outcomes.

Foundation for IRIS self-improvement phases (usefulness scoring, bandit
retrieval strategy, Ghost retraining). Without this log, there's no way to
close the "did that memory help?" loop — the signal lives only inside the
turn that produced it.

Output path: `alice/data/turn_logs/recalls-YYYY-MM-DD.jsonl`
  (reuses the existing turn_logs directory — sibling to `turns_*.jsonl`).

Disable via `ALICE_TELEMETRY=0`. Fails open everywhere: if the log sink
breaks, recall keeps working and the turn isn't affected.

Event shape (one line per turn):
    {
      "turn_id": str,       "ts": iso8601,
      "user_id": str,       "query": str,
      "strategy_id": int,                   # 0 = default ACT-R weights
      "tier": "session" | "total" | "mixed",
      "candidates": [{                      # pre-rerank picture
          "mem_id", "snippet", "actr_score",
          "rel", "imp", "rec", "freq",
          "decay_state"
      }],
      "reranked": bool,                     # LLM reranker ran?
      "picked": [mem_id],                   # top-k after rerank
      "used": [mem_id],                     # referenced in response
      "usefulness": {mem_id: float},        # per-memory score
      "response_len": int,
      "latency_ms": float,
    }

Usage pattern (chat.py side):
    tid = telemetry.start_turn(user_id, query)
    context_data["turn_id"] = tid              # also stashed thread-local
    # ... search happens, search.py calls record_candidates via thread-local
    # ... response generated
    telemetry.record_usage(used=[...], usefulness={...}, response_len=N)
    telemetry.finalize_turn(latency_ms=...)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FLAG_ENV = "ALICE_TELEMETRY"
_DEFAULT_LOG_DIR = Path("alice/data/turn_logs")


def is_enabled() -> bool:
    """Default-on; set ALICE_TELEMETRY=0 to disable."""
    val = os.environ.get(_FLAG_ENV, "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def new_turn_id() -> str:
    """8-hex turn id; matches the format already in turns_*.jsonl."""
    return secrets.token_hex(4)


@dataclass
class _TurnBuffer:
    """In-memory accumulator for a single turn's telemetry."""
    turn_id: str
    user_id: str
    query: str
    started_at: float
    ts: str                                  # iso8601
    strategy_id: int = 0
    tier: str = "total"
    # Bandit features captured at pick time — chat.py reads these to issue the
    # post-turn reward update so it doesn't have to re-derive session state.
    strategy_features: List[float] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    reranked: bool = False
    picked: List[str] = field(default_factory=list)
    used: List[str] = field(default_factory=list)
    usefulness: Dict[str, float] = field(default_factory=dict)
    # Parallel-log of the legacy word-overlap score (Phase 4 comparison). Keep
    # until we're confident cosine usefulness doesn't regress ranking.
    word_overlap: Dict[str, float] = field(default_factory=dict)
    response_len: int = 0
    latency_ms: float = 0.0


# Thread-local so chat.py and search.py (which runs deep in the call stack)
# can share the current turn without plumbing turn_id through every signature.
_local = threading.local()
_write_lock = threading.Lock()


def _log_path() -> Path:
    """Path of today's recall log file."""
    return _DEFAULT_LOG_DIR / f"recalls-{date.today().isoformat()}.jsonl"


def start_turn(user_id: str, query: str, strategy_id: int = 0) -> str:
    """
    Open a new turn buffer. Returns the generated turn_id. Safe to call even
    when telemetry is disabled — returns a turn_id either way so callers can
    correlate with their own logs.
    """
    tid = new_turn_id()
    if not is_enabled():
        _local.buffer = None
        return tid

    _local.buffer = _TurnBuffer(
        turn_id=tid,
        user_id=user_id or "",
        query=(query or "")[:500],
        started_at=time.perf_counter(),
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        strategy_id=strategy_id,
    )
    return tid


def _buf() -> Optional[_TurnBuffer]:
    return getattr(_local, "buffer", None)


def current_turn_id() -> Optional[str]:
    b = _buf()
    return b.turn_id if b else None


def record_strategy(
    strategy_id: int,
    features: Optional[List[float]] = None,
    tier: Optional[str] = None,
) -> None:
    """
    Called from iris/search.py right after the bandit picks an arm. Stashes
    the arm id + feature vector so chat.py can issue a matching reward
    update once the turn is scored. Safe to call without a buffer.
    """
    b = _buf()
    if b is None:
        return
    try:
        b.strategy_id = int(strategy_id)
        if features is not None:
            b.strategy_features = list(features)
        if tier:
            b.tier = tier
    except Exception as e:
        logger.debug(f"telemetry.record_strategy failed: {e}")


def current_strategy() -> Optional[Dict[str, Any]]:
    """
    Return the current turn's bandit pick (arm id + features) so the caller
    can issue a LinUCB update after response scoring. None when no buffer
    is open or telemetry is disabled.
    """
    b = _buf()
    if b is None:
        return None
    return {
        "strategy_id": int(b.strategy_id),
        "features": list(b.strategy_features),
    }


def record_candidates(
    candidates: List[Dict[str, Any]],
    tier: str = "total",
    reranked: bool = False,
    picked: Optional[List[str]] = None,
) -> None:
    """
    Called from iris/search.py AFTER ranking/rerank, with the pre-rerank
    candidate pool and the final picks. Multiple calls merge.
    """
    b = _buf()
    if b is None:
        return
    try:
        b.candidates.extend(candidates or [])
        b.tier = tier
        b.reranked = b.reranked or reranked
        if picked:
            b.picked = list(picked)
    except Exception as e:
        logger.debug(f"telemetry.record_candidates failed: {e}")


def record_usage(
    used: Optional[List[str]] = None,
    usefulness: Optional[Dict[str, float]] = None,
    response_len: int = 0,
    word_overlap: Optional[Dict[str, float]] = None,
) -> None:
    """Called from chat.py after Alice responds."""
    b = _buf()
    if b is None:
        return
    try:
        if used:
            b.used = list(used)
        if usefulness:
            b.usefulness.update(usefulness)
        if word_overlap:
            b.word_overlap.update(word_overlap)
        b.response_len = int(response_len)
    except Exception as e:
        logger.debug(f"telemetry.record_usage failed: {e}")


def finalize_turn(latency_ms: Optional[float] = None) -> None:
    """
    Flush the turn buffer to disk as one JSONL record and clear thread-local
    state. No-op when telemetry disabled or no buffer open.
    """
    b = _buf()
    _local.buffer = None
    if b is None:
        return

    try:
        b.latency_ms = float(latency_ms) if latency_ms is not None else (
            (time.perf_counter() - b.started_at) * 1000.0
        )
        record = {
            "turn_id": b.turn_id,
            "ts": b.ts,
            "user_id": b.user_id,
            "query": b.query,
            "strategy_id": b.strategy_id,
            "strategy_features": b.strategy_features,
            "tier": b.tier,
            "candidates": b.candidates,
            "reranked": b.reranked,
            "picked": b.picked,
            "used": b.used,
            "usefulness": b.usefulness,
            "word_overlap": b.word_overlap,
            "response_len": b.response_len,
            "latency_ms": round(b.latency_ms, 2),
        }
        _write_record(record)
    except Exception as e:
        logger.debug(f"telemetry.finalize_turn failed: {e}")


def _write_record(record: Dict[str, Any]) -> None:
    """Append one JSON object as a line to today's log file."""
    path = _log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _write_lock:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.debug(f"telemetry write failed: {e}")


def cancel_turn() -> None:
    """Drop the current turn buffer without writing (for aborted turns)."""
    _local.buffer = None


def candidate_from_search_result(result: Any) -> Dict[str, Any]:
    """
    Build a candidate dict from a SearchResult. Centralized so telemetry stays
    consistent across call sites. Defensive about missing attrs — search
    results come from several paths (session, FAISS, keyword).
    """
    mem = getattr(result, "memory", None)
    content = getattr(mem, "content", "") if mem else ""
    decay = getattr(mem, "decay_state", None)
    decay_str = getattr(decay, "value", str(decay)) if decay else None

    return {
        "mem_id": getattr(mem, "id", "") if mem else "",
        "snippet": (content or "").replace("\n", " ")[:160],
        "actr_score": round(float(getattr(result, "relevance_score", 0.0)), 4),
        "decay_state": decay_str,
        "match_reasons": list(getattr(result, "match_reasons", []) or []),
        "importance": round(float(getattr(mem, "importance", 0.0) or 0.0), 4) if mem else 0.0,
        "access_count": int(getattr(mem, "access_count", 0) or 0) if mem else 0,
    }


__all__ = [
    "is_enabled",
    "new_turn_id",
    "start_turn",
    "current_turn_id",
    "record_candidates",
    "record_strategy",
    "current_strategy",
    "record_usage",
    "finalize_turn",
    "cancel_turn",
    "candidate_from_search_result",
]
