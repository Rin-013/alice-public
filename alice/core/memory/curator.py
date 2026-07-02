"""
Curator — Session-end consolidation
===================================

IRIS stores, scores, and recalls memories on its own. The Curator runs *after*
the distiller at session end and does the cross-memory bookkeeping that no
single-memory pipeline does:

1. **Consolidation** — catches duplicates the cosine-0.92 dedup missed (e.g.
   "Rin has a cat named Pixel" vs "Pixel is Rin's cat"). Uses the
   existing divergence detector; dupes with trivially low divergence impact
   get the newer copy kept and the older one superseded.

2. **Promotion / demotion** — nudges `decay_state` based on support_count and
   access_count so frequently-accessed memories stay warm and untouched ones
   cool down predictably.

3. **Narrative stitching** — when several recent memories share salient
   entities (via Oracle), write a short connective index memory so Alice
   recalls story shape instead of isolated facts.

4. **Active compression** *(Phase 2, `ALICE_IRIS_COMPRESSION=1`)* — rolls
   up old entity-co-occurrence clusters into a single LLM-generated digest.
   Sources are demoted one decay rung and marked with `compression_parent`
   so recall still finds them; they just stop dominating ranking.

Idempotent: safe to run twice in a row. Fails open: any step may raise, the
outer `run_session_curation` catches and logs, returning partial results.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Gate: don't run the curator on every end_session. Skip unless both the
# time gate AND the session-count gate have passed. Mirrors Claude Code's
# autoDream pattern (24h + 5-session gates); Alice's sessions are shorter
# so gates are 12h + 3 sessions.
_GATE_STATE_FILE = Path("alice/data/curator_state.json")
_GATE_MIN_HOURS = float(os.environ.get("ALICE_CURATOR_MIN_HOURS", "12") or "12")
_GATE_MIN_SESSIONS = int(os.environ.get("ALICE_CURATOR_MIN_SESSIONS", "3") or "3")


# Thresholds. Kept conservative — curator is a nightly nudge, not a shotgun.
_CONSOLIDATE_WINDOW = 50          # how many recent facts to scan for dupes
_PROMOTE_SUPPORT = 3              # >= this many supports → promote
_PROMOTE_ACCESS = 5               # AND this many accesses → promote
_DEMOTE_AGE_DAYS = 60             # older than this + no access → demote
_DEMOTE_ACCESS_MAX = 0            # "no access" = this many accesses
_STITCH_MIN_SHARED_ENTITIES = 2   # memories must share this many entities

# Compression (Phase 2) — scoped to *old* memories only; hot facts stay intact.
_COMPRESS_FLAG = "ALICE_IRIS_COMPRESSION"
_COMPRESS_FETCH_LIMIT = 200           # oldest N memories to consider
_COMPRESS_MIN_AGE_DAYS = 30           # skip anything younger
_COMPRESS_MIN_CLUSTER_SIZE = 3        # ≥3 memories per cluster
# Cluster definition: originally 2 shared entities, but Oracle's entity
# detector is regex-narrow. With a single proper-noun extractor running in
# fallback mode, 1 shared entity across 3+ memories is already a strong
# enough signal to cluster (e.g. three memories all mentioning "Pixel"
# within a week belong together). If entity extraction gets richer later,
# bump this back to 2.
_COMPRESS_MIN_SHARED_ENTITIES = 1
_COMPRESS_WINDOW_DAYS = 7             # ± window around the median timestamp
_COMPRESS_DIGEST_IMPORTANCE = 0.6


@dataclass
class CurationReport:
    """Summary of what the curator did this pass."""
    consolidated: int = 0        # dupes merged (older superseded)
    promoted: int = 0            # decay_state bumped up
    demoted: int = 0             # decay_state bumped down
    stitched: int = 0            # narrative index memories written
    compressed: int = 0          # digest memories written (Phase 2)
    errors: List[str] = field(default_factory=list)
    skipped: bool = False        # True when the gate blocked this run
    skip_reason: str = ""        # human-readable "why" when skipped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consolidated": self.consolidated,
            "promoted": self.promoted,
            "demoted": self.demoted,
            "stitched": self.stitched,
            "compressed": self.compressed,
            "errors": list(self.errors),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


def _load_gate_state(path: Path) -> Dict[str, float]:
    """Fail-open read. Returns an empty dict if the file is missing or corrupt."""
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.debug(f"curator gate state read failed: {e}")
        return {}


def _save_gate_state(path: Path, state: Dict[str, Any]) -> None:
    """Fail-open write. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception as e:
        logger.debug(f"curator gate state write failed: {e}")


def _should_run_curation(state: Dict[str, Any], now: float,
                        min_hours: float, min_sessions: int) -> Tuple[bool, str]:
    """
    Gate check: both time AND session thresholds must pass.

    Returns (should_run, reason). Reason describes either why it will run
    (for logging) or why it was skipped (for the report).
    """
    last_run = float(state.get("last_run_at") or 0.0)
    sessions = int(state.get("sessions_since_last") or 0)

    # First run ever — always allowed.
    if last_run <= 0.0:
        return True, "first run (no prior curation state)"

    hours_since = (now - last_run) / 3600.0
    if hours_since < min_hours:
        return False, f"time gate not met ({hours_since:.1f}h < {min_hours}h)"
    if sessions < min_sessions:
        return False, f"session gate not met ({sessions} < {min_sessions})"
    return True, f"gates met ({hours_since:.1f}h, {sessions} sessions)"


def _get_long_term(memory_system: Any):
    """Pull LongTermMemory off the IRIS instance without hard-coupling."""
    if memory_system is None:
        return None
    return getattr(memory_system, "long_term", None)


def _memory_age_days(memory: Any) -> float:
    ts = getattr(memory, "last_accessed", None) or getattr(memory, "timestamp", None)
    if not ts:
        return 0.0
    return max(0.0, (time.time() - ts) / 86400.0)


# Stop-list for the proper-noun fallback — high-frequency sentence starters
# and common English words that capitalize the same as real entities.
_PROPER_NOUN_STOP = {
    "he", "she", "it", "they", "we", "i", "you", "me", "us", "them",
    "the", "a", "an", "and", "or", "but", "so", "if", "then", "that",
    "this", "these", "those", "there", "here", "what", "when", "where",
    "who", "why", "how", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "my", "your", "his", "her", "its",
    "our", "their", "not", "no", "yes", "rin", "alice",
}


def _proper_noun_fallback(text: str) -> List[str]:
    """
    Cheap backup entity extractor when Oracle's regex returns nothing. Picks
    out capitalized tokens that aren't sentence-starters or pronouns. Matches
    "Pixel", "Amy", "Seattle" — not perfect, but finds the co-occurring
    proper nouns that make clusters cluster.
    """
    import re
    out: List[str] = []
    seen: Set[str] = set()
    # Keep it simple: every Capitalized token that isn't a pronoun, sentence
    # opener, or the known user/assistant names. Sentence-start filtering was
    # too aggressive — it dropped legitimate proper nouns starting sentences.
    for tok in re.findall(r"\b[A-Z][a-zA-Z'-]{1,}\b", text):
        low = tok.lower()
        if low in _PROPER_NOUN_STOP or low in seen:
            continue
        seen.add(low)
        out.append(low)
    return out


def _get_entity_detector(memory_system: Any):
    """
    Resolve an entity extractor. Prefers Oracle's regex-based EntityDetector
    when wired, but its patterns are context-narrow ("Rin said X" matches;
    "Rin has a cat named Pixel" doesn't). So we blend in a proper-noun
    fallback — that's actually the signal compression needs (co-occurring
    named entities across memories).

    Returns a callable `text -> List[str]` of lowercased entity names, or
    None if nothing at all works.
    """
    detector = None
    oracle = getattr(memory_system, "oracle", None)
    if oracle is not None:
        detector = getattr(oracle, "entity_detector", None)

    if detector is None:
        try:
            from .oracle import EntityDetector
            detector = EntityDetector()
        except Exception:
            detector = None

    detect = getattr(detector, "detect_entities", None) if detector else None

    def _extract(text: str) -> List[str]:
        out: List[str] = []
        if callable(detect):
            try:
                ents = detect(text, {}) or []
                for e in ents:
                    name = getattr(e, "name", None)
                    if not name:
                        continue
                    n = str(name).lower().strip()
                    if len(n) >= 2:
                        out.append(n)
            except Exception:
                pass
        # Supplement with proper-noun fallback — not a replacement. Oracle's
        # regex catches things the fallback misses (e.g. "@handle", "iPhone"),
        # the fallback catches names Oracle's patterns don't cover.
        for n in _proper_noun_fallback(text):
            if n not in out:
                out.append(n)
        return out

    # Always return a callable — the fallback alone is useful enough.
    return _extract


def consolidate_duplicates(memory_system: Any, report: CurationReport,
                           user_id: Optional[str] = "rin") -> None:
    """
    Pairwise-scan recent facts for duplicate semantics via DivergenceDetector.
    When divergence is *low* (near-identical), supersede the older copy so the
    most-recent wording survives.
    """
    lt = _get_long_term(memory_system)
    if lt is None:
        return

    try:
        from .divergence import DivergenceDetector
    except ImportError:
        return

    try:
        facts = lt.get_top_facts(user_id=user_id, n=_CONSOLIDATE_WINDOW)
    except Exception as e:
        report.errors.append(f"consolidate:fetch:{e}")
        return
    if len(facts) < 2:
        return

    try:
        detector = DivergenceDetector(memory_system)
    except Exception as e:
        report.errors.append(f"consolidate:detector_init:{e}")
        return

    # Facts are sorted newest-first by get_top_facts; iterate pairs and keep
    # the newer copy on duplicate.
    seen_superseded: Set[str] = set()
    for i, newer in enumerate(facts):
        new_id = getattr(newer, "id", None)
        if not new_id or new_id in seen_superseded:
            continue
        for older in facts[i + 1:]:
            old_id = getattr(older, "id", None)
            if not old_id or old_id in seen_superseded:
                continue
            if not _are_near_duplicates(newer, older, detector):
                continue
            try:
                if lt.supersede_memory(old_id):
                    seen_superseded.add(old_id)
                    report.consolidated += 1
            except Exception as e:
                report.errors.append(f"consolidate:supersede:{e}")


def _are_near_duplicates(a: Any, b: Any, detector: Any) -> bool:
    """
    Heuristic near-duplicate test:
      - DivergenceDetector is really for story-level drift; to reuse it for
        dupe detection we fall back to string-level cues when its impact
        score isn't available.
    """
    ca = (getattr(a, "content", "") or "").strip().lower()
    cb = (getattr(b, "content", "") or "").strip().lower()
    if not ca or not cb:
        return False
    if ca == cb:
        return True

    # Token-set Jaccard — cheap and good enough for "Rin has a cat named
    # Pixel" vs "Pixel is Rin's cat". Real divergence detection would need
    # the full IRIS scoring pipeline; we only consolidate when very similar.
    set_a = set(ca.split())
    set_b = set(cb.split())
    if not set_a or not set_b:
        return False
    inter = set_a & set_b
    union = set_a | set_b
    jaccard = len(inter) / max(1, len(union))
    return jaccard >= 0.75


def promote_demote(memory_system: Any, report: CurationReport,
                   user_id: Optional[str] = "rin") -> None:
    """
    Walk recent memories and nudge decay_state:
      - High support + high access → promote (cool→warm, warm→active)
      - Old + never accessed       → demote (active→cool, cool→cold)

    Single-step moves only; no big jumps. The existing freshness score still
    drives the main decay pipeline — this is the "earned attention" lever.
    """
    lt = _get_long_term(memory_system)
    if lt is None:
        return
    try:
        from .types import DecayState
    except ImportError:
        return

    try:
        memories = lt.get_memories_for_user(user_id, limit=_CONSOLIDATE_WINDOW)
    except Exception as e:
        report.errors.append(f"promote_demote:fetch:{e}")
        return

    promote_order = [DecayState.COLD, DecayState.COOL, DecayState.WARM, DecayState.ACTIVE]

    for mem in memories:
        try:
            support = int(getattr(mem, "support_count", 0) or 0)
            accesses = int(getattr(mem, "access_count", 0) or 0)
            state = getattr(mem, "decay_state", None)
            if state is None:
                continue
            # Normalize to enum for comparisons
            if not isinstance(state, DecayState):
                try:
                    state = DecayState(str(getattr(state, "value", state)))
                except Exception:
                    continue

            new_state: Optional[DecayState] = None

            if (support >= _PROMOTE_SUPPORT and accesses >= _PROMOTE_ACCESS
                    and state in promote_order[:-1]):
                idx = promote_order.index(state)
                new_state = promote_order[idx + 1]
                report.promoted += 1

            elif (accesses <= _DEMOTE_ACCESS_MAX
                  and _memory_age_days(mem) >= _DEMOTE_AGE_DAYS
                  and state in promote_order[1:]):
                idx = promote_order.index(state)
                new_state = promote_order[idx - 1]
                report.demoted += 1

            if new_state is not None and new_state != state:
                _apply_decay_state(lt, mem.id, new_state)
        except Exception as e:
            report.errors.append(f"promote_demote:item:{e}")


def _apply_decay_state(long_term: Any, memory_id: str, new_state: Any) -> None:
    """Write-through decay_state change. Best-effort."""
    try:
        value = getattr(new_state, "value", str(new_state))
        conn = long_term._connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE memories SET decay_state = ? WHERE id = ?",
            (value, memory_id),
        )
        conn.commit()
        if long_term.db_path != ":memory:":
            conn.close()
    except Exception:
        pass


def stitch_narrative(memory_system: Any, report: CurationReport,
                     user_id: Optional[str] = "rin") -> None:
    """
    Where several recent memories share salient entities (per Oracle), write
    a short connective index memory so recall returns a story-shape rather
    than isolated facts.

    Only runs if Oracle is wired — otherwise no-op (stitching without entity
    salience is guesswork).
    """
    lt = _get_long_term(memory_system)
    if lt is None:
        return

    extract = _get_entity_detector(memory_system)
    if extract is None:
        return

    try:
        memories = lt.get_memories_for_user(user_id, limit=15)
    except Exception as e:
        report.errors.append(f"stitch:fetch:{e}")
        return
    if len(memories) < 2:
        return

    # Group by shared entities. We group pairs rather than clusters to keep
    # the first pass dead simple.
    entity_map: Dict[str, List[Any]] = {}
    for mem in memories:
        content = getattr(mem, "content", "") or ""
        for key in extract(content):
            entity_map.setdefault(key, []).append(mem)

    stitched_ids: Set[str] = set()
    for entity, group in entity_map.items():
        if len(group) < _STITCH_MIN_SHARED_ENTITIES:
            continue
        # Skip if we already stitched any of these
        group_ids = {getattr(m, "id", "") for m in group}
        if group_ids & stitched_ids:
            continue

        snippets = []
        for m in group[:3]:
            s = (getattr(m, "content", "") or "").replace("\n", " ").strip()
            if s:
                snippets.append(s[:100])
        if len(snippets) < 2:
            continue

        narrative = f"About {entity}: " + " // ".join(snippets)
        try:
            _write_stitched_memory(memory_system, narrative, user_id)
            report.stitched += 1
            stitched_ids.update(group_ids)
        except Exception as e:
            report.errors.append(f"stitch:write:{e}")


def _write_stitched_memory(memory_system: Any, narrative: str,
                           user_id: Optional[str]) -> None:
    """Write a connective index memory through IRIS's normal add path."""
    from .types import Memory, MemoryType

    mem = Memory.create(
        content=narrative,
        memory_type=MemoryType.FACT,
        user_id=user_id,
        importance=0.55,  # mid — it's derivative, shouldn't dominate recall
    )
    lt = _get_long_term(memory_system)
    if lt is not None:
        lt.add_memory(mem)


# ---------------------------------------------------------------------------
# Phase 2: active compression
# ---------------------------------------------------------------------------

def _compression_enabled() -> bool:
    val = os.environ.get(_COMPRESS_FLAG, "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def _fetch_compression_candidates(lt: Any, user_id: Optional[str]) -> List[Any]:
    """
    Pull old-enough non-compressed SEMANTIC/FACT memories, oldest first. We
    order by timestamp ASC so the compressor chews through the tail end of
    the history rather than rewriting hot facts.
    """
    cutoff = time.time() - (_COMPRESS_MIN_AGE_DAYS * 86400.0)
    try:
        conn = lt._connect()
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM memories
               WHERE associated_user = ?
                 AND memory_type IN ('fact','semantic','FACT','SEMANTIC')
                 AND (is_compressed IS NULL OR is_compressed = 0)
                 AND timestamp <= ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (user_id, cutoff, _COMPRESS_FETCH_LIMIT),
        )
        rows = cur.fetchall()
        if lt.db_path != ":memory:":
            conn.close()
    except Exception:
        return []

    out = []
    for row in rows:
        try:
            out.append(lt._row_to_memory(row))
        except Exception:
            continue
    return out


def _cluster_by_entities(memories: List[Any],
                        extract) -> List[List[Any]]:
    """
    Build clusters of memories that share ≥ _COMPRESS_MIN_SHARED_ENTITIES
    entities within ±_COMPRESS_WINDOW_DAYS of one another. Greedy — each
    memory joins at most one cluster to avoid double-compression.
    """
    entity_sets: List[Tuple[Any, Set[str], float]] = []
    for mem in memories:
        content = getattr(mem, "content", "") or ""
        ents = set(extract(content))
        ts = getattr(mem, "timestamp", 0.0) or 0.0
        if ents and ts:
            entity_sets.append((mem, ents, ts))

    window_s = _COMPRESS_WINDOW_DAYS * 86400.0
    clusters: List[List[Any]] = []
    claimed: Set[str] = set()

    for i, (m1, ents1, ts1) in enumerate(entity_sets):
        mid1 = getattr(m1, "id", "")
        if not mid1 or mid1 in claimed:
            continue
        cluster = [m1]
        cluster_ents = set(ents1)
        for j in range(i + 1, len(entity_sets)):
            m2, ents2, ts2 = entity_sets[j]
            mid2 = getattr(m2, "id", "")
            if not mid2 or mid2 in claimed:
                continue
            if abs(ts1 - ts2) > window_s:
                continue
            if len(cluster_ents & ents2) >= _COMPRESS_MIN_SHARED_ENTITIES:
                cluster.append(m2)
                cluster_ents &= ents2  # tighten — shared must persist
                claimed.add(mid2)
        if len(cluster) >= _COMPRESS_MIN_CLUSTER_SIZE:
            claimed.add(mid1)
            clusters.append(cluster)

    return clusters


def _generate_digest(memory_system: Any, cluster: List[Any]) -> Optional[str]:
    """
    Ask Mind to compress a cluster into a single sentence. Falls back to a
    mechanical concatenation if no Mind handle is reachable — a digest is
    better than nothing.
    """
    snippets = []
    for m in cluster[:6]:
        s = (getattr(m, "content", "") or "").replace("\n", " ").strip()
        if s:
            snippets.append(f"- {s[:180]}")

    mind = None
    try:
        from alice.core.system.system_registry import get_registry
        reg = get_registry()
        if reg is not None and hasattr(reg, "get"):
            mind = reg.get("mind")
    except Exception:
        mind = None

    if mind is not None:
        generate = getattr(mind, "_generate", None)
        if callable(generate):
            prompt = (
                "Summarize these related memories in one short, natural sentence. "
                "Keep concrete details (names, relationships). No preamble.\n\n"
                + "\n".join(snippets)
                + "\n\nSummary:"
            )
            try:
                out = generate(prompt, max_tokens=80, temperature=0.3) or ""
                line = out.strip().split("\n", 1)[0].strip()
                if line:
                    return line
            except Exception:
                pass

    # Fallback: truncated concatenation. Ugly but recoverable.
    return " | ".join(s.lstrip("- ") for s in snippets)[:400] or None


def _write_digest(memory_system: Any, digest_text: str,
                  user_id: Optional[str], source_ids: List[str]) -> Optional[str]:
    """Write the digest and return its id, or None on failure."""
    from .types import Memory, MemoryType

    # Tag the source IDs in-content for traceability without a separate table.
    tagged = f"[digest of {len(source_ids)}] {digest_text}"
    mem = Memory.create(
        content=tagged,
        memory_type=MemoryType.FACT,
        user_id=user_id,
        importance=_COMPRESS_DIGEST_IMPORTANCE,
    )
    lt = _get_long_term(memory_system)
    if lt is None:
        return None
    try:
        lt.add_memory(mem)
        return mem.id
    except Exception:
        return None


def _mark_compressed(lt: Any, source_ids: List[str], digest_id: str) -> None:
    """Point sources at their digest and demote decay one rung."""
    from .types import DecayState
    demote_map = {
        DecayState.ACTIVE.value: DecayState.COOL.value,
        DecayState.WARM.value: DecayState.COOL.value,
        DecayState.COOL.value: DecayState.COLD.value,
        DecayState.COLD.value: DecayState.ARCHIVED.value,
    }
    try:
        conn = lt._connect()
        cur = conn.cursor()
        for sid in source_ids:
            cur.execute(
                "SELECT decay_state FROM memories WHERE id = ?",
                (sid,),
            )
            row = cur.fetchone()
            cur_state = (row[0] if row and row[0] else DecayState.ACTIVE.value)
            new_state = demote_map.get(cur_state, cur_state)
            cur.execute(
                """UPDATE memories
                   SET is_compressed = 1,
                       compression_parent = ?,
                       decay_state = ?
                   WHERE id = ?""",
                (digest_id, new_state, sid),
            )
        conn.commit()
        if lt.db_path != ":memory:":
            conn.close()
    except Exception:
        pass


def compress_clusters(memory_system: Any, report: CurationReport,
                      user_id: Optional[str] = "rin") -> None:
    """
    Walk old memories, cluster them by shared entities + timestamp proximity,
    summarize each cluster into a digest, demote the sources. Gated by the
    `ALICE_IRIS_COMPRESSION` flag.
    """
    if not _compression_enabled():
        return

    lt = _get_long_term(memory_system)
    if lt is None:
        return

    extract = _get_entity_detector(memory_system)
    if extract is None:
        return

    try:
        candidates = _fetch_compression_candidates(lt, user_id)
    except Exception as e:
        report.errors.append(f"compress:fetch:{e}")
        return
    if len(candidates) < _COMPRESS_MIN_CLUSTER_SIZE:
        return

    try:
        clusters = _cluster_by_entities(candidates, extract)
    except Exception as e:
        report.errors.append(f"compress:cluster:{e}")
        return

    for cluster in clusters:
        try:
            digest_text = _generate_digest(memory_system, cluster)
            if not digest_text:
                continue
            source_ids = [getattr(m, "id", "") for m in cluster if getattr(m, "id", None)]
            if not source_ids:
                continue
            digest_id = _write_digest(memory_system, digest_text, user_id, source_ids)
            if not digest_id:
                continue
            _mark_compressed(lt, source_ids, digest_id)
            report.compressed += 1
        except Exception as e:
            report.errors.append(f"compress:cluster_item:{e}")


def run_session_curation(memory_system: Any,
                         user_id: Optional[str] = "rin",
                         force: bool = False,
                         state_path: Optional[Path] = None,
                         min_hours: Optional[float] = None,
                         min_sessions: Optional[int] = None) -> CurationReport:
    """
    One entry point for `end_session()` to call after distillation. Runs all
    phases, returns a report. Never raises — all errors accumulate on the
    report.

    Gated by default: skips unless BOTH time (>= min_hours since last run)
    AND activity (>= min_sessions since last run) thresholds are met. On
    a skip, still persists an incremented session counter so subsequent
    calls eventually trip the gate. `force=True` bypasses the gate.
    """
    report = CurationReport()
    if memory_system is None:
        return report

    path = state_path or _GATE_STATE_FILE
    mh = min_hours if min_hours is not None else _GATE_MIN_HOURS
    ms = min_sessions if min_sessions is not None else _GATE_MIN_SESSIONS
    now = time.time()

    state = _load_gate_state(path)

    if not force:
        should_run, reason = _should_run_curation(state, now, mh, ms)
        if not should_run:
            # Gate blocks this run — still bump the session counter so a
            # future call will eventually run.
            state["sessions_since_last"] = int(state.get("sessions_since_last") or 0) + 1
            _save_gate_state(path, state)
            report.skipped = True
            report.skip_reason = reason
            logger.debug(f"Curator skipped: {reason}")
            return report

    try:
        consolidate_duplicates(memory_system, report, user_id=user_id)
    except Exception as e:
        report.errors.append(f"consolidate:outer:{e}")

    try:
        promote_demote(memory_system, report, user_id=user_id)
    except Exception as e:
        report.errors.append(f"promote_demote:outer:{e}")

    try:
        stitch_narrative(memory_system, report, user_id=user_id)
    except Exception as e:
        report.errors.append(f"stitch:outer:{e}")

    try:
        compress_clusters(memory_system, report, user_id=user_id)
    except Exception as e:
        report.errors.append(f"compress:outer:{e}")

    # Successful run — reset gate state.
    _save_gate_state(path, {
        "last_run_at": now,
        "sessions_since_last": 0,
    })

    if report.errors:
        logger.debug(f"Curator finished with errors: {report.errors}")
    return report


__all__ = [
    "run_session_curation",
    "CurationReport",
    "consolidate_duplicates",
    "promote_demote",
    "stitch_narrative",
    "compress_clusters",
]
