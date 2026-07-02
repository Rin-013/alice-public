# Copyright 2025 Rin - Alice AI System
"""
IRIS - Intelligent Retrieval and Indexing System
=================================================

THE single memory interface for Alice.

Handles:
  - 3-tier storage: short_term (prompt), session (FAISS), long_term (SQLite+FAISS)
  - Tiered retrieval: session first, then total with ACT-R scoring
  - Conversation storage, session lifecycle, distillation
  - Thought/observation storage, user profiles, IRIS feedback loop
  - Pinned context (always-on facts), recent context

ACT-R scoring (total memory only):
  score = relevance * 0.50
        + importance * 0.25
        + recency    * 0.15    (exponential decay from last_accessed)
        + frequency  * 0.10    (log-scaled access_count)

Session results get a +SESSION_BOOST to score, ensuring current-session
memories always rank above equally-relevant old memories.
"""

import logging
import math
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("alice.memory.iris")

from ..types import (
    Memory, MemoryType, MemoryDepth,
    SearchQuery, SearchResult, SearchType,
    UserProfile, AkashicRecord, ChoiceRecord,
)

# Canonical DB path: always resolve to alice/data/databases/ regardless of cwd
_DEFAULT_DB_PATH = str(Path(__file__).resolve().parents[3] / "data" / "databases" / "alice_memory.db")

# Score added to session-tier results to ensure they surface above total-tier
SESSION_BOOST = 0.20

# Recency decay rate per hour (0.990 -> half-life ~ 68 hours ~ 2.8 days)
RECENCY_DECAY_RATE = 0.990


def _actr_score(memory, relevance: float, now: float,
                weights: Optional[tuple] = None) -> float:
    """
    ACT-R-inspired retrieval score for total-memory results.

    Parameters tuned for personal companion use:
    - Recency based on last_accessed (not creation) - rewards frequently reviewed facts
    - Frequency log-scaled - more accesses = higher base activation
    - Importance stored at write time - core facts (name, location) outrank small talk
    - Usefulness EMA kicks in once `usefulness_n >= _USEFULNESS_WARMUP` so a
      single bad sample can't tank a memory's ranking.
    - `weights` optionally overrides (rel, imp, rec, freq). When the bandit
      is active, this comes from its selected arm; otherwise we fall back
      to production.
    """
    last_acc = getattr(memory, 'last_accessed', None) or getattr(memory, 'timestamp', now)
    hours_since_access = max(0.0, (now - last_acc) / 3600.0)
    recency = RECENCY_DECAY_RATE ** hours_since_access

    access_count = getattr(memory, 'access_count', 0) or 0
    frequency = math.log1p(access_count) / 3.0  # normalised: ~1.0 at 20 accesses

    importance = min(1.0, getattr(memory, 'importance', 0.5) or 0.5)
    rel = max(0.0, min(1.0, relevance))

    w_rel, w_imp, w_rec, w_freq = weights if weights else (0.50, 0.25, 0.15, 0.10)

    u_ema = float(getattr(memory, 'usefulness_ema', 0.0) or 0.0)
    u_n = int(getattr(memory, 'usefulness_n', 0) or 0)
    if u_n >= _USEFULNESS_WARMUP:
        # Usefulness takes 10% from whichever knob the current arm weights
        # the frequency term at — keeps rel/imp/rec contributions intact
        # regardless of which arm was picked.
        u_take = min(w_freq, 0.10)
        return (rel * w_rel + importance * w_imp + recency * w_rec
                + u_ema * u_take + frequency * max(0.0, w_freq - u_take))
    return rel * w_rel + importance * w_imp + recency * w_rec + frequency * w_freq


# Minimum number of usefulness samples before ACT-R starts blending the EMA.
# Keeps a single noisy turn from dominating ranking.
_USEFULNESS_WARMUP = 5


@dataclass
class IRISConfig:
    default_limit: int = 10
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    recency_weight: float = 0.1
    enable_semantic: bool = True
    enable_emotional_matching: bool = True
    fast_path_threshold_ms: float = 100.0


class IRIS:
    """
    IRIS - THE single memory interface for Alice.

    Manages:
    - short_term:    In-memory list of recent exchanges (prompt context)
    - session_store: FAISS-backed session index (searchable, cleared each session)
    - long_term:     SQLite + persistent FAISS (all sessions, ACT-R scored)
    - Tiered search: session first, then total with ACT-R scoring
    """

    def __init__(self,
                 db_path: str = _DEFAULT_DB_PATH,
                 storage=None,
                 config: IRISConfig = None,
                 legacy_iris=None,
                 fast_path=None,
                 session_store=None):
        from ..storage import ShortTermMemory
        from ..storage.long_term import LongTermMemory, LONG_TERM_AVAILABLE
        from ..storage.session import SessionMemoryStore
        from ..importance import score_importance

        self.db_path = db_path
        self.config = config or IRISConfig()
        self._legacy_iris = legacy_iris
        self._fast_path = fast_path
        self._score_importance = score_importance

        # Tier 1 - prompt context list
        self.short_term = ShortTermMemory()

        # Tier 2 - session FAISS index (in-memory, cleared each session)
        self.session_store = session_store or SessionMemoryStore()

        # Tier 3 - persistent storage
        if storage is not None:
            self.storage = storage
        elif LONG_TERM_AVAILABLE:
            self.storage = LongTermMemory(db_path)
        else:
            self.storage = None
            print("   LongTermMemory not available")
        self.long_term = self.storage  # alias

        # Current user tracking
        self.current_user: Optional[UserProfile] = None
        self.current_session_id: Optional[str] = None

        # Optional subsystems (lazy-initialized)
        self._smart_facts = None
        self._divergence = None
        self._trauma = None
        self._depth = None  # (AkashicRecords, IndexSystem) — the two-bank lived record

        self.stats = {
            "total_searches": 0,
            "session_hits": 0,
            "total_hits": 0,
            "fast_path_hits": 0,
            "semantic_searches": 0,
            "keyword_fallbacks": 0,
            "average_time_ms": 0.0,
        }

        print("✅ IRIS memory system initialized")
        print(f"   Short-term:    Ready")
        print(f"   Session store: Ready")
        print(f"   Long-term:     {'Ready' if self.storage else 'Unavailable'}")

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, user_id: str, name: str) -> UserProfile:
        """Start a new session. Clears session store, resets short-term."""
        self.short_term.clear()
        self.short_term.set_user(user_id)
        self.current_session_id = self.short_term.session_id

        # Clear session store for fresh session
        self.session_store.clear()

        # Get or create user profile
        if self.long_term:
            self.current_user = self.long_term.get_or_create_user(user_id, name)
        else:
            self.current_user = UserProfile.create(user_id, name)

        return self.current_user

    def end_session(self, distiller=None, distill_model=None,
                    distill_tokenizer=None) -> Dict[str, Any]:
        """
        End the current session.

        If a distiller + model are provided, runs LLM-based fact extraction
        over the session exchanges and stores distilled facts in total memory.

        If no distiller is provided, falls back to flushing raw session
        conversation pairs into total memory.
        """
        summary = self.short_term.get_session_summary()

        if self.long_term:
            if distiller is not None and distill_model is not None:
                n_facts = distiller.distill_into_memory(
                    self, distill_model, distill_tokenizer
                )
                summary["distilled_facts"] = n_facts
                summary["session_flush_mode"] = "distilled"
            else:
                flushed = 0
                for entry in self.session_store.get_all():
                    memory = Memory.create(
                        content=entry["content"],
                        memory_type=MemoryType.CONVERSATION,
                        user_id=self.current_user.user_id if self.current_user else None,
                        importance=entry["metadata"].get("importance", 0.5),
                    )
                    try:
                        self.long_term.add_memory(memory)
                        flushed += 1
                    except Exception:
                        pass
                summary["session_flushed"] = flushed
                summary["session_flush_mode"] = "raw"

        # Update user's last interaction
        if self.current_user and self.long_term:
            self.current_user.record_interaction()
            self.long_term.save_user_profile(self.current_user)

        # Divergence review (optional)
        if self._divergence is None:
            try:
                from ..divergence import DivergenceDetector
                self._divergence = DivergenceDetector(self)
            except ImportError:
                self._divergence = False
        if self._divergence:
            try:
                div_events = self._divergence.run_retrospective_divergence_review()
                if div_events:
                    summary["divergence_events"] = len(div_events)
            except Exception:
                pass

        # Curator pass — consolidates dupes, promotes/demotes decay, stitches
        # narrative. Minimal first pass; runs after distillation so it operates
        # on the flushed long-term state.
        try:
            from ..curator import run_session_curation
            user_id = self.current_user.user_id if self.current_user else "rin"
            curation = run_session_curation(self, user_id=user_id)
            summary["curation"] = curation.to_dict()
        except Exception:
            pass

        # Persist FAISS to disk
        if self.long_term and hasattr(self.long_term, 'flush_faiss'):
            self.long_term.flush_faiss()

        # Clear session store for next session
        self.session_store.clear()

        return summary

    # =========================================================================
    # CONVERSATION MEMORY
    # =========================================================================

    def add_conversation(self,
                         user_message: str,
                         alice_response: str,
                         emotional_context: Optional[Dict] = None,
                         importance: float = None,
                         drive_snapshot: Optional[Any] = None) -> Optional[str]:
        """
        Add a conversation exchange to memory.

        Writes to Tier 2 (session store) only.
        Tier 3 (long-term) is written on end_session() flush.

        emotional_context, if not provided, is built from `drive_snapshot`
        (Alice's internal affect at write time). When neither is provided,
        the memory is tagged with no emotional context.
        """
        content = f"User: {user_message}\nAlice: {alice_response}"

        scored_importance = self._score_importance(
            user_message, base=importance if importance is not None else 0.35
        )

        # Tier 1 - short-term exchange list (prompt context)
        self.short_term.add_exchange(
            user_message=user_message,
            assistant_response=alice_response,
            emotional_context=str(emotional_context) if emotional_context else None
        )

        # Emotional tagging from Alice's drive snapshot at write time.
        if emotional_context is None and drive_snapshot is not None:
            try:
                from ..emotional_tagging import tag_memory_emotion
                emotional_context = tag_memory_emotion(
                    user_message, alice_response, drive_snapshot=drive_snapshot
                )
            except ImportError:
                pass

        # Tier 2 - session store (searchable in-session)
        memory_id = self.session_store.add(
            content=content,
            metadata={
                "importance": scored_importance,
                "emotional_context": emotional_context,
                "user_id": self.current_user.user_id if self.current_user else None,
            }
        )

        # Depth layer (revived 2026-06-10): akashic keeps the objective record
        # of the exchange, index_memories keeps Alice's subjective copy —
        # emotionally colored when Mind's tags flow, surface-depth until then.
        # Same event_id links the pair so divergence review can compare them.
        self._write_depth_record(memory_id, content, emotional_context,
                                 scored_importance)

        # Smart fact extraction — OFF by default since 2026-06-10: the regex
        # extractor produced 27/27 garbage facts at importance 0.9-1.0 in the
        # live DB ("Rin's name is lol ok.", "company = Liyue"), polluting the
        # pinned "What I know about Rin" prompt slot. Re-enable for testing
        # with ALICE_SMART_FACTS=1. Real fix is a trained extractor or letting
        # Mind do it (train-don't-heuristic) — see IRIS deep-dive task.
        if self._smart_facts is None:
            import os
            if os.environ.get("ALICE_SMART_FACTS", "0") != "1":
                self._smart_facts = False
            else:
                try:
                    from ..smart_facts import SmartFactExtractor
                    self._smart_facts = SmartFactExtractor(self)
                except ImportError:
                    self._smart_facts = False
        if self._smart_facts and self.current_user:
            try:
                self._smart_facts.extract_facts_from_conversation(
                    user_message, alice_response, self.current_user.user_id
                )
            except Exception:
                pass

        return memory_id

    def _write_depth_record(self, event_id: str, content: str,
                            emotional_context: Optional[Dict],
                            importance: float) -> None:
        """
        Write one exchange into the two-bank lived record:
        akashic_records (objective, immutable) + index_memories (Alice's
        subjective copy: depth, influence, emotional coloring).

        Fail-open — the conversation never blocks on the depth layer.
        """
        if self._depth is None:
            try:
                from ..storage.akashic import AkashicRecords
                from ..storage.index_system import IndexSystem
                self._depth = (AkashicRecords(str(self.db_path)),
                               IndexSystem(str(self.db_path)))
            except Exception as e:
                logger.warning(f"depth layer unavailable: {e}")
                self._depth = False
        if not self._depth:
            return

        akashic, index = self._depth
        user_id = self.current_user.user_id if self.current_user else None
        try:
            akashic.add_record(event_id, who=[user_id or "user", "alice"],
                               what=content)

            ec = emotional_context or {}
            mem = Memory(
                event_id=event_id,
                content=content,
                memory_type=MemoryType.CONVERSATION,
                user_id=user_id,
                importance=importance,
                emotional_context_data=ec,
                emotional_markers=list(ec.get("markers") or []),
                # Influence inputs: emotion from Alice's affect at write time,
                # salience seeded from scored importance. The rest (repetition,
                # uniqueness, contrast) accrue through lived access patterns.
                emotion_intensity=float(ec.get("arousal") or 0.0),
                salience=importance,
                alice_emotional_valence=ec.get("valence"),
                alice_emotional_arousal=ec.get("arousal"),
                alice_emotional_weight=ec.get("weight"),
            )
            # Deep/core candidates get a divergence review a week out —
            # that's what end_session's retrospective review consumes.
            mem.calculate_influence_scores()
            if mem.influence_long >= 0.6:
                mem.review_scheduled = time.time() + 7 * 86400
            index.add_memory(mem)  # recomputes influence + assigns depth
        except Exception as e:
            logger.warning(f"depth-layer write failed for {event_id}: {e}")

    # =========================================================================
    # SEARCH
    # =========================================================================

    def search(self,
               query: str,
               user_id: Optional[str] = None,
               k: int = None,
               search_type: SearchType = SearchType.HYBRID,
               emotional_context: Optional[Dict[str, float]] = None,
               include_archived: bool = False,
               memory_type_filter: Optional[str] = None) -> List[SearchResult]:
        """
        Tiered search: session store -> total memory with ACT-R scoring.

        Session results always surface above equally-relevant total results.
        Total results are re-scored using ACT-R (recency x frequency x importance x relevance).
        """
        start = time.perf_counter()
        k = k or self.config.default_limit
        self.stats["total_searches"] += 1
        now = time.time()

        results: List[SearchResult] = []

        # -- Tier 2: Session store (fast, high signal) -----------------------
        if self.session_store is not None:
            try:
                session_hits = self.session_store.search(query, k=k)
                for hit in session_hits:
                    mem = Memory.create(
                        content=hit["content"],
                        memory_type="conversation",
                    )
                    mem.id = hit["memory_id"]
                    mem.timestamp = hit["timestamp"]
                    score = min(1.0, hit["score"] + SESSION_BOOST)
                    results.append(SearchResult(
                        memory=mem,
                        relevance_score=score,
                        match_reasons=["session", "faiss_semantic"],
                    ))
                if session_hits:
                    self.stats["session_hits"] += 1
            except Exception:
                pass

        # -- Tier 3: Total memory (persistent, ACT-R scored) -----------------
        search_query = SearchQuery(
            text=query,
            user_id=user_id,
            search_type=search_type,
            include_archived=include_archived,
            limit=k,
        )

        total_results: List[SearchResult] = []
        if search_type == SearchType.HYBRID:
            total_results = self._hybrid_search(search_query, emotional_context, memory_type_filter)
        elif search_type == SearchType.SEMANTIC:
            total_results = self._semantic_search(search_query, emotional_context, memory_type_filter)
        elif search_type == SearchType.TEXT:
            total_results = self._keyword_search(search_query)
        elif search_type == SearchType.EMOTIONAL:
            total_results = self._emotional_search(search_query, emotional_context)
        else:
            total_results = self._hybrid_search(search_query, emotional_context, memory_type_filter)

        # Bandit picks which weight tuple to score this turn with. When the
        # flag's off, `pick_strategy` returns 0 → production weights. The
        # chosen arm is also stashed in telemetry so reward attribution works.
        strategy_id = 0
        strategy_features: list[float] = []
        weights = None
        try:
            from . import strategy_bandit
            if strategy_bandit.is_enabled():
                session_idx = len(self.short_term.conversation_history) if self.short_term else 0
                strategy_features = strategy_bandit.featurize(query, session_turn_idx=session_idx)
                bandit = strategy_bandit.get_bandit()
                strategy_id = bandit.pick_strategy(strategy_features)
                weights = strategy_bandit.weights_for(strategy_id)
        except Exception:
            strategy_id = 0
            strategy_features = []
            weights = None

        # Apply ACT-R re-scoring to total results
        for r in total_results:
            r.relevance_score = _actr_score(r.memory, r.relevance_score, now, weights=weights)
            r.match_reasons.append("actr_scored")

        if total_results:
            self.stats["total_hits"] += 1

        results.extend(total_results)

        # -- Dedupe and rank --------------------------------------------------
        # Over-fetch when the LLM re-ranker is active so Ghost has more to
        # choose from; it'll trim back to k. No over-fetch cost when disabled.
        from . import llm_ranker
        reranker_on = llm_ranker.is_enabled()
        if reranker_on:
            pre_rerank = self._dedupe_and_rank(results, max(k * 3, 20))
            final = llm_ranker.rerank(query, pre_rerank, k=k)
        else:
            pre_rerank = self._dedupe_and_rank(results, k)
            final = pre_rerank

        # Telemetry: emit the pre-rerank candidate pool and final picks so
        # downstream phases (usefulness scoring, bandit reward, Ghost training)
        # can reconstruct why each memory surfaced. Fail-open.
        try:
            from .. import telemetry as _tele
            if _tele.is_enabled() and _tele.current_turn_id():
                tier_label = "session" if self.stats.get("session_hits") and not total_results else (
                    "mixed" if (self.stats.get("session_hits") and total_results) else "total"
                )
                cand_dicts = [_tele.candidate_from_search_result(r) for r in pre_rerank]
                _tele.record_candidates(
                    candidates=cand_dicts,
                    tier=tier_label,
                    reranked=reranker_on,
                    picked=[r.memory.id for r in final if getattr(r, "memory", None)],
                )
                _tele.record_strategy(strategy_id, features=strategy_features, tier=tier_label)
        except Exception:
            pass

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._update_stats(elapsed_ms)

        return final

    def search_memories(self,
                        query: str,
                        user_id: Optional[str] = None,
                        k: int = 10,
                        search_type: SearchType = SearchType.HYBRID,
                        memory_type_filter: Optional[str] = None) -> List[SearchResult]:
        """Backward-compat wrapper for search()."""
        if user_id is None and self.current_user:
            user_id = self.current_user.user_id
        return self.search(
            query=query, user_id=user_id, k=k,
            search_type=search_type, memory_type_filter=memory_type_filter
        )

    def get_recent_context(self, num_exchanges: int = 5) -> List[Dict]:
        """Get recent conversation context from short-term memory."""
        return self.short_term.get_recent_context(num_exchanges)

    # =========================================================================
    # ALICE-SPECIFIC
    # =========================================================================

    def alice_think(self, thought: str, thought_type: str = "reflection") -> Optional[str]:
        """Record one of Alice's internal thoughts (goes directly to long-term)."""
        if not self.long_term:
            return None
        memory = Memory.create(
            content=thought,
            memory_type=MemoryType.THOUGHT,
            importance=0.4
        )
        try:
            self.long_term.add_memory(memory)
            return memory.id
        except Exception as e:
            print(f"   Failed to store thought: {e}")
            return None

    def alice_observe(self, observation: str) -> Optional[str]:
        """Record something Alice observed (goes directly to long-term)."""
        if not self.long_term:
            return None
        memory = Memory.create(
            content=observation,
            memory_type=MemoryType.OBSERVATION,
            importance=0.3
        )
        try:
            self.long_term.add_memory(memory)
            return memory.id
        except Exception:
            return None

    # =========================================================================
    # USER MANAGEMENT
    # =========================================================================

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        if self.long_term:
            return self.long_term.get_or_create_user(user_id, "Unknown")
        return None

    def update_user_trust(self, delta: float):
        if self.current_user:
            self.current_user.trust_level = max(0, min(10,
                self.current_user.trust_level + delta))
            if self.long_term:
                self.long_term.save_user_profile(self.current_user)

    # =========================================================================
    # IRIS FEEDBACK (self-learning)
    # =========================================================================

    def record_usage(self, memory_id: str, delta: float = 0.05):
        """
        Signal that a retrieved memory was actually used in a response.
        Boosts its importance so it surfaces higher next time.
        """
        if self.storage and hasattr(self.storage, 'boost_importance'):
            try:
                self.storage.boost_importance(memory_id, delta)
            except Exception:
                pass

    def record_memory_used(self, memory_id: str, delta: float = 0.05):
        """Backward-compat alias for record_usage()."""
        self.record_usage(memory_id, delta)

    # =========================================================================
    # PINNED CONTEXT
    # =========================================================================

    def get_pinned_context(self, n: int = 10) -> str:
        """
        Return the top-N highest-importance facts about the current user,
        formatted for direct injection into the system prompt.
        """
        if not self.long_term:
            return ""
        user_id = self.current_user.user_id if self.current_user else None
        facts = self.long_term.get_top_facts(user_id=user_id, n=n)
        if not facts:
            return ""
        lines = [f"- {f.content}" for f in facts]
        return "[What I know about Rin]\n" + "\n".join(lines)

    def get_identity_context(self, n: int = 16) -> str:
        """
        Return Alice's own identity memories (the cartridge canon: interests,
        formative moments, standing opinions), formatted for prompt injection.
        Stable across turns — this is "who you are", not "what's relevant now".
        """
        if not self.long_term:
            return ""
        getter = getattr(self.long_term, "get_self_memories", None)
        if getter is None:
            return ""
        memories = getter(n=n)
        if not memories:
            return ""
        lines = [f"- {m.content}" for m in memories]
        return "[Who you are]\n" + "\n".join(lines)

    def add_self_memory(self, text: str, importance: float = 0.65,
                        memory_type: str = "thought") -> bool:
        """
        Store a memory about Alice herself — an opinion she formed, something
        she realized she likes, a moment that shaped her. These grow the
        identity cartridge: they surface through get_identity_context() and
        compete with the seeded canon by importance. Semantic dedup and
        contradiction supersede come free from long_term.add_memory().
        """
        text = (text or "").strip()
        if not text or not self.long_term:
            return False
        try:
            import uuid as _uuid
            from ..types import Memory, MemoryType
            try:
                mtype = MemoryType(memory_type)
            except ValueError:
                mtype = MemoryType.THOUGHT
            mem = Memory(
                id=f"self_{_uuid.uuid4().hex[:12]}",
                content=text,
                timestamp=time.time(),
                memory_type=mtype,
                importance=max(0.0, min(0.7, importance)),
                user_id="alice",
            )
            self.long_term.add_memory(mem)
            return True
        except Exception:
            return False

    # =========================================================================
    # STATS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "short_term": {
                "exchanges": len(self.short_term.conversation_history),
                "session_id": self.short_term.session_id,
                "topics": self.short_term.active_topics,
            },
            "session_store": {
                "entries": self.session_store.size,
            },
            "iris": self.stats.copy(),
            "current_user": self.current_user.user_id if self.current_user else None,
        }
        if self.long_term and hasattr(self.long_term, 'get_memory_decay_stats'):
            try:
                stats["decay"] = self.long_term.get_memory_decay_stats()
            except Exception:
                pass
        return stats

    # =========================================================================
    # KNOWLEDGE BASE — static YAML facts (Genshin / Hololive / anime / etc.)
    # =========================================================================

    def search_knowledge_base(self, query: str, k: int = 5) -> list:
        """
        Look up static knowledge_base entries (alice/core/memory/knowledge_base/).

        Returns list of dicts: {name, source, facts, fact_list, score}.
        Cosine-nearest over MiniLM embeddings of (name + joined facts).
        Empty list if KB unavailable (faiss/embeddings not loaded).
        """
        try:
            from alice.core.memory.knowledge_base_search import get_knowledge_base
            kb = get_knowledge_base()
            return kb.search(query, k=k)
        except Exception as e:
            logger.warning(f"kb search failed: {e}")
            return []

    # =========================================================================
    # INTERNAL SEARCH METHODS
    # =========================================================================

    def _hybrid_search(self,
                       query: SearchQuery,
                       emotional_context: Optional[Dict],
                       memory_type_filter: Optional[str] = None) -> List[SearchResult]:
        results = []

        if self.storage and hasattr(self.storage, 'vector_search') and self.config.enable_semantic:
            try:
                if hasattr(self.storage, 'vector_search_with_type'):
                    vector_results = self.storage.vector_search_with_type(
                        query.text, k=query.limit, user_id=query.user_id,
                        index_type=memory_type_filter
                    )
                else:
                    vector_results = self.storage.vector_search(
                        query.text, k=query.limit, user_id=query.user_id
                    )
                if vector_results:
                    self.stats["semantic_searches"] += 1
                    for mem in vector_results:
                        score = getattr(mem, 'similarity_score', 0.7)
                        results.append(SearchResult(
                            memory=mem if isinstance(mem, Memory) else self._convert_to_memory(mem),
                            relevance_score=score,
                            match_reasons=["vector_semantic"],
                        ))
            except Exception:
                pass

        if self._fast_path and self.config.enable_semantic:
            try:
                fast_results = self._fast_path_search(query)
                if fast_results:
                    self.stats["fast_path_hits"] += 1
                    results.extend(fast_results)
            except Exception:
                pass

        if self._legacy_iris and self.config.enable_semantic:
            try:
                semantic_results = self._semantic_search(query, emotional_context)
                self.stats["semantic_searches"] += 1
                results.extend(semantic_results)
            except Exception:
                pass

        keyword_results = self._keyword_search(query)
        if keyword_results and not results:
            self.stats["keyword_fallbacks"] += 1
        results.extend(keyword_results)

        return self._dedupe_and_rank(results, query.limit)

    def _semantic_search(self,
                         query: SearchQuery,
                         emotional_context: Optional[Dict],
                         memory_type_filter: Optional[str] = None) -> List[SearchResult]:
        if not self._legacy_iris:
            return []
        try:
            from ..iris import SearchContext
            context = SearchContext(
                query=query.text,
                user_id=query.user_id or "",
                emotional_state=emotional_context or {},
                search_intent="general",
            )
            matches = self._legacy_iris.smart_search(context, top_k=query.limit)
            results = []
            for match in matches:
                memory = Memory.create(
                    content=match.content,
                    memory_type=Memory.MemoryType.CONVERSATION if hasattr(Memory, 'MemoryType') else "conversation",
                )
                memory.id = match.memory_id
                memory.similarity_score = match.relevance_score
                results.append(SearchResult(
                    memory=memory,
                    relevance_score=match.relevance_score,
                    match_reasons=match.match_reasons,
                ))
            return results
        except Exception:
            return []

    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        if not self.storage:
            return []
        try:
            if hasattr(self.storage, 'search_memories'):
                memories = self.storage.search_memories(query.text, user_id=query.user_id)
                return [
                    SearchResult(
                        memory=mem if isinstance(mem, Memory) else self._convert_to_memory(mem),
                        relevance_score=0.5,
                        match_reasons=["keyword_match"],
                    )
                    for mem in memories[:query.limit]
                ]
        except Exception:
            pass
        return []

    def _emotional_search(self,
                          query: SearchQuery,
                          emotional_context: Optional[Dict]) -> List[SearchResult]:
        results = self._semantic_search(query, emotional_context)
        if emotional_context and results:
            for result in results:
                boost = self._calculate_emotional_resonance(result.memory, emotional_context)
                result.relevance_score *= (1 + boost * 0.3)
            results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def _fast_path_search(self, query: SearchQuery) -> List[SearchResult]:
        if not self._fast_path:
            return []
        try:
            result = self._fast_path.retrieve(query.text, limit=query.limit, user_id=query.user_id)
            search_results = []
            for conv in result.get("conversations", []):
                memory = Memory.create(content=conv.get("content", ""), memory_type="conversation")
                memory.id = conv.get("id", "")
                memory.timestamp = conv.get("timestamp", 0)
                memory.similarity_score = conv.get("similarity", 0.5)
                search_results.append(SearchResult(
                    memory=memory,
                    relevance_score=conv.get("similarity", 0.5),
                    match_reasons=["fast_path", "faiss_semantic"],
                ))
            return search_results
        except Exception:
            return []

    def _dedupe_and_rank(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        seen_ids = set()
        unique = []
        for r in results:
            mid = r.memory.id
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique.append(r)
        unique.sort(key=lambda x: x.relevance_score, reverse=True)
        return unique[:limit]

    def _calculate_emotional_resonance(self, memory: Memory, emotional_context: Dict) -> float:
        if not emotional_context:
            return 0.0
        resonance = 0.0
        if hasattr(memory, 'emotional_valence'):
            resonance += (1 - abs(memory.emotional_valence - emotional_context.get('valence', 0))) * 0.5
        if hasattr(memory, 'emotional_arousal'):
            resonance += (1 - abs(memory.emotional_arousal - emotional_context.get('arousal', 0.5))) * 0.3
        if hasattr(memory, 'emotional_markers') and memory.emotional_markers:
            ctx_emotions = emotional_context.get('emotions', [])
            if ctx_emotions:
                matches = len(set(memory.emotional_markers) & set(ctx_emotions))
                resonance += (matches / max(len(memory.emotional_markers), 1)) * 0.2
        return min(resonance, 1.0)

    def _convert_to_memory(self, legacy_memory) -> Memory:
        from ..types import MemoryType
        return Memory(
            id=getattr(legacy_memory, 'id', ''),
            content=getattr(legacy_memory, 'content', ''),
            timestamp=getattr(legacy_memory, 'timestamp', 0),
            memory_type=MemoryType.CONVERSATION,
            importance=getattr(legacy_memory, 'importance', 0.5),
            user_id=getattr(legacy_memory, 'associated_user', None),
            emotional_valence=getattr(legacy_memory, 'emotional_valence', 0),
            emotional_arousal=getattr(legacy_memory, 'emotional_arousal', 0),
            tags=getattr(legacy_memory, 'tags', []) or [],
        )

    def _update_stats(self, elapsed_ms: float):
        n = self.stats["total_searches"]
        avg = self.stats["average_time_ms"]
        self.stats["average_time_ms"] = ((avg * (n - 1)) + elapsed_ms) / n


# =========================================================================
# SINGLETON / FACTORY
# =========================================================================

_instance: Optional[IRIS] = None


def get_iris(db_path: str = _DEFAULT_DB_PATH,
             storage=None, legacy_iris=None, fast_path=None,
             session_store=None) -> IRIS:
    """Get or create the IRIS singleton."""
    global _instance
    if _instance is None:
        _instance = IRIS(
            db_path=db_path,
            storage=storage,
            legacy_iris=legacy_iris,
            fast_path=fast_path,
            session_store=session_store,
        )
    return _instance


def _reset_iris():
    """Reset the IRIS singleton (for testing)."""
    global _instance
    _instance = None


__all__ = ['IRIS', 'IRISConfig', 'get_iris', '_reset_iris']
