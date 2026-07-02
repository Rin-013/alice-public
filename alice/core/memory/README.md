# Memory (IRIS)

> **Status**: Production (June 2026)
> **Retrieval brain**: IRIS — tiered session → total with ACT-R re-ranking, LLM rerank, and per-turn telemetry
> **Self-improvement**: usefulness-labeled feedback, contextual bandit over ACT-R weights, session-end compression, Mind training pipeline
> **Primary interface**: `AliceMemorySystem` (`memory.py`)
> **Persistence**: SQLite + dual FAISS (conversation + fact) + recall JSONL telemetry

IRIS is Alice's whole memory pipeline — storage, retrieval, ranking, and the
learning loops that tune retrieval from her own lived turns. Everything below
is feature-flagged; the default path is always safe.

---

## 3-Tier Architecture (unchanged)

```
Tier 1 — In-context
    conversation_history[] always injected into the prompt.
    ShortTermMemory — last N exchanges.

Tier 2 — Session store (in-memory FAISS)
    SessionMemoryStore — current session only, <1ms lookup, no disk writes.
    add_conversation() writes here. Session hits get a +0.20 ACT-R boost
    so they rank above equivalent-relevance historical memories.

Tier 3 — Total memory (SQLite + persistent FAISS)
    LongTermMemory — all sessions, survives restarts.
    end_session() flushes Tier 2 → Tier 3 and runs the Curator passes.

Always-on
    get_pinned_context() — top-N FACT memories always in the system prompt.
    Never requires retrieval; Alice always knows Rin's core identity facts.
```

### ACT-R Scoring (Tier 3, default weights)

```
score = relevance  × 0.50      # cosine / keyword match
      + importance × 0.25      # write-time importance score (boosted by feedback)
      + recency    × 0.15      # 0.990^hours_since_last_access, half-life ≈ 2.8 days
      + frequency  × 0.10      # log1p(access_count) / 3.0
```

Once a memory has accumulated ≥5 usefulness observations (Phase 4 warmup),
the scorer rebalances: frequency loses 0.10 to a new `usefulness_ema` term,
so memories that *actually helped* outrank memories that are just frequently
touched.

---

## File Map

```
memory/
├── memory.py                    # AliceMemorySystem — main facade
├── types.py                     # Memory, MemoryType, SearchResult, etc.
├── importance.py                # Write-time rule-based importance scorer
├── distiller.py                 # SessionDistiller — end-of-session LLM fact extraction
├── recall_gate.py               # Heuristic + optional model gate on per-turn search
├── curator.py                   # End-of-session narrative stitch + compression
├── freshness_narration.py       # "hazy memory, but..." age-aware framing
├── modification_detector.py     # Notices when a fact just changed ("wait, didn't you say...")
├── telemetry.py                 # Per-turn recall JSONL log (foundation for self-learning)
├── usefulness.py                # Cosine-based memory-used scoring (replaces word-overlap)
├── wiring.py                    # Glues memory-hint injection, framing, etc.
├── contradiction.py             # Supersede old facts when a new one conflicts
├── seed_cartridge.py            # Identity cartridge seeder (idempotent, from cartridge_alice.yaml)
├── emotional_tagging.py, trauma.py, divergence.py, smart_facts.py   # subsystems
├── oracle.py                    # Entity detector used by Curator + wiring
│
├── iris/
│   ├── search.py                # IRIS — tiered search + ACT-R + bandit hook + telemetry + depth records
│   ├── context.py               # Identity context (get_identity_context, render_identity)
│   ├── llm_ranker.py            # Mind-based semantic rerank (filter)
│   ├── fast_path.py             # Legacy re-export shim (expected-absent, gated behind ALICE_DEBUG)
│   ├── semantic.py              # Legacy re-export shim
│   └── strategy_bandit.py       # LinUCB over ACT-R weight tuples
│
├── storage/
│   ├── short_term.py            # Tier 1 — recent exchange list
│   ├── session.py               # Tier 2 — in-memory FAISS
│   ├── long_term.py             # Tier 3 — SQLite + dual FAISS (migrations live here)
│   ├── akashic.py               # Akashic records — objective immutable bank (depth layer)
│   ├── index_system.py          # IndexSystem — Alice's subjective memory copy (depth layer)
│   └── vector.py                # FAISSMemoryIndex — conversation + fact indices
```

---

## Retrieval Pipeline (one turn)

```
user query ──► recall_gate (heuristic/model) ──► IRIS.search()
                                                       │
                       ┌───────────────────────────────┼─────────────────────────┐
                       ▼                                                         ▼
              Tier 2: session FAISS                                   Tier 3: total FAISS
              (in-memory, <1ms)                                       (SQLite-backed)
                       │                                                         │
                       └───────────────────────────────┬─────────────────────────┘
                                                       ▼
                                         ACT-R re-rank (weights from bandit)
                                                       │
                                                       ▼
                                        Mind LLM rerank  (optional, flag)
                                                       │
                                                       ▼
                                           top-k memories + framing
                                                       │
                                                       ▼
                              recall_gate.fetch() → system prompt context
```

After Alice responds:
- `usefulness.score_usefulness(content, response)` → cosine score per recalled memory.
- Scores ≥ `COSINE_USED_THRESHOLD` (0.55) bump importance and set `record_memory_used`.
- Telemetry flushes the whole turn (query, candidates, picks, usefulness, bandit arm, latency) as one JSONL line.
- Bandit receives `mean(usefulness)` as reward for the chosen ACT-R weight arm.

---

## Self-Improvement Layer

Every piece below is **off by default**. Flip the flag to enable; bugs fail
open and Alice keeps talking.

| Feature | Flag | What it does |
|---------|------|--------------|
| Recall telemetry | `ALICE_TELEMETRY=1` (default on) | Append-only JSONL of every IRIS search + outcome |
| LLM rerank (Mind) | `ALICE_IRIS_LLM_RERANK=1` | Mind picks relevant memories from ACT-R shortlist |
| LLM usefulness judge | `ALICE_IRIS_LLM_USEFULNESS=1` | Mind scores ambiguous cosine band [0.35, 0.65] |
| Session-end compression | `ALICE_IRIS_COMPRESSION=1` | Curator 4th phase — cluster and digest old memories |
| ACT-R weight bandit | `ALICE_IRIS_BANDIT=1` | LinUCB picks one of 4 weight tuples per turn |

### Recall Telemetry (`telemetry.py`)

Per-turn append-only JSONL at `alice/data/turn_logs/recalls-YYYY-MM-DD.jsonl`.
One line per turn, containing:

```json
{
  "turn_id": "a1b2c3d4",         "ts": "2026-04-16T10:00:00Z",
  "user_id": "rin",            "query": "what's my cat's name",
  "strategy_id": 0,               "strategy_features": [...],
  "tier": "total",                "reranked": true,
  "candidates": [ ...pre-rerank pool with per-memory scores... ],
  "picked":   [ "mem_..." ],
  "used":     [ "mem_..." ],
  "usefulness":   { "mem_...": 0.82 },
  "word_overlap": { "mem_...": 0.05 },   // parallel log of the old signal
  "response_len": 142,
  "latency_ms": 58.3
}
```

Foundation for everything else. Disable via `ALICE_TELEMETRY=0` if disk is
tight, but expect the learning loops to go dark too.

### Usefulness Signal (`usefulness.py`)

Replaces the old word-overlap heuristic. Uses the shared MiniLM embedding to
compute cosine similarity between each recalled memory and Alice's response.

- `COSINE_USED_THRESHOLD = 0.55` — binary "used?" floor.
- Ambiguous band `[0.35, 0.65]` → optional Mind yes/partial/no judge
  (`ALICE_IRIS_LLM_USEFULNESS=1`) runs post-TTS on Mind's idle cycle.
- Per-memory EMA (`usefulness_ema`, α=0.2) stored on `memories`; feeds the
  ACT-R scorer once a memory has `usefulness_n >= 5`.
- Word-overlap is still logged in parallel in telemetry for regression
  comparison — we kill it once cosine proves stable.

### Curator (`curator.py`) — end-of-session passes

Runs when `memory.end_session()` fires. Four phases, each independent:

1. **Importance boost** — memories Alice used get +0.05 importance.
2. **Narrative stitch** — fact + episodic memories linked by shared entities
   become a single compound `FACT` memory.
3. **Contradiction cleanup** — superseded memories get decayed.
4. **Active compression** *(new, flag-gated)* — memories >30 days old that
   share ≥1 entity and fall within a ±7 day window are clustered (≥3 members)
   and digested into a single `SEMANTIC`/`FACT` memory. Sources are demoted
   one decay rung (`ACTIVE → COOL → COLD → ARCHIVED`, never `PURGED`) and
   stamped with `compression_parent = digest_id` for back-traceability.

Oracle's entity detection path: `oracle.entity_detector.detect_entities(text, context)`.
When Oracle's regex misses entities (it's trigger-word biased), the Curator
falls back to a proper-noun extractor with a stop-list.

### Modification Detector + Freshness (`modification_detector.py`, `freshness_narration.py`)

Per-memory framing attached by `wiring.py` (and mirrored in `recall_gate.py`):

- Modification detector — spots that the incoming fact supersedes a stored
  one; Alice gets a "wait, didn't you say X before?" framing prefix.
- Freshness narration — age-aware prefixes for stale memories ("it's been a
  while, but..."). Modification wins over age — surprise > staleness.

### ACT-R Weight Bandit (`iris/strategy_bandit.py`)

LinUCB over 4 arms (weight tuples for `rel, imp, rec, freq`):

| Arm | `rel` | `imp` | `rec` | `freq` | Profile |
|-----|-------|-------|-------|--------|---------|
| 0   | 0.50  | 0.25  | 0.15  | 0.10   | Production — safety floor |
| 1   | 0.60  | 0.20  | 0.10  | 0.10   | Relevance-heavy (paraphrase) |
| 2   | 0.40  | 0.30  | 0.20  | 0.10   | Importance-heavy (identity facts) |
| 3   | 0.45  | 0.15  | 0.15  | 0.25   | Frequency-heavy (revisited topics) |

Features: `[query_len_norm, has_question_mark, has_entity, session_turn_idx_norm]`.

Safety properties:
- **Arm 0 floor (20%)** — 20% of pulls always go to production weights, so
  the worst case is "learning stalled," not "week-1 regression."
- **ε-greedy cold start** — ε decays 0.30 → 0.05 over first 200 turns.
- **Short-reply filter** — turns with `response_len < 30` skip reward updates
  (the usefulness label is too noisy to learn from).
- **Fail-open** — any math or IO failure reverts to Arm 0 without affecting
  the turn. Persists to `alice/data/databases/alice_retrieval.db` (separate
  from the memory DB so bandit churn can't damage memories).

Reward = mean usefulness across picked memories this turn. The reward
update lives in `chat.py` post-response and reads the arm + feature vector
back from thread-local telemetry (`telemetry.current_strategy()`).

---

## Importance Scorer

Rule-based, fires at write time. ACT-R importance term (25%) depends on it.

| Category | Score | Examples |
|----------|-------|----------|
| Identity | 0.90 | name, age, location, job |
| Relationships | 0.85 | family, pets, partner |
| Goals / projects | 0.75 | building Alice, career goals |
| Preferences | 0.70 | favorite games, hobbies |
| Past events | 0.60 | where Rin used to live |
| Opinions | 0.50 | beliefs, views |
| Trivial | −0.20 penalty | "right now", "today", "lol" |

Feedback loop bumps it over time: each `record_memory_used()` adds +0.05
(capped at 1.0). Memories that consistently help rise; memories that are
retrieved but ignored stay flat.

---

## Recall Gate (`recall_gate.py`)

Two-layer gate deciding whether to run IRIS search for a given message:

1. **Heuristic regex** (<1ms) — catches obvious "do you remember…" forms.
2. **Optional model gate** — Mind-based classifier for the gray zone.

Returns an empty string when the turn doesn't warrant a search (don't
waste Mind time on "hi"). Framing prefixes (modification notice > age
prefix) are attached inline here and in `wiring.py` so Alice always sees
the same context shape.

---

## Usage (from the chat loop)

```python
from alice.core.memory.memory import get_alice_memory
from alice.core.memory import telemetry

memory = get_alice_memory()
memory.start_session("rin", "Rin")

# Telemetry: open a turn buffer so IRIS can log candidates/picks into it.
turn_id = telemetry.start_turn("rin", user_input)

# Build prompt — always-on facts + on-demand recall.
pinned = memory.get_pinned_context(n=10)
recalled = gate.fetch("what is rin's cat")

# Generate response, store the turn.
memory.add_conversation(user_message, alice_response)

# Feedback: cosine usefulness replaces word-overlap. `record_usefulness`
# updates each memory's EMA; `record_memory_used` bumps importance above
# the threshold.
from alice.core.memory import usefulness as _use
for r in retrieved:
    score = _use.score_usefulness(r.memory.content, alice_response)
    if score is None:
        continue
    memory.long_term.record_usefulness(r.memory.id, score)
    if score >= _use.COSINE_USED_THRESHOLD:
        memory.record_memory_used(r.memory.id)

telemetry.record_usage(used=[...], usefulness={...}, response_len=len(response))
telemetry.finalize_turn(latency_ms=...)

# At session end — triggers the Curator passes (compression if flag set).
summary = memory.end_session()
```

---

## Persistence

- **SQLite**: `alice/data/databases/alice_memory.db` (canonical path).
- **FAISS**: `alice_memory_faiss_conversation.faiss` + `alice_memory_faiss_fact.faiss`, saved every 10 writes and at `end_session()`.
- **FAISS mappings**: `alice_memory_faiss_mappings.pkl`.
- **Bandit state**: `alice/data/databases/alice_retrieval.db` (separate DB so bandit writes can't corrupt memories).
- **Telemetry**: `alice/data/turn_logs/recalls-YYYY-MM-DD.jsonl` (daily rotation).

FAISS indices load automatically on startup; missing indices are rebuilt
from SQLite.

---

## Deduplication

Before any SQLite INSERT, FAISS cosine similarity ≥ 0.92 against existing
vectors drops near-duplicates silently. Stops the same fact from
accumulating dozens of reworded copies over many sessions.

---

## Testing

```bash
python tests/memory/test_memory_full.py   # 114-assertion suite
```

Covers: importance scorer (all categories + edge cases), session store,
long-term storage, pinned context, IRIS tiered search, IRIS feedback loop,
distiller JSON parser, contradiction supersede, depth layer, and full session lifecycle.

Phase-specific verification is documented in the rollout plan at
`.claude/plans/tranquil-scribbling-biscuit.md` (lives per-machine).

---

**Last updated**: June 19, 2026
**Baseline**: 114/114 memory test assertions
