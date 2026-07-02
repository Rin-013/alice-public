"""
ACT-R Weight Bandit
===================

A LinUCB contextual bandit that picks which weight tuple to use when scoring
total-memory hits. Four arms — each is a plausible configuration of the
four ACT-R factors (relevance, importance, recency, frequency). Arm 0 is
the current production tuple, so the worst case is "we never learn anything"
rather than "week-1 regression."

Features per decision (pre-normalized to [0, 1]):
  0: query_len_norm       — len(query)/120, clipped
  1: has_question_mark    — 1 if '?' present
  2: has_entity           — 1 if Oracle-ish proper noun present
  3: session_turn_idx     — current session turn index / 20, clipped

Update loop lives in chat.py after the turn completes: `reward` is the mean
cosine-usefulness across the picked memories (Phase 4 signal). Turns with
short replies (<30 chars) are skipped — too-noisy labels corrupt the arms.

State is persisted to `alice/data/databases/alice_retrieval.db` (new DB —
kept separate from the memory DB so bandit churn can't damage memories).
Every arm owns an A matrix and b vector; UCB uses them to compute a score
per arm and picks the argmax. Fails open: if anything blows up, we return
Arm 0.

Gated by `ALICE_IRIS_BANDIT=1`. Disabled by default until Phase 4 has
accumulated enough usefulness labels to produce stable rewards.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import random
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_FLAG_ENV = "ALICE_IRIS_BANDIT"

# Arm 0 MUST remain the production tuple (rel, imp, rec, freq). Any change
# here breaks the "cold-start safety floor" contract.
ARMS: List[Tuple[float, float, float, float]] = [
    (0.50, 0.25, 0.15, 0.10),   # Arm 0 — current production
    (0.60, 0.20, 0.10, 0.10),   # Arm 1 — relevance-heavy (paraphrase queries)
    (0.40, 0.30, 0.20, 0.10),   # Arm 2 — importance-heavy (personal facts)
    (0.45, 0.15, 0.15, 0.25),   # Arm 3 — frequency-heavy (revisited topics)
]

_N_FEATURES = 4
_UCB_ALPHA = 1.0                # exploration strength

# Cold-start: ε-greedy decays from 0.3 → 0.05 over first 200 turns. This
# keeps exploration alive early without bleeding performance forever.
_EPSILON_START = 0.30
_EPSILON_END = 0.05
_EPSILON_DECAY_TURNS = 200

# Arm 0 safety floor — even after the bandit has learned, give it 20% of
# pulls so ranking never drifts far from production baseline.
_ARM0_FLOOR = 0.20

_DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "databases" / "alice_retrieval.db"
)

_lock = threading.Lock()

# Thread-local "pending pick" so chat.py can issue the reward update without
# plumbing arm_id/features through every signature — and without depending
# on telemetry being enabled. Decoupling matters: ALICE_TELEMETRY=0 with
# ALICE_IRIS_BANDIT=1 should still learn.
_tls = threading.local()


def is_enabled() -> bool:
    val = os.environ.get(_FLAG_ENV, "0").strip().lower()
    return val in ("1", "true", "yes", "on")


@dataclass
class _ArmState:
    """LinUCB per-arm state. A ∈ R^{d x d}, b ∈ R^d."""
    arm_id: int
    A: "any"  # numpy array
    b: "any"
    n_pulls: int = 0


class StrategyBandit:
    """LinUCB bandit over ACT-R weight configurations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._arms: Dict[int, _ArmState] = {}
        self._total_pulls: int = 0
        self._init_db()
        self._load_state()

    # ---- persistence -----------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS bandit_arms (
                    arm_id INTEGER PRIMARY KEY,
                    A_matrix BLOB,
                    b_vector BLOB,
                    n_pulls INTEGER,
                    updated_at REAL
                )"""
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"bandit: init_db failed: {e}")

    def _load_state(self) -> None:
        try:
            import numpy as np
        except ImportError:
            logger.debug("bandit: numpy unavailable; running disabled")
            return
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT arm_id, A_matrix, b_vector, n_pulls FROM bandit_arms")
            rows = cur.fetchall()
            conn.close()

            for arm_id, A_blob, b_blob, n_pulls in rows:
                try:
                    A = pickle.loads(A_blob)
                    b = pickle.loads(b_blob)
                    self._arms[arm_id] = _ArmState(
                        arm_id=arm_id, A=A, b=b, n_pulls=int(n_pulls or 0),
                    )
                    self._total_pulls += int(n_pulls or 0)
                except Exception as e:
                    logger.debug(f"bandit: load arm {arm_id} failed: {e}")

            # Seed missing arms with identity A and zero b.
            for arm_id in range(len(ARMS)):
                if arm_id not in self._arms:
                    self._arms[arm_id] = _ArmState(
                        arm_id=arm_id,
                        A=np.eye(_N_FEATURES),
                        b=np.zeros(_N_FEATURES),
                        n_pulls=0,
                    )
        except Exception as e:
            logger.debug(f"bandit: load_state failed: {e}")

    def _save_arm(self, arm: _ArmState) -> None:
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO bandit_arms (arm_id, A_matrix, b_vector, n_pulls, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(arm_id) DO UPDATE SET
                     A_matrix = excluded.A_matrix,
                     b_vector = excluded.b_vector,
                     n_pulls = excluded.n_pulls,
                     updated_at = excluded.updated_at""",
                (
                    arm.arm_id,
                    pickle.dumps(arm.A),
                    pickle.dumps(arm.b),
                    int(arm.n_pulls),
                    time.time(),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"bandit: save_arm {arm.arm_id} failed: {e}")

    # ---- decision --------------------------------------------------------

    def _epsilon(self) -> float:
        if self._total_pulls >= _EPSILON_DECAY_TURNS:
            return _EPSILON_END
        frac = self._total_pulls / max(1, _EPSILON_DECAY_TURNS)
        return _EPSILON_START + (_EPSILON_END - _EPSILON_START) * frac

    def pick_strategy(self, features: List[float]) -> int:
        """
        Choose an arm given a feature vector. Returns arm_id in [0, len(ARMS)).
        Falls back to Arm 0 on any numerical or IO failure.

        Side effect: stashes (arm_id, features) in a thread-local so
        `consume_pending()` can pair the reward with the pick later. The
        stash is overwritten on the next pick — if two searches run in one
        turn without an interleaved reward, we reward only the latest arm.
        """
        if not is_enabled():
            return 0
        try:
            import numpy as np
        except ImportError:
            return 0

        with _lock:
            # Arm 0 safety floor — random 20% of the time pick production.
            if random.random() < _ARM0_FLOOR:
                arm_id = 0
            elif random.random() < self._epsilon():
                # ε-greedy exploration during cold-start.
                arm_id = random.randrange(len(ARMS))
            else:
                x = np.asarray(
                    features[:_N_FEATURES] + [0.0] * max(0, _N_FEATURES - len(features)),
                    dtype=float,
                )
                best_arm, best_score = 0, -math.inf
                for aid, arm in self._arms.items():
                    try:
                        A_inv = np.linalg.inv(arm.A)
                        theta = A_inv @ arm.b
                        mean = float(theta @ x)
                        var = float(x @ A_inv @ x)
                        score = mean + _UCB_ALPHA * math.sqrt(max(0.0, var))
                    except Exception:
                        continue
                    if score > best_score:
                        best_score = score
                        best_arm = aid
                arm_id = best_arm

        # Outside the lock: stash the pending pick. If this fails, we only
        # lose the reward attribution for this turn — never the pick itself.
        try:
            _tls.pending = {"arm_id": int(arm_id), "features": list(features)}
        except Exception:
            pass
        return arm_id

    def update_reward(self, arm_id: int, features: List[float], reward: float) -> None:
        """
        LinUCB update: A += x x^T, b += reward * x. Safe to call with reward=0
        (just reinforces "this arm saw this context"); skip entirely if the
        turn was too short to score (caller enforces the filter).
        """
        if not is_enabled():
            return
        if arm_id not in self._arms:
            return
        try:
            import numpy as np
        except ImportError:
            return

        try:
            x = np.asarray(features[:_N_FEATURES] + [0.0] * max(0, _N_FEATURES - len(features)),
                           dtype=float)
            r = float(max(0.0, min(1.0, reward)))
        except Exception:
            return

        with _lock:
            arm = self._arms[arm_id]
            try:
                arm.A = arm.A + np.outer(x, x)
                arm.b = arm.b + r * x
                arm.n_pulls += 1
                self._total_pulls += 1
            except Exception as e:
                logger.debug(f"bandit: update math failed: {e}")
                return
            self._save_arm(arm)


# ---- singleton + feature extraction -------------------------------------

_instance: Optional[StrategyBandit] = None


def get_bandit() -> StrategyBandit:
    global _instance
    if _instance is None:
        _instance = StrategyBandit()
    return _instance


def _reset_bandit() -> None:
    """Testing helper — wipe in-process state. Does NOT delete the DB."""
    global _instance
    _instance = None
    try:
        _tls.pending = None
    except Exception:
        pass


def consume_pending() -> Optional[Dict[str, object]]:
    """
    Return and CLEAR the current thread's pending pick (if any). Called by
    chat.py after the turn is scored so the reward update matches the
    arm+features that produced the recall. Returns None when no pick is
    pending (bandit disabled, never called, or already consumed).
    """
    try:
        pending = getattr(_tls, "pending", None)
        if pending is None:
            return None
        _tls.pending = None
        return pending
    except Exception:
        return None


def clear_pending() -> None:
    """Drop any pending pick without consuming (for aborted turns)."""
    try:
        _tls.pending = None
    except Exception:
        pass


def featurize(query: str, session_turn_idx: int = 0) -> List[float]:
    """Build the feature vector from a raw query + session state."""
    q = query or ""
    q_len = min(1.0, len(q) / 120.0)
    has_q = 1.0 if "?" in q else 0.0
    # Cheap entity proxy: any capitalized alpha token (2+ chars) anywhere
    # after position 0. Strip trailing punctuation so "Pixel?" still registers.
    has_ent = 0.0
    tokens = [t.strip(".,?!;:'\"") for t in q.split()]
    for tok in tokens[1:]:
        if len(tok) >= 2 and tok[0].isupper() and tok.isalpha():
            has_ent = 1.0
            break
    if has_ent == 0.0 and len(tokens) == 1 and tokens[0] \
            and len(tokens[0]) >= 2 and tokens[0][0].isupper() and tokens[0].isalpha():
        # Single-entity queries like "Pixel?" still should register.
        has_ent = 1.0
    idx_norm = min(1.0, max(0, session_turn_idx) / 20.0)
    return [q_len, has_q, has_ent, idx_norm]


def weights_for(arm_id: int) -> Tuple[float, float, float, float]:
    """Look up the (rel, imp, rec, freq) tuple for an arm id. Falls back to Arm 0."""
    try:
        return ARMS[arm_id]
    except (IndexError, TypeError):
        return ARMS[0]


__all__ = [
    "is_enabled",
    "StrategyBandit",
    "get_bandit",
    "consume_pending",
    "clear_pending",
    "featurize",
    "weights_for",
    "ARMS",
]
