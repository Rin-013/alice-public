# Copyright 2025 Rin - Alice AI System
"""
Divergence Detection — 12-factor retrospective memory analysis.

Rewritten for IRIS. Operates directly on long_term SQLite tables.
Identifies life-shaping events via retrospective divergence scoring.
"""

import json
import time
import sqlite3
from typing import List, Dict, Any, Optional


class _IndexMemoryRow:
    """Lightweight wrapper for index_memories row data."""
    __slots__ = (
        'event_id', 'content', 'depth', 'emotion_intensity', 'uniqueness',
        'contrast', 'access_count', 'edit_resistance', 'emotional_markers',
        'emotional_mismatch_score',
    )

    def __init__(self, row: dict):
        self.event_id = row.get('event_id', '')
        self.content = row.get('content', '')
        self.depth = row.get('depth', 'surface')
        self.emotion_intensity = row.get('emotion_intensity', 0.0) or 0.0
        self.uniqueness = row.get('uniqueness', 0.0) or 0.0
        self.contrast = row.get('contrast', 0.0) or 0.0
        self.access_count = row.get('access_count', 0) or 0
        self.edit_resistance = row.get('edit_resistance', 0.0) or 0.0
        markers = row.get('emotional_markers', '[]')
        try:
            self.emotional_markers = json.loads(markers) if markers else []
        except (json.JSONDecodeError, TypeError):
            self.emotional_markers = []
        self.emotional_mismatch_score = row.get('emotional_mismatch_score', 0.0) or 0.0


class DivergenceDetector:
    """
    12-factor retrospective divergence analysis.

    Scores memories on 12 factors to identify life-shaping divergence events.
    Runs at end_session() to review scheduled memories.
    """

    def __init__(self, iris):
        """
        Args:
            iris: IRIS instance
        """
        self.iris = iris

    def _connect(self):
        """Get DB connection via long_term (handles :memory: correctly)."""
        return self.iris.long_term._connect()

    def _close(self, conn):
        """Close connection (skip for :memory:)."""
        if self.iris.long_term.db_path != ":memory:":
            conn.close()

    def _get_memory(self, event_id: str) -> Optional[_IndexMemoryRow]:
        """Fetch an index_memories row as an _IndexMemoryRow."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM index_memories WHERE event_id = ?', (event_id,))
        row = cursor.fetchone()
        self._close(conn)
        if row:
            return _IndexMemoryRow(dict(row))
        return None

    def _get_memories_in_timeframe(self, user_id: str, start: float, end: float) -> List[_IndexMemoryRow]:
        """Fetch index_memories rows in a time window."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM index_memories
            WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
        ''', (user_id, start, end))
        rows = [_IndexMemoryRow(dict(r)) for r in cursor.fetchall()]
        self._close(conn)
        return rows

    def _get_choices_for_event(self, event_id: str) -> List[Dict]:
        """Fetch choice_ledger rows for an event."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM choice_ledger WHERE related_event_id = ?', (event_id,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        self._close(conn)
        return rows

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    def run_retrospective_divergence_review(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Review memories scheduled for divergence analysis.

        Returns list of divergent events detected.
        """
        if not self.iris.current_user:
            return []

        user_id = self.iris.current_user.user_id
        now = time.time()

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT event_id, review_scheduled FROM index_memories
            WHERE user_id = ? AND review_scheduled IS NOT NULL
            AND review_scheduled <= ? AND divergence_flag = FALSE
        ''', (user_id, now))
        scheduled = cursor.fetchall()
        self._close(conn)

        events = []
        for event_id, scheduled_time in scheduled:
            score = self._calculate_retrospective_divergence(event_id, scheduled_time)
            if score > 0.6:
                self._mark_as_divergence_event(event_id, score)
                events.append({
                    "event_id": event_id,
                    "divergence_score": score,
                    "review_date": scheduled_time,
                })

        return events

    # ========================================================================
    # 12-FACTOR ANALYSIS
    # ========================================================================

    def _calculate_retrospective_divergence(self, event_id: str, event_time: float) -> float:
        """12-factor divergence scoring."""
        if not self.iris.current_user:
            return 0.0

        mem = self._get_memory(event_id)
        if not mem:
            return 0.0

        user_id = self.iris.current_user.user_id
        window_end = event_time + (7 * 24 * 3600)
        choices = self._get_choices_for_event(event_id)

        c = {}
        # Core 4
        c['mood_shift'] = self._calculate_mood_baseline_shift(event_time, window_end) * 0.15
        c['access_frequency'] = min(mem.access_count / 10, 1.0) * 0.10
        c['uniqueness'] = mem.uniqueness * 0.10
        c['choice_cascade'] = min(len(choices) * 0.1, 0.15) * 0.10

        # Advanced 8
        c['personality_shift'] = self._calc_personality_shift(choices, user_id, event_time, window_end) * 0.12
        c['social_ripple'] = self._calc_social_impact(mem, user_id, event_time, window_end) * 0.08
        c['creative_breakthrough'] = self._calc_creative_impact(mem, user_id, event_time) * 0.10
        c['value_conflict'] = self._calc_value_conflict(mem) * 0.08
        c['behavioral_pattern'] = self._calc_behavioral_change(user_id, event_time, window_end) * 0.10
        c['relationship_evolution'] = self._calc_relationship_evolution(mem) * 0.07
        c['emotional_echo'] = self._calc_emotional_echo(mem) * 0.08
        c['memory_cluster'] = self._calc_memory_cluster(mem) * 0.05

        return min(sum(c.values()), 1.0)

    # ========================================================================
    # FACTOR IMPLEMENTATIONS
    # ========================================================================

    def _calculate_mood_baseline_shift(self, start: float, end: float) -> float:
        """Mood baseline shift (stub — needs mood tracking)."""
        return 0.4

    def _calc_personality_shift(self, choices: List[Dict], user_id: str,
                                event_time: float, window_end: float) -> float:
        """Personality shift from choices + emotional pattern change."""
        total = 0.0
        for ch in choices:
            pd = ch.get('personality_delta')
            if pd:
                try:
                    deltas = json.loads(pd) if isinstance(pd, str) else pd
                    total += min(sum(abs(v) for v in deltas.values()), 1.0)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

        # Emotional pattern shift via direct SQL (replaces EmotionalAnalysis import)
        pre_mems = self._get_memories_in_timeframe(user_id, event_time - 7*86400, event_time)
        post_mems = self._get_memories_in_timeframe(user_id, event_time, window_end)

        pre_avg = (sum(m.emotion_intensity for m in pre_mems) / len(pre_mems)) if pre_mems else 0.0
        post_avg = (sum(m.emotion_intensity for m in post_mems) / len(post_mems)) if post_mems else 0.0
        total += abs(post_avg - pre_avg) * 0.5

        return min(total, 1.0)

    def _calc_social_impact(self, mem: _IndexMemoryRow, user_id: str,
                            event_time: float, window_end: float) -> float:
        """Social ripple effects."""
        pre = self._get_memories_in_timeframe(user_id, event_time - 7*86400, event_time)
        post = self._get_memories_in_timeframe(user_id, event_time, window_end)

        pre_freq = len(pre) / 7.0
        post_freq = len(post) / max((window_end - event_time) / 86400, 1.0)
        change = abs(post_freq - pre_freq) / max(pre_freq, 1.0) if pre_freq > 0 else 0.0

        milestones = ["relationship", "trust", "friendship", "closer", "distance", "conflict"]
        if any(m in mem.content.lower() for m in milestones):
            change += 0.3

        return min(change, 1.0)

    def _calc_creative_impact(self, mem: _IndexMemoryRow, user_id: str,
                              event_time: float) -> float:
        """Creative breakthrough detection."""
        markers = [
            "breakthrough", "insight", "realization", "epiphany", "understanding",
            "creative", "innovative", "new perspective", "paradigm", "revelation",
        ]
        score = sum(0.15 for m in markers if m in mem.content.lower())

        if mem.emotion_intensity > 0.7 and mem.uniqueness > 0.6:
            score += 0.4

        # Creative follow-ups in next 7 days
        followups = self._get_memories_in_timeframe(
            user_id, event_time, event_time + 7*86400
        )
        score += min(sum(1 for f in followups if f.uniqueness > 0.7) * 0.1, 0.3)

        return min(score, 1.0)

    def _calc_value_conflict(self, mem: _IndexMemoryRow) -> float:
        """Value conflict / moral dilemma detection."""
        markers = [
            "conflict", "dilemma", "torn", "struggle", "contradiction",
            "values", "ethics", "moral", "right", "wrong", "should",
        ]
        score = sum(0.1 for m in markers if m in mem.content.lower())

        if mem.contrast > 0.7:
            score += 0.4

        conflict_emotions = ["confusion", "anxiety", "guilt", "uncertainty"]
        score += sum(0.15 for e in conflict_emotions if e in mem.emotional_markers)

        return min(score, 1.0)

    def _calc_behavioral_change(self, user_id: str, event_time: float,
                                window_end: float) -> float:
        """Behavioral pattern changes (tone + response length)."""
        pre = self._get_memories_in_timeframe(user_id, event_time - 7*86400, event_time)
        post = self._get_memories_in_timeframe(user_id, event_time, window_end)

        # Tone
        pre_tone = self._tone_score(pre)
        post_tone = self._tone_score(post)
        tone_shift = abs(post_tone - pre_tone)

        # Response length
        pre_len = (sum(len(m.content) for m in pre) / len(pre)) if pre else 100.0
        post_len = (sum(len(m.content) for m in post) / len(post)) if post else 100.0
        len_change = abs(post_len - pre_len) / max(pre_len, 100)

        return min(tone_shift + len_change * 0.5, 1.0)

    def _calc_relationship_evolution(self, mem: _IndexMemoryRow) -> float:
        """Relationship milestone detection."""
        markers = [
            "first time", "never told", "personal", "vulnerable", "trust",
            "friendship", "closer", "understanding", "connection", "bond",
        ]
        score = sum(0.15 for m in markers if m in mem.content.lower())

        if mem.emotion_intensity > 0.6 and any(
            e in mem.emotional_markers for e in ["trust", "connection", "intimacy"]
        ):
            score += 0.4

        return min(score, 1.0)

    def _calc_emotional_echo(self, mem: _IndexMemoryRow) -> float:
        """Long-term emotional echo strength."""
        score = 0.0
        if mem.emotion_intensity > 0.7:
            score += 0.4
        if mem.emotional_mismatch_score:
            score += mem.emotional_mismatch_score * 0.3
        if len(mem.emotional_markers) >= 3:
            score += 0.2
        if mem.edit_resistance > 0.7:
            score += 0.2
        return min(score, 1.0)

    def _calc_memory_cluster(self, mem: _IndexMemoryRow) -> float:
        """Memory cluster formation detection."""
        score = min(mem.access_count / 15, 0.6)
        if mem.uniqueness > 0.7 and mem.access_count > 5:
            score += 0.4
        return min(score, 1.0)

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _tone_score(self, memories: List[_IndexMemoryRow]) -> float:
        """0=negative, 1=positive tone from emotional markers."""
        if not memories:
            return 0.5
        pos = ["joy", "excitement", "love", "gratitude", "connection"]
        neg = ["sadness", "anger", "frustration", "confusion", "anxiety"]
        p = n = 0
        for m in memories:
            for mk in m.emotional_markers:
                if mk in pos:
                    p += 1
                elif mk in neg:
                    n += 1
        total = p + n
        return p / total if total else 0.5

    def _mark_as_divergence_event(self, event_id: str, score: float):
        """Mark memory as divergence point."""
        conn = self._connect()
        cursor = conn.cursor()
        div_id = f"div_{int(time.time())}"

        cursor.execute('''
            UPDATE index_memories
            SET divergence_flag = TRUE, divergence_id = ?,
                divergence_impact = ?, depth = 'core',
                edit_resistance = MIN(0.98, edit_resistance + 0.5),
                decay_floor = ?
            WHERE event_id = ?
        ''', (div_id, score, 0.6 * score, event_id))

        cursor.execute('''
            UPDATE akashic_records
            SET divergence_flag = TRUE, divergence_id = ?, divergence_score = ?
            WHERE event_id = ?
        ''', (div_id, score, event_id))

        conn.commit()
        self._close(conn)
