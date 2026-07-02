# Copyright 2025 Rin - Alice AI System
"""
Trauma Quarantine — quarantine/reconcile traumatic memories.

Rewritten for IRIS. Operates directly on long_term SQLite tables.
"""

import time
import json
import sqlite3
import random
import uuid
from typing import List, Dict, Any


class TraumaQuarantine:
    """
    Traumatic memory quarantine and protection system.

    Responsibilities:
    - Quarantine traumatic memories
    - Reconcile via 4 strategies (integrate, compartmentalize, reframe, partial_suppress)
    - Implement edit resistance for important memories
    - Memory rewriting with truth pointers
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
            self._close(conn)

    def quarantine_traumatic_memory(self, event_id: str, trauma_markers: List[str],
                                    severity: float = 0.8) -> Dict[str, Any]:
        """Quarantine a potentially traumatic memory for processing."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT event_id FROM index_memories WHERE event_id = ?', (event_id,)
        )
        if not cursor.fetchone():
            self._close(conn)
            return {"error": f"Memory {event_id} not found"}

        quarantine_id = f"quarantine_{uuid.uuid4().hex[:12]}"
        now = time.time()

        cursor.execute('''
            INSERT INTO memory_quarantine
            (quarantine_id, event_id, trauma_markers, severity, quarantine_timestamp, status)
            VALUES (?, ?, ?, ?, ?, 'active')
        ''', (quarantine_id, event_id, json.dumps(trauma_markers), severity, now))

        cursor.execute('''
            UPDATE index_memories
            SET edit_resistance = ?,
                decay_floor = ?
            WHERE event_id = ?
        ''', (min(0.95, severity + 0.2), max(0.8, severity), event_id))

        # Log to akashic_records
        cursor.execute('''
            INSERT OR IGNORE INTO akashic_records
            (event_id, timestamp, who, what)
            VALUES (?, ?, ?, ?)
        ''', (f"quarantine_{quarantine_id}", now,
              json.dumps(["system"]),
              f"Memory {event_id} quarantined: markers={trauma_markers}, severity={severity}"))

        conn.commit()
        self._close(conn)

        return {
            "quarantine_id": quarantine_id,
            "event_id": event_id,
            "severity": severity,
            "trauma_markers": trauma_markers,
            "status": "quarantined",
        }

    def reconcile_quarantined_memory(self, quarantine_id: str,
                                     resolution_strategy: str = "integrate",
                                     processing_notes: str = "") -> Dict[str, Any]:
        """
        Reconcile a quarantined memory.

        Strategies: integrate, compartmentalize, reframe, partial_suppress
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT quarantine_id, event_id, trauma_markers, severity, status
            FROM memory_quarantine WHERE quarantine_id = ?
        ''', (quarantine_id,))
        row = cursor.fetchone()
        if not row:
            self._close(conn)
            return {"error": f"Quarantine record {quarantine_id} not found"}

        _, event_id, trauma_markers_json, severity, status = row
        if status != 'active':
            self._close(conn)
            return {"error": f"Quarantine {quarantine_id} not active (status: {status})"}

        result = {}

        if resolution_strategy == "integrate":
            er = max(0.3, severity * 0.4)
            df = max(0.2, severity * 0.3)
            cursor.execute('''
                UPDATE index_memories
                SET edit_resistance = ?, decay_floor = ?
                WHERE event_id = ?
            ''', (er, df, event_id))
            result["integration_successful"] = True

        elif resolution_strategy == "compartmentalize":
            cursor.execute('''
                UPDATE index_memories SET decay_floor = 0.5
                WHERE event_id = ?
            ''', (event_id,))
            result["compartmentalization_successful"] = True

        elif resolution_strategy == "reframe":
            cursor.execute('''
                UPDATE index_memories SET edit_resistance = ?
                WHERE event_id = ?
            ''', (max(0.2, severity * 0.3), event_id))
            result["reframing_successful"] = True

        elif resolution_strategy == "partial_suppress":
            cursor.execute('''
                UPDATE index_memories
                SET salience = salience * 0.4,
                    emotion_intensity = emotion_intensity * 0.5
                WHERE event_id = ?
            ''', (event_id,))
            result["suppression_successful"] = True

        # Update quarantine record
        cursor.execute('''
            UPDATE memory_quarantine
            SET status = 'resolved',
                resolution_strategy = ?,
                resolution_timestamp = ?
            WHERE quarantine_id = ?
        ''', (resolution_strategy, time.time(), quarantine_id))

        # Log
        cursor.execute('''
            INSERT OR IGNORE INTO akashic_records
            (event_id, timestamp, who, what)
            VALUES (?, ?, ?, ?)
        ''', (f"reconcile_{quarantine_id}", time.time(),
              json.dumps(["system"]),
              f"Reconciled {event_id} via {resolution_strategy}"))

        conn.commit()
        self._close(conn)

        result.update({
            "quarantine_id": quarantine_id,
            "event_id": event_id,
            "resolution_strategy": resolution_strategy,
            "status": "resolved",
        })
        return result

    def implement_memory_edit_resistance(self, event_id: str,
                                         resistance_level: float = 0.8,
                                         reason: str = "high_emotional_significance") -> Dict[str, Any]:
        """Implement edit resistance for important memories."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT event_id, edit_resistance, decay_floor, emotion_intensity, salience
            FROM index_memories WHERE event_id = ?
        ''', (event_id,))
        row = cursor.fetchone()
        if not row:
            self._close(conn)
            return {"error": f"Memory {event_id} not found"}

        _, current_resistance, current_decay_floor, emotion_intensity, salience = row

        new_resistance = min(0.95, max(current_resistance or 0, resistance_level))
        new_decay_floor = min(0.9, max(current_decay_floor or 0, resistance_level * 0.8))

        intensity_bonus = min(0.1, (emotion_intensity or 0) * 0.2)
        salience_bonus = min(0.1, (salience or 0) * 0.15)

        final_resistance = min(0.98, new_resistance + intensity_bonus + salience_bonus)
        final_decay_floor = min(0.95, new_decay_floor + (intensity_bonus + salience_bonus) * 0.5)

        cursor.execute('''
            UPDATE index_memories
            SET edit_resistance = ?, decay_floor = ?
            WHERE event_id = ?
        ''', (final_resistance, final_decay_floor, event_id))

        conn.commit()
        self._close(conn)

        return {
            "event_id": event_id,
            "edit_resistance_applied": final_resistance,
            "decay_floor_applied": final_decay_floor,
            "resistance_reason": reason,
        }

    def get_quarantine_status(self) -> Dict[str, Any]:
        """Get current quarantine system status."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM memory_quarantine WHERE status = "active"')
        active = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM memory_quarantine WHERE status = "resolved"')
        resolved = cursor.fetchone()[0]

        cursor.execute('''
            SELECT quarantine_id, event_id, severity, trauma_markers, status, quarantine_timestamp
            FROM memory_quarantine ORDER BY quarantine_timestamp DESC LIMIT 5
        ''')
        recent = []
        for r in cursor.fetchall():
            recent.append({
                "quarantine_id": r[0], "event_id": r[1], "severity": r[2],
                "trauma_markers": json.loads(r[3]), "status": r[4], "timestamp": r[5],
            })

        self._close(conn)

        return {
            "active_quarantines": active,
            "resolved_quarantines": resolved,
            "recent_quarantines": recent,
            "system_health": "good" if active < 10 else "needs_attention",
        }

    def process_memory_rewrite(self, event_id: str, new_content: str,
                               preserve_truth_pointer: bool = True) -> Dict[str, Any]:
        """Rewrite memory content while optionally preserving truth pointers."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT event_id, content, edit_resistance, timestamp
            FROM index_memories WHERE event_id = ?
        ''', (event_id,))
        row = cursor.fetchone()
        if not row:
            self._close(conn)
            return {"error": f"Memory {event_id} not found"}

        _, original_content, edit_resistance, timestamp = row

        # Probabilistic resistance check
        if (edit_resistance or 0) > random.random():
            self._close(conn)
            return {
                "rewrite_blocked": True,
                "reason": "high_edit_resistance",
                "resistance_level": edit_resistance,
            }

        # Store original in akashic if preserving truth
        if preserve_truth_pointer:
            cursor.execute('''
                INSERT OR IGNORE INTO akashic_records
                (event_id, timestamp, who, what)
                VALUES (?, ?, ?, ?)
            ''', (f"truth_{event_id}_{int(timestamp)}", time.time(),
                  json.dumps(["system"]),
                  f"Original content before rewrite: {original_content}"))

        cursor.execute('''
            UPDATE index_memories SET content = ? WHERE event_id = ?
        ''', (new_content, event_id))

        conn.commit()
        self._close(conn)

        return {
            "rewrite_successful": True,
            "event_id": event_id,
            "truth_pointer_preserved": preserve_truth_pointer,
        }
