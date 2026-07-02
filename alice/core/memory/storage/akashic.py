# Copyright 2025 Rin - Alice AI System
"""
Akashic Records Storage
=======================

Factual, emotionless truth storage - the objective record of "what happened".
Immutable records that form the ground truth for Alice's memory.

Extracted from legacy index.py - Dec 2025
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..types import AkashicRecord

AKASHIC_AVAILABLE = True


class AkashicRecords:
    """
    Memory v2.0: Factual truth storage (emotionless, immutable)

    Stores objective facts about what happened, who was involved,
    and where it occurred. No emotional coloring - just the facts.
    """

    def __init__(self, db_path: str = "alice/data/databases/alice_memory.db"):
        # Handle SQLite's special :memory: database
        self.db_path = db_path if db_path == ":memory:" else Path(db_path)
        self._memory_conn = None
        self._ensure_table()

    def _connect(self):
        """Get database connection (persistent for :memory:)"""
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._memory_conn
        else:
            return sqlite3.connect(self.db_path)

    def _ensure_table(self):
        """Ensure akashic_records table exists"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS akashic_records (
                event_id TEXT PRIMARY KEY,
                timestamp REAL,
                who TEXT,
                what TEXT,
                where_location TEXT,
                facts TEXT,
                divergence_flag INTEGER DEFAULT 0,
                divergence_id TEXT,
                divergence_score REAL DEFAULT 0.0
            )
        ''')
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()

    def add_record(self, event_id: str, who: List[str], what: str,
                   where: Optional[str] = None, facts: Optional[Dict[str, Any]] = None) -> str:
        """Add factual record to Akashic storage"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO akashic_records
            (event_id, timestamp, who, what, where_location, facts)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event_id, time.time(), json.dumps(who), what, where,
            json.dumps(facts or {})
        ))

        conn.commit()
        if self.db_path != ":memory:":
            conn.close()
        return event_id

    def get_record(self, event_id: str) -> Optional[AkashicRecord]:
        """Get factual record by event ID"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM akashic_records WHERE event_id = ?', (event_id,))
        row = cursor.fetchone()
        if self.db_path != ":memory:":
            conn.close()

        if row:
            return AkashicRecord(
                event_id=row[0],
                timestamp=row[1],
                who=json.loads(row[2]),
                what=row[3],
                where=row[4],
                facts=json.loads(row[5] or '{}'),
                divergence_flag=bool(row[6]),
                divergence_id=row[7],
                divergence_score=row[8] or 0.0
            )
        return None

    def search_records(self, what_filter: str = None, who_filter: List[str] = None,
                       limit: int = 20) -> List[AkashicRecord]:
        """Search factual records"""
        conn = self._connect()
        cursor = conn.cursor()

        conditions = []
        params = []

        if what_filter:
            conditions.append("what LIKE ?")
            params.append(f"%{what_filter}%")

        if who_filter:
            for person in who_filter:
                conditions.append("who LIKE ?")
                params.append(f'%"{person}"%')

        query = "SELECT * FROM akashic_records"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        records = []
        for row in cursor.fetchall():
            records.append(AkashicRecord(
                event_id=row[0], timestamp=row[1], who=json.loads(row[2]),
                what=row[3], where=row[4], facts=json.loads(row[5] or '{}'),
                divergence_flag=bool(row[6]), divergence_id=row[7],
                divergence_score=row[8] or 0.0
            ))

        if self.db_path != ":memory:":
            conn.close()
        return records

    def mark_divergence(self, event_id: str, divergence_id: str, score: float = 1.0):
        """Mark a record as having a life-changing divergence point"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE akashic_records
            SET divergence_flag = 1, divergence_id = ?, divergence_score = ?
            WHERE event_id = ?
        ''', (divergence_id, score, event_id))

        conn.commit()
        if self.db_path != ":memory:":
            conn.close()

    def get_divergence_points(self, min_score: float = 0.5) -> List[AkashicRecord]:
        """Get all records marked as divergence points"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM akashic_records
            WHERE divergence_flag = 1 AND divergence_score >= ?
            ORDER BY divergence_score DESC
        ''', (min_score,))

        records = []
        for row in cursor.fetchall():
            records.append(AkashicRecord(
                event_id=row[0], timestamp=row[1], who=json.loads(row[2]),
                what=row[3], where=row[4], facts=json.loads(row[5] or '{}'),
                divergence_flag=bool(row[6]), divergence_id=row[7],
                divergence_score=row[8] or 0.0
            ))

        if self.db_path != ":memory:":
            conn.close()
        return records


__all__ = ['AkashicRecords', 'AkashicRecord', 'AKASHIC_AVAILABLE']
