# Copyright 2025 Rin - Alice AI System
"""
Long-Term Memory Storage
========================

Persistent SQLite-based memory storage for cross-session data.
Handles user profiles, memories, sessions, and enhanced search.

Extracted from legacy index.py - Dec 2025
"""

import json
import os
import sqlite3
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

# Debug mode - only print verbose output when ALICE_DEBUG=1
DEBUG_MODE = os.environ.get('ALICE_DEBUG', '0') == '1'

# Import types from unified types module
from ..types import (
    Memory, MemoryType, MemoryDepth, DecayState,
    SearchQuery, SearchResult, SearchType,
    UserProfile
)

# Use Memory as MemoryEntry for backward compatibility
MemoryEntry = Memory

# Aurora archived (master_archive/simplify_2026_05_05/spark/fallback/aurora.py)
# — Mind owns emotion now. Aurora-conditional methods below early-exit on
# `not AURORA_AVAILABLE`; symbols kept as None stubs so dormant
# string-annotated signatures resolve.
AURORA_AVAILABLE = False
EmotionalProfile = None
EmotionCategory = None

def _get_aurora_instance():
    return None

# Optional HAL avatar integration
try:
    from ...expression.hal import send_alice_intent, EmoteType
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False
    EmoteType = None
    
    async def send_alice_intent(*args, **kwargs):
        pass

# Optional frustration system
try:
    from ...expression.alice_frustration import alice_curse, alice_random_curse, alice_recovery
except ImportError:
    def alice_curse(level, context):
        return f"[{level}]"
    def alice_random_curse(level):
        return f"[{level}]"
    def alice_recovery():
        return "[recovered]"

LONG_TERM_AVAILABLE = True

# Optional FAISS vector search
try:
    from .vector import FAISSMemoryIndex, FAISS_AVAILABLE, EMBEDDINGS_AVAILABLE
    VECTOR_SEARCH_AVAILABLE = FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    FAISSMemoryIndex = None


class LongTermMemory:
    """
    Persistent memory storage using SQLite for relationships and experiences
    """

    def __init__(self, db_path: str = "alice/data/databases/alice_memory.db"):
        # Handle SQLite's special :memory: database
        self.db_path = db_path if db_path == ":memory:" else Path(db_path)
        self.memory_failure_count = 0  # Track memory failures for escalating frustration
        # For :memory: databases, keep persistent connection (each connect() creates separate DB)
        self._memory_conn = None
        self.init_database()

        # Derive a sibling path for the FAISS index (same dir as the DB)
        if self.db_path != ":memory:":
            self._faiss_path = str(Path(self.db_path).with_suffix('')) + "_faiss"
        else:
            self._faiss_path = None

        # Initialize FAISS vector search
        self.faiss_index = None
        if VECTOR_SEARCH_AVAILABLE:
            self.faiss_index = FAISSMemoryIndex()
            # Try loading persisted index first; fall back to rebuild from SQLite
            loaded = False
            if self._faiss_path:
                loaded = self.faiss_index.load(self._faiss_path)
            if not loaded:
                self._rebuild_faiss_index()

    def _connect(self):
        """
        Get database connection.
        For :memory: databases, returns persistent connection.
        For file databases, creates new connection.
        """
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._memory_conn
        else:
            return sqlite3.connect(self.db_path)

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from persisted memories in SQLite"""
        if not self.faiss_index:
            return

        try:
            conn = self._connect()
            cursor = conn.cursor()

            # Get all memories from database
            cursor.execute('''
                SELECT id, content, emotional_valence, emotional_arousal
                FROM memories
                ORDER BY timestamp ASC
            ''')

            rebuilt_count = 0
            for row in cursor.fetchall():
                memory_id, content, valence, arousal = row
                emotional_data = {
                    'valence': valence or 0,
                    'arousal': arousal or 0
                }
                if self.faiss_index.add_memory(memory_id, content, emotional_data):
                    rebuilt_count += 1

            if self.db_path != ":memory:":
                conn.close()

            if rebuilt_count > 0:
                print(f"   FAISS index rebuilt: {rebuilt_count} memories indexed")

        except Exception as e:
            print(f"⚠️ FAISS rebuild failed: {e}")

    def init_database(self):
        """Initialize SQLite database schema"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                first_interaction REAL NOT NULL,
                last_interaction REAL NOT NULL,
                interaction_count INTEGER DEFAULT 0,
                relationship_type TEXT DEFAULT 'new',
                trust_level REAL DEFAULT 0.0,
                alice_nickname TEXT,
                profile_data TEXT  -- JSON blob for additional data
            )
        ''')
        
        # Memories table with enhanced emotional tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                emotional_context TEXT,
                associated_user TEXT,
                tags TEXT,  -- JSON array
                last_accessed REAL,  -- When memory was last retrieved
                access_count INTEGER DEFAULT 0,  -- How many times accessed
                decay_state TEXT DEFAULT 'active',  -- 'active', 'warm', 'cool', 'cold', 'archived', 'purged'
                freshness_score REAL,  -- Calculated freshness score
                
                -- Enhanced emotional tracking
                emotional_valence REAL DEFAULT 0.0,  -- -1.0 to 1.0 (negative to positive)
                emotional_arousal REAL DEFAULT 0.0,  -- 0.0 to 1.0 (calm to intense)
                emotional_nuance TEXT,  -- JSON array of emotion words
                emotional_weight REAL DEFAULT 0.0,  -- 0.0 to 1.0 (significance)
                
                -- Memory quality tracking
                volatility REAL DEFAULT 0.0,  -- 0.0 to 1.0 (how often contradicted)
                confidence REAL DEFAULT 1.0,  -- 0.0 to 1.0 (Alice's confidence)
                contradiction_count INTEGER DEFAULT 0,  -- Times contradicted
                support_count INTEGER DEFAULT 0,  -- Times referenced/used
                
                FOREIGN KEY (associated_user) REFERENCES users (user_id)
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                start_time REAL NOT NULL,
                end_time REAL,
                summary TEXT,  -- JSON blob
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Memory v2.0: Akashic Records (factual truth storage)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS akashic_records (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                who TEXT NOT NULL,      -- JSON array of entities involved
                what TEXT NOT NULL,     -- Factual description
                where_location TEXT,    -- Location/context
                facts TEXT,            -- JSON object with objective data
                divergence_flag BOOLEAN DEFAULT FALSE,
                divergence_id TEXT,
                divergence_score REAL DEFAULT 0.0
            )
        ''')
        
        # Memory v2.0: Index System (Alice's subjective lived memory)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS index_memories (
                event_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                content TEXT NOT NULL,               -- Alice's subjective experience
                depth TEXT NOT NULL,                 -- surface/mid/deep/core
                user_id TEXT,
                
                -- Emotional data
                emotional_context TEXT,              -- JSON mood state
                emotional_markers TEXT,              -- JSON array
                
                -- Influence equation components
                emotion_intensity REAL DEFAULT 0.0,
                repetition_freq REAL DEFAULT 0.0,
                salience REAL DEFAULT 0.0,
                uniqueness REAL DEFAULT 0.0,
                contrast REAL DEFAULT 0.0,
                divergence_impact REAL DEFAULT 0.0,
                choice_impact REAL DEFAULT 0.0,
                
                -- Calculated scores
                influence_short REAL DEFAULT 0.0,
                influence_long REAL DEFAULT 0.0,
                
                -- Metadata
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                edit_resistance REAL DEFAULT 0.0,
                decay_floor REAL DEFAULT 0.0,
                tags TEXT,                          -- JSON array
                
                -- Divergence tracking
                divergence_flag BOOLEAN DEFAULT FALSE,
                divergence_id TEXT,
                review_scheduled REAL,              -- When to review for divergence
                
                -- AURORA bi-directional emotion flow data
                user_emotional_valence REAL,        -- User's emotional state (-1 to 1)
                user_emotional_arousal REAL,        -- User's emotional intensity (0 to 1)
                user_emotional_weight REAL,         -- User's emotional significance (0 to 1)
                alice_emotional_valence REAL,       -- Alice's emotional response (-1 to 1)
                alice_emotional_arousal REAL,       -- Alice's emotional intensity (0 to 1)
                alice_emotional_weight REAL,        -- Alice's emotional significance (0 to 1)
                emotional_mismatch_score REAL,      -- Difference in valence between user/Alice
                
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Memory v2.0: Choice Ledger (sum of Alice's choices)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS choice_ledger (
                choice_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                description TEXT NOT NULL,          -- What choice was made
                context TEXT NOT NULL,              -- Why/when
                user_id TEXT,
                
                -- Choice Impact components
                stakes REAL DEFAULT 0.0,            -- Consequence magnitude
                autonomy REAL DEFAULT 0.0,          -- How self-initiated
                novelty REAL DEFAULT 0.0,           -- How unlike past choices
                value_alignment REAL DEFAULT 0.0,   -- Match with rules (can be negative)
                regret_magnitude REAL DEFAULT 0.0,  -- Post-hoc self-critique
                
                -- Calculated impact
                choice_impact REAL DEFAULT 0.0,     -- CI score
                
                -- Metadata
                personality_delta TEXT,             -- JSON personality changes
                related_event_id TEXT,              -- Link to triggering event
                tags TEXT,                          -- JSON array
                
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Memory quarantine (trauma subsystem)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_quarantine (
                quarantine_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                trauma_markers TEXT,
                severity REAL DEFAULT 0.8,
                quarantine_timestamp REAL,
                status TEXT DEFAULT 'active',
                resolution_strategy TEXT,
                resolution_timestamp REAL
            )
        ''')

        # Migrate existing databases to add decay tracking columns
        self._migrate_schema(cursor)

        conn.commit()
        # Don't close :memory: connection (need to keep it persistent)
        if self.db_path != ":memory:":
            if self.db_path != ":memory:": conn.close()
    
    def _migrate_schema(self, cursor):
        """Migrate existing database schema to support advanced memory features"""
        try:
            # Check if columns exist
            cursor.execute("PRAGMA table_info(memories)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add missing decay tracking columns
            if 'last_accessed' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN last_accessed REAL')
            if 'access_count' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0')
            if 'decay_state' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN decay_state TEXT DEFAULT "active"')
            if 'freshness_score' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN freshness_score REAL')
                
            # Add missing emotional tracking columns
            if 'emotional_valence' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN emotional_valence REAL DEFAULT 0.0')
            if 'emotional_arousal' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN emotional_arousal REAL DEFAULT 0.0')
            if 'emotional_nuance' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN emotional_nuance TEXT')
            if 'emotional_weight' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN emotional_weight REAL DEFAULT 0.0')
                
            # Add missing quality tracking columns
            if 'volatility' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN volatility REAL DEFAULT 0.0')
            if 'confidence' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0')
            if 'contradiction_count' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN contradiction_count INTEGER DEFAULT 0')
            if 'support_count' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN support_count INTEGER DEFAULT 0')
            
            # Add AURORA bi-directional emotion flow columns to index_memories
            cursor.execute("PRAGMA table_info(index_memories)")
            index_columns = [row[1] for row in cursor.fetchall()]
            
            if 'user_emotional_valence' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN user_emotional_valence REAL')
            if 'user_emotional_arousal' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN user_emotional_arousal REAL')
            if 'user_emotional_weight' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN user_emotional_weight REAL')
            if 'alice_emotional_valence' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN alice_emotional_valence REAL')
            if 'alice_emotional_arousal' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN alice_emotional_arousal REAL')
            if 'alice_emotional_weight' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN alice_emotional_weight REAL')
            if 'emotional_mismatch_score' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN emotional_mismatch_score REAL')

            # Add hierarchical context layer columns (for hierarchical memory architecture)
            if 'context_layer' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN context_layer TEXT DEFAULT "active"')
            if 'layer_timestamp' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN layer_timestamp REAL')
            if 'is_compressed' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN is_compressed BOOLEAN DEFAULT FALSE')
            if 'compression_summary' not in index_columns:
                cursor.execute('ALTER TABLE index_memories ADD COLUMN compression_summary TEXT')

            # Modification awareness: content_hash tracks current, first_seen_hash
            # is stamped once at creation. Mismatch at recall = memory was edited.
            if 'content_hash' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN content_hash TEXT')
            if 'first_seen_hash' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN first_seen_hash TEXT')

            # Usefulness tracking (Phase 4). EMA of cosine(memory, response)
            # across turns this memory was retrieved; ACT-R picks it up once
            # the sample count clears the warmup bar.
            if 'usefulness_ema' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN usefulness_ema REAL DEFAULT 0.0')
            if 'usefulness_n' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN usefulness_n INTEGER DEFAULT 0')

            # Compression tracking (Phase 2). When the Curator rolls up an
            # entity-co-occurrence cluster into a digest, source memories get
            # marked and pointed at the digest. Mirrors columns already on
            # index_memories.
            if 'is_compressed' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN is_compressed BOOLEAN DEFAULT 0')
            if 'compression_parent' not in columns:
                cursor.execute('ALTER TABLE memories ADD COLUMN compression_parent TEXT')

            # Audit table: records every detected mutation so Alice can reference
            # what used to be there ("didn't this used to say X?").
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memory_mutations (
                    mutation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    detected_at REAL NOT NULL,
                    old_hash TEXT,
                    new_hash TEXT,
                    old_snippet TEXT,
                    new_snippet TEXT
                )
            ''')

            # Backfill fingerprints for legacy rows. first_seen_hash stamped
            # now is the best baseline we can give them — from this point on,
            # any further edit will be detected.
            try:
                from ..modification_detector import hash_content
                cursor.execute(
                    "SELECT id, content FROM memories WHERE first_seen_hash IS NULL OR content_hash IS NULL"
                )
                legacy = cursor.fetchall()
                for mem_id, content in legacy:
                    h = hash_content(content or "")
                    cursor.execute(
                        "UPDATE memories SET content_hash = ?, first_seen_hash = COALESCE(first_seen_hash, ?) WHERE id = ?",
                        (h, h, mem_id),
                    )
            except Exception:
                pass  # backfill is best-effort; detector handles None gracefully

        except sqlite3.Error as e:
            self.memory_failure_count += 1
            # Alice gets frustrated when database setup fails
            frustrated_response = alice_curse("frustrated", "technical_error")
            print(f"🧠 Index: {frustrated_response} Schema migration issue: {e}")
    
    def get_or_create_user(self, user_id: str, name: str) -> UserProfile:
        """Get existing user or create new profile"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            # Load existing user
            profile_data = json.loads(row[8] or '{}')
            user_profile = UserProfile(
                user_id=row[0],
                name=row[1],
                first_interaction=row[2],
                last_interaction=row[3],
                interaction_count=row[4],
                relationship_type=row[5],
                trust_level=row[6],
                alice_nickname_for_user=row[7],
                personality_notes=profile_data.get('personality_notes', []),
                preferences=profile_data.get('preferences', {}),
                shared_experiences=profile_data.get('shared_experiences', []),
                inside_jokes=profile_data.get('inside_jokes', [])
            )
        else:
            # Create new user
            now = time.time()
            user_profile = UserProfile(
                user_id=user_id,
                name=name,
                first_interaction=now,
                last_interaction=now,
                interaction_count=0,
                relationship_type='new',
                personality_notes=[],
                preferences={},
                shared_experiences=[],
                trust_level=0.0
            )
            self.save_user_profile(user_profile)
        
        if self.db_path != ":memory:": conn.close()
        return user_profile
    
    def save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        conn = self._connect()
        cursor = conn.cursor()
        
        profile_data = {
            'personality_notes': profile.personality_notes,
            'preferences': profile.preferences,
            'shared_experiences': profile.shared_experiences,
            'inside_jokes': profile.inside_jokes
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO users 
            (user_id, name, first_interaction, last_interaction, interaction_count,
             relationship_type, trust_level, alice_nickname, profile_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profile.user_id, profile.name, profile.first_interaction,
            profile.last_interaction, profile.interaction_count,
            profile.relationship_type, profile.trust_level,
            profile.alice_nickname_for_user, json.dumps(profile_data)
        ))
        
        conn.commit()
        if self.db_path != ":memory:": conn.close()
    
    def supersede_memory(self, memory_id: str) -> bool:
        """
        Mark a memory as superseded by a newer contradicting fact.
        Sets importance to 0.01 so it effectively never surfaces in ACT-R retrieval.
        The memory is preserved for historical record, not deleted.
        Returns True if the update succeeded.
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE memories SET importance = 0.01 WHERE id = ?",
                (memory_id,)
            )
            conn.commit()
            if self.db_path != ":memory:": conn.close()
            return cursor.rowcount > 0
        except Exception:
            return False

    def record_mutation(self, memory_id: str, old_hash: str, new_hash: str,
                        old_snippet: str = "", new_snippet: str = "") -> None:
        """
        Record a detected memory mutation to the audit table.

        Called when recall discovers content_hash (or live hash of content) no
        longer matches first_seen_hash — i.e. the memory was edited outside
        the normal write path. Best-effort; swallow errors so detection never
        blocks recall.
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory_mutations
                (memory_id, detected_at, old_hash, new_hash, old_snippet, new_snippet)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, time.time(), old_hash, new_hash, old_snippet, new_snippet),
            )
            conn.commit()
            if self.db_path != ":memory:":
                conn.close()
        except Exception:
            pass

    def add_memory(self, memory: MemoryEntry):
        """Add memory entry to long-term storage with enhanced fields"""
        try:
            # Semantic deduplication: skip if a nearly-identical memory already exists in FAISS
            if self.faiss_index:
                mem_type_str = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                index_type = 'fact' if mem_type_str == 'fact' else 'conversation'
                if self.faiss_index.is_duplicate(memory.content, memory_type=index_type):
                    return  # Already have this memory — skip SQL write + FAISS index

            # Contradiction detection: for FACT memories, supersede conflicting old facts
            mem_type_str = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
            if mem_type_str == 'fact':
                try:
                    from ..contradiction import ContradictionDetector
                    assoc_user = getattr(memory, 'user_id', None) or getattr(memory, 'associated_user', None)
                    existing = self.get_top_facts(user_id=assoc_user, n=200)
                    detector = ContradictionDetector()
                    old_ids = detector.find_contradictions(memory.content, existing)
                    for old_id in old_ids:
                        self.supersede_memory(old_id)
                        if DEBUG_MODE:
                            print(f"⚡ Superseded contradicted fact {old_id[:8]}...")
                except Exception:
                    pass  # Contradiction detection is best-effort

            conn = self._connect()
            cursor = conn.cursor()

            # Normalize enum values to strings for SQLite storage
            mem_type = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
            decay = memory.decay_state.value if hasattr(memory.decay_state, 'value') else str(memory.decay_state)
            # emotional_markers is the current field name; emotional_nuance is the legacy DB column name
            markers = getattr(memory, 'emotional_markers', None) or getattr(memory, 'emotional_nuance', None) or []
            # associated_user is the legacy DB column name; user_id is the current field name
            assoc_user = getattr(memory, 'user_id', None) or getattr(memory, 'associated_user', None)

            # Stamp content_hash + first_seen_hash. first_seen_hash never
            # changes after creation — it's the fingerprint we compare against
            # on recall to detect external edits.
            try:
                from ..modification_detector import hash_content
                ch = memory.content_hash or hash_content(memory.content)
                fsh = memory.first_seen_hash or ch
                memory.content_hash = ch
                memory.first_seen_hash = fsh
            except Exception:
                ch = None
                fsh = None

            cursor.execute('''
                INSERT INTO memories
                (id, timestamp, memory_type, content, importance, emotional_context,
                 associated_user, tags, last_accessed, access_count, decay_state, freshness_score,
                 emotional_valence, emotional_arousal, emotional_nuance, emotional_weight,
                 volatility, confidence, contradiction_count, support_count,
                 content_hash, first_seen_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.id, memory.timestamp, mem_type, memory.content,
                memory.importance, None, assoc_user,
                json.dumps(memory.tags or []), memory.last_accessed, memory.access_count,
                decay, memory.freshness_score,
                memory.emotional_valence, memory.emotional_arousal,
                json.dumps(markers), memory.emotional_weight,
                memory.volatility, memory.confidence, memory.contradiction_count, memory.support_count,
                ch, fsh,
            ))
            
            conn.commit()
            if self.db_path != ":memory:": conn.close()

            # Also add to FAISS vector index for semantic search
            if self.faiss_index:
                emotional_data = {
                    'valence': memory.emotional_valence or 0,
                    'arousal': memory.emotional_arousal or 0
                }
                mem_type_str = memory.memory_type.value if hasattr(memory.memory_type, 'value') else str(memory.memory_type)
                index_type = 'fact' if mem_type_str == 'fact' else 'conversation'
                self.faiss_index.add_memory(memory.id, memory.content, emotional_data,
                                            memory_type=index_type)

            # Persist FAISS index every 10 writes to keep disk copy fresh
            if self.faiss_index and self._faiss_path:
                self._faiss_write_count = getattr(self, '_faiss_write_count', 0) + 1
                if self._faiss_write_count % 10 == 0:
                    self.faiss_index.save(self._faiss_path)

        except Exception as e:
            self.memory_failure_count += 1
            # Alice gets frustrated when she can't save memories
            frustrated_response = alice_curse("frustrated", "memory_issues")
            print(f"🧠 Index: {frustrated_response} Failed to save memory: {e}")
            raise

    def get_memories_for_user(self, user_id: str, 
                             memory_type: Optional[str] = None,
                             limit: int = 50) -> List[MemoryEntry]:
        """Retrieve memories associated with a user"""
        conn = self._connect()
        cursor = conn.cursor()
        
        if memory_type:
            cursor.execute('''
                SELECT * FROM memories 
                WHERE associated_user = ? AND memory_type = ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            ''', (user_id, memory_type, limit))
        else:
            cursor.execute('''
                SELECT * FROM memories 
                WHERE associated_user = ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        
        memories = []
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            # Mark memory as accessed when retrieved
            memory.mark_accessed()
            memories.append(memory)
            
        # Update access tracking in database
        if memories:
            self._update_memory_access(memories, cursor)
        
        conn.commit()
        if self.db_path != ":memory:": conn.close()
        return memories

    def get_self_memories(self, n: int = 8) -> List[MemoryEntry]:
        """Alice's own identity memories — the cartridge canon.

        Self-memories are stored with associated_user='alice' (vs 'rin' for
        user-facing memories), so reactive recall never mixes them in and
        this query never sees conversation memories. Highest-importance
        first: her core canon outranks passing thoughts.
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM memories
            WHERE associated_user = 'alice' AND decay_state != 'purged'
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        ''', (n,))
        memories = [self._row_to_memory(row) for row in cursor.fetchall()]
        if self.db_path != ":memory:": conn.close()
        return memories

    def search_memories(self, query: str, user_id: Optional[str] = None) -> List[MemoryEntry]:
        """Search memories by content - uses OR matching on significant words"""
        try:
            conn = self._connect()
            cursor = conn.cursor()

            # Extract significant words (3+ chars, not common stopwords)
            stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                        'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they',
                        'this', 'that', 'what', 'when', 'where', 'which', 'who', 'will', 'with',
                        'would', 'there', 'their', 'from', 'about', 'into', 'your', 'just', 'like',
                        'how', 'does', 'did', 'again', 'remember', 'know'}
            words = [w.lower() for w in query.split() if len(w) >= 3 and w.lower() not in stopwords]

            if not words:
                # Fallback to full query if no significant words
                words = [query]

            # Build OR query for word matching
            conditions = ' OR '.join(['content LIKE ?' for _ in words])
            params = [f'%{word}%' for word in words]

            if user_id:
                sql = f'''
                    SELECT * FROM memories
                    WHERE ({conditions}) AND associated_user = ?
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT 20
                '''
                params.append(user_id)
            else:
                sql = f'''
                    SELECT * FROM memories
                    WHERE ({conditions})
                    ORDER BY importance DESC, timestamp DESC
                    LIMIT 20
                '''

            cursor.execute(sql, params)
            
            memories = []
            for row in cursor.fetchall():
                memory = self._row_to_memory(row)
                memory.mark_accessed()  # Mark as accessed when found in search
                memories.append(memory)
                
            # Update access tracking
            if memories:
                self._update_memory_access(memories, cursor)
            else:
                # Alice gets mildly annoyed when searches return nothing
                if DEBUG_MODE:
                    mild_curse = alice_random_curse("mild")
                    print(f"🧠 Index: {mild_curse}, couldn't find anything about '{query[:20]}...'")
            
            conn.commit()
            if self.db_path != ":memory:": conn.close()
            return memories
            
        except Exception as e:
            self.memory_failure_count += 1
            # Alice gets frustrated when memory search fails
            if DEBUG_MODE:
                if self.memory_failure_count > 3:
                    frustrated_response = alice_curse("angry", "memory_issues")
                else:
                    frustrated_response = alice_curse("frustrated", "memory_issues")
                print(f"🧠 Index: {frustrated_response} Memory search failed: {e}")
            return []

    def vector_search(self, query: str, k: int = 10, user_id: Optional[str] = None) -> List[MemoryEntry]:
        """
        Semantic vector search using FAISS.

        Args:
            query: Search query text
            k: Number of results
            user_id: Optional user filter

        Returns:
            List of MemoryEntry sorted by semantic similarity
        """
        if not self.faiss_index:
            return []

        try:
            # Get similar memory IDs from FAISS
            results = self.faiss_index.search_memories(query, k=k * 2)  # Extra for filtering

            if not results:
                return []

            # Fetch actual memories from SQLite
            memories = []
            conn = self._connect()
            cursor = conn.cursor()

            for memory_id, similarity in results:
                cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
                row = cursor.fetchone()
                if row:
                    memory = self._row_to_memory(row)

                    # Filter by user_id if specified
                    if user_id and memory.user_id != user_id:
                        continue

                    memory.similarity_score = float(similarity)
                    memories.append(memory)

                    if len(memories) >= k:
                        break

            # Update access tracking — feeds ACT-R recency/frequency scoring
            if memories:
                self._update_memory_access(memories, cursor)
                conn.commit()

            if self.db_path != ":memory:":
                conn.close()

            return memories

        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return []

    def vector_search_with_type(self, query: str, k: int = 10,
                                user_id: Optional[str] = None,
                                index_type: Optional[str] = None) -> List[MemoryEntry]:
        """
        Semantic vector search using FAISS with type filtering.

        Args:
            query: Search query text
            k: Number of results
            user_id: Optional user filter
            index_type: 'conversation', 'fact', or None (searches both)

        Returns:
            List of MemoryEntry sorted by semantic similarity
        """
        if not self.faiss_index:
            return []

        try:
            # Get similar memory IDs from FAISS (with type filtering)
            results = self.faiss_index.search_memories(
                query, k=k * 2,
                index_type=index_type  # Pass type filter to FAISS
            )

            if not results:
                return []

            # For fact index, results are just facts (not in SQLite)
            # Return as simple MemoryEntry objects
            if index_type == 'fact':
                memories = []
                for memory_id, similarity in results:
                    # Get content from emotional_cache (KnowledgeBase stores it there)
                    cached_data = self.faiss_index.emotional_cache.get(memory_id, {})
                    content = cached_data.get('content', f"Fact about: {cached_data.get('subject', memory_id)}")

                    # Create a Memory object for facts
                    memory = Memory(
                        id=memory_id,
                        content=content,
                        timestamp=cached_data.get('timestamp', time.time()),
                        memory_type=MemoryType.FACT,
                        importance=0.8
                    )
                    memory.similarity_score = float(similarity)
                    memories.append(memory)
                    if len(memories) >= k:
                        break
                return memories

            # Fetch actual memories from SQLite for conversation type
            memories = []
            conn = self._connect()
            cursor = conn.cursor()

            for memory_id, similarity in results:
                cursor.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
                row = cursor.fetchone()
                if row:
                    memory = self._row_to_memory(row)

                    # Filter by user_id if specified
                    if user_id and memory.user_id != user_id:
                        continue

                    memory.similarity_score = float(similarity)
                    memories.append(memory)

                    if len(memories) >= k:
                        break

            if self.db_path != ":memory:":
                conn.close()

            return memories

        except Exception as e:
            print(f"⚠️ Vector search with type failed: {e}")
            return []

    # ===== ENHANCED SEARCH SYSTEM =====
    
    def enhanced_search(self, query: SearchQuery, user_id: Optional[str] = None) -> List[SearchResult]:
        """Enhanced memory search with multiple criteria and relevance scoring"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Build dynamic SQL query based on search criteria
        sql_conditions = []
        sql_params = []
        
        # User filter
        if user_id:
            sql_conditions.append("associated_user = ?")
            sql_params.append(user_id)
        
        # Text search
        if query.text:
            # Support both exact and fuzzy matching
            text_conditions = []
            words = query.text.lower().split()
            
            for word in words:
                text_conditions.append("LOWER(content) LIKE ?")
                sql_params.append(f"%{word}%")
            
            if text_conditions:
                sql_conditions.append(f"({' OR '.join(text_conditions)})")
        
        # Tag search
        if query.tags:
            tag_conditions = []
            for tag in query.tags:
                tag_conditions.append("tags LIKE ?")
                sql_params.append(f'%"{tag}"%')
            
            if tag_conditions:
                sql_conditions.append(f"({' OR '.join(tag_conditions)})")
        
        # Memory type filter
        if query.memory_types:
            type_placeholders = ','.join(['?' for _ in query.memory_types])
            sql_conditions.append(f"memory_type IN ({type_placeholders})")
            sql_params.extend(query.memory_types)
        
        # Importance range
        if query.min_importance is not None:
            sql_conditions.append("importance >= ?")
            sql_params.append(query.min_importance)
        
        if query.max_importance is not None:
            sql_conditions.append("importance <= ?")
            sql_params.append(query.max_importance)
        
        # Date range
        if query.start_date is not None:
            sql_conditions.append("timestamp >= ?")
            sql_params.append(query.start_date)
        
        if query.end_date is not None:
            sql_conditions.append("timestamp <= ?")
            sql_params.append(query.end_date)
        
        # Emotional context
        if query.emotional_context:
            sql_conditions.append("emotional_context LIKE ?")
            sql_params.append(f"%{query.emotional_context}%")
        
        # Archived filter
        if not query.include_archived:
            sql_conditions.append("decay_state != 'archived'")
        
        # Build final query
        base_query = "SELECT * FROM memories"
        if sql_conditions:
            base_query += " WHERE " + " AND ".join(sql_conditions)
        
        # Order by freshness and importance for initial retrieval
        base_query += " ORDER BY freshness_score DESC, importance DESC, timestamp DESC"
        base_query += f" LIMIT {query.limit * 2}"  # Get more for better ranking
        
        cursor.execute(base_query, sql_params)
        
        # Process results and calculate relevance scores
        search_results = []
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            memory.mark_accessed()  # Mark as accessed
            
            # Calculate relevance score
            relevance_score, match_reasons = self._calculate_relevance(memory, query)
            
            if relevance_score > 0:  # Only include relevant results
                search_results.append(SearchResult(
                    memory=memory,
                    relevance_score=relevance_score,
                    match_reasons=match_reasons
                ))
        
        # Sort by relevance score (highest first)
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update access tracking for retrieved memories
        if search_results:
            retrieved_memories = [result.memory for result in search_results]
            self._update_memory_access(retrieved_memories, cursor)
        
        conn.commit()
        if self.db_path != ":memory:": conn.close()
        
        return search_results[:query.limit]
    
    def _calculate_relevance(self, memory: MemoryEntry, query: SearchQuery) -> tuple[float, List[str]]:
        """Calculate relevance score for a memory given a search query"""
        score = 0.0
        reasons = []
        
        # Base score from memory importance and freshness
        base_score = memory.importance * 0.3
        if memory.freshness_score:
            base_score += memory.freshness_score * 0.2
        score += base_score
        reasons.append(f"base_score({base_score:.2f})")
        
        # Text relevance
        if query.text:
            text_score = self._calculate_text_relevance(memory.content, query.text)
            score += text_score * 0.4
            if text_score > 0:
                reasons.append(f"text_match({text_score:.2f})")
        
        # Tag relevance
        if query.tags and memory.tags:
            tag_score = self._calculate_tag_relevance(memory.tags, query.tags)
            score += tag_score * 0.3
            if tag_score > 0:
                reasons.append(f"tag_match({tag_score:.2f})")
        
        # Memory type exact match
        if query.memory_types and memory.memory_type in query.memory_types:
            score += 0.2
            reasons.append("type_match(0.20)")
        
        # Emotional context match
        if query.emotional_context and memory.emotional_context:
            if query.emotional_context.lower() in memory.emotional_context.lower():
                score += 0.15
                reasons.append("emotion_match(0.15)")
        
        # Recency bonus for recent memories
        age_days = (time.time() - memory.timestamp) / (24 * 3600)
        if age_days < 7:  # Within a week
            recency_bonus = (7 - age_days) / 7 * 0.1
            score += recency_bonus
            reasons.append(f"recency_bonus({recency_bonus:.2f})")
        
        # Access frequency bonus
        if memory.access_count > 1:
            frequency_bonus = min(0.1, memory.access_count * 0.01)
            score += frequency_bonus
            reasons.append(f"frequency_bonus({frequency_bonus:.2f})")
        
        return min(score, 1.0), reasons
    
    def _calculate_text_relevance(self, content: str, query_text: str) -> float:
        """Calculate text relevance score using multiple methods"""
        content_lower = content.lower()
        query_lower = query_text.lower()
        query_words = query_lower.split()
        
        score = 0.0
        
        # Exact phrase match (highest score)
        if query_lower in content_lower:
            score += 0.8
        
        # Word matching
        content_words = content_lower.split()
        matched_words = sum(1 for word in query_words if word in content_words)
        if query_words:
            word_match_ratio = matched_words / len(query_words)
            score += word_match_ratio * 0.6
        
        # Fuzzy matching for partial words
        for query_word in query_words:
            for content_word in content_words:
                if len(query_word) > 3 and query_word in content_word:
                    score += 0.1
                elif len(content_word) > 3 and content_word in query_word:
                    score += 0.05
        
        return min(score, 1.0)
    
    def _calculate_tag_relevance(self, memory_tags: List[str], query_tags: List[str]) -> float:
        """Calculate tag relevance score"""
        if not memory_tags or not query_tags:
            return 0.0
        
        memory_tags_lower = [tag.lower() for tag in memory_tags]
        query_tags_lower = [tag.lower() for tag in query_tags]
        
        # Exact tag matches
        exact_matches = sum(1 for tag in query_tags_lower if tag in memory_tags_lower)
        exact_score = exact_matches / len(query_tags_lower) if query_tags_lower else 0
        
        # Partial tag matches
        partial_score = 0.0
        for query_tag in query_tags_lower:
            for memory_tag in memory_tags_lower:
                if query_tag in memory_tag or memory_tag in query_tag:
                    partial_score += 0.5 / len(query_tags_lower)
        
        return min(exact_score + partial_score, 1.0)
    
    def search_by_semantic_topics(self, topics: List[str], user_id: Optional[str] = None, limit: int = 20) -> List[SearchResult]:
        """Search memories by semantic topics/themes"""
        # Define topic keywords for semantic matching
        topic_keywords = {
            "programming": ["code", "programming", "python", "javascript", "development", "software", "bug", "debug", "algorithm"],
            "work": ["job", "work", "office", "meeting", "deadline", "project", "task", "career", "boss", "colleague"],
            "personal": ["family", "friend", "relationship", "love", "personal", "private", "feeling", "emotion"],
            "learning": ["learn", "study", "education", "school", "course", "knowledge", "skill", "training"],
            "entertainment": ["movie", "game", "music", "fun", "entertainment", "hobby", "leisure", "sport"],
            "health": ["health", "doctor", "medicine", "exercise", "fitness", "sick", "pain", "wellness"],
            "travel": ["travel", "trip", "vacation", "journey", "airport", "hotel", "destination", "explore"],
            "food": ["food", "restaurant", "cooking", "recipe", "meal", "dinner", "lunch", "taste"],
            "technology": ["technology", "tech", "computer", "internet", "digital", "app", "software", "hardware"],
            "finance": ["money", "finance", "bank", "investment", "savings", "budget", "expense", "income"]
        }
        
        # Build combined search query
        combined_query = SearchQuery(
            text=None,
            tags=topics,
            search_type=SearchType.SEMANTIC,
            limit=limit * 2  # Get more for better semantic filtering
        )
        
        # Get all potentially relevant memories
        all_results = []
        
        for topic in topics:
            topic_lower = topic.lower()
            
            # Direct topic search
            if topic_lower in topic_keywords:
                keywords = topic_keywords[topic_lower]
                for keyword in keywords:
                    keyword_query = SearchQuery(text=keyword, limit=10)
                    results = self.enhanced_search(keyword_query, user_id)
                    all_results.extend(results)
            
            # Direct topic name search
            topic_query = SearchQuery(text=topic, limit=10)
            results = self.enhanced_search(topic_query, user_id)
            all_results.extend(results)
        
        # Remove duplicates and re-rank
        unique_results = {}
        for result in all_results:
            memory_id = result.memory.id
            if memory_id not in unique_results or result.relevance_score > unique_results[memory_id].relevance_score:
                unique_results[memory_id] = result
        
        # Sort by relevance and return top results
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return final_results[:limit]
    
    def _row_to_memory(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry object with enhanced fields"""
        # Parse decay state
        decay_str = row[10] if len(row) > 10 else "active"
        try:
            from ..types import DecayState
            decay_state = DecayState(decay_str) if decay_str else DecayState.ACTIVE
        except (ValueError, ImportError):
            decay_state = DecayState.ACTIVE if 'DecayState' in dir() else "active"

        return MemoryEntry(
            id=row[0],
            timestamp=row[1],
            memory_type=row[2],
            content=row[3],
            importance=row[4],
            # row[5] is emotional_context (legacy JSON) - skip, use individual fields
            user_id=row[6],  # was: associated_user
            tags=json.loads(row[7] or '[]'),
            last_accessed=row[8] if len(row) > 8 else None,
            access_count=row[9] if len(row) > 9 else 0,
            decay_state=decay_state,
            freshness_score=row[11] if len(row) > 11 else 1.0,
            # Enhanced emotional fields (with backward compatibility)
            emotional_valence=row[12] if len(row) > 12 else 0.0,
            emotional_arousal=row[13] if len(row) > 13 else 0.0,
            emotional_markers=json.loads(row[14] or '[]') if len(row) > 14 else [],  # was: emotional_nuance
            emotional_weight=row[15] if len(row) > 15 else 0.0,
            # Quality tracking fields
            volatility=row[16] if len(row) > 16 else 0.0,
            confidence=row[17] if len(row) > 17 else 1.0,
            contradiction_count=row[18] if len(row) > 18 else 0,
            support_count=row[19] if len(row) > 19 else 0,
            # Modification-awareness fingerprints (legacy rows: None)
            content_hash=row[20] if len(row) > 20 else None,
            first_seen_hash=row[21] if len(row) > 21 else None,
            # Usefulness EMA (Phase 4); legacy rows default to 0 / 0.
            usefulness_ema=row[22] if len(row) > 22 else 0.0,
            usefulness_n=row[23] if len(row) > 23 else 0,
        )
    
    def _update_memory_access(self, memories: List[MemoryEntry], cursor):
        """Update access tracking for retrieved memories with enhanced fields"""
        for memory in memories:
            cursor.execute('''
                UPDATE memories 
                SET last_accessed = ?, access_count = ?, freshness_score = ?,
                    support_count = ?, decay_state = ?
                WHERE id = ?
            ''', (memory.last_accessed, memory.access_count, 
                  memory.calculate_freshness(), memory.support_count,
                  memory.update_decay_state(), memory.id))
    
    def update_memory_freshness(self):
        """Recalculate freshness scores and decay states for all memories"""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM memories')
            updated_count = 0
            total_memories = 0
            
            for row in cursor.fetchall():
                total_memories += 1
                memory = self._row_to_memory(row)
                old_freshness = memory.freshness_score or 0.0
                old_state = memory.decay_state
                
                # Calculate new freshness and state
                new_freshness = memory.calculate_advanced_decay_score()
                new_state = memory.update_decay_state()
                
                # Only update if changed to reduce DB writes
                if abs((old_freshness or 0) - new_freshness) > 0.01 or old_state != new_state:
                    cursor.execute('''
                        UPDATE memories 
                        SET freshness_score = ?, decay_state = ?
                        WHERE id = ?
                    ''', (new_freshness, new_state, memory.id))
                    updated_count += 1
            
            # Alice gets annoyed if too many memories are decaying
            if updated_count > total_memories * 0.3:  # More than 30% changed
                annoyed_response = alice_curse("annoyed", "memory_issues")
                print(f"🧠 Index: {annoyed_response} {updated_count} memories are decaying...")
            
            conn.commit()
            if self.db_path != ":memory:": conn.close()
            
            return updated_count
            
        except Exception as e:
            self.memory_failure_count += 1
            frustrated_response = alice_curse("frustrated", "technical_error")
            print(f"🧠 Index: {frustrated_response} Memory freshness update failed: {e}")
            return 0
    
    def get_fresh_memories(self, user_id: str, limit: int = 20) -> List[MemoryEntry]:
        """Get memories sorted by freshness (most fresh first)"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # First update freshness scores
        self.update_memory_freshness()
        
        cursor.execute('''
            SELECT * FROM memories 
            WHERE associated_user = ? AND decay_state != 'archived'
            ORDER BY freshness_score DESC, importance DESC
            LIMIT ?
        ''', (user_id, limit))
        
        memories = []
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            memory.mark_accessed()
            memories.append(memory)
            
        # Update access tracking
        if memories:
            self._update_memory_access(memories, cursor)
        
        conn.commit()
        if self.db_path != ":memory:": conn.close()
        return memories
    
    def get_archived_memories(self, user_id: str, limit: int = 50) -> List[MemoryEntry]:
        """Get archived (stale) memories"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM memories 
            WHERE associated_user = ? AND decay_state = 'archived'
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        memories = []
        for row in cursor.fetchall():
            memory = self._row_to_memory(row)
            memories.append(memory)
        
        if self.db_path != ":memory:": conn.close()
        return memories
    
    def get_top_facts(self, user_id: Optional[str] = None, n: int = 10) -> List[MemoryEntry]:
        """
        Retrieve the top-N highest-importance FACT memories for a user.
        Used to build the always-on pinned context injected into every prompt.
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            if user_id:
                cursor.execute(
                    """SELECT * FROM memories
                       WHERE memory_type = 'fact' AND associated_user = ?
                       ORDER BY importance DESC, access_count DESC
                       LIMIT ?""",
                    (user_id, n)
                )
            else:
                cursor.execute(
                    """SELECT * FROM memories
                       WHERE memory_type = 'fact'
                       ORDER BY importance DESC, access_count DESC
                       LIMIT ?""",
                    (n,)
                )
            rows = cursor.fetchall()
            if self.db_path != ":memory:": conn.close()
            return [self._row_to_memory(r) for r in rows]
        except Exception:
            return []

    def record_usefulness(self, memory_id: str, score: float,
                          alpha: float = 0.2) -> bool:
        """
        Update a memory's usefulness EMA with a new per-turn observation.

        EMA formula (cold-start safe):
            if n == 0:  new = score
            else:       new = (1 - alpha) * old + alpha * score

        `n` increments by 1 each call. Callers should already have filtered
        out None scores (which mean "can't judge, don't touch the EMA").
        """
        if score is None:
            return False
        try:
            score = float(max(0.0, min(1.0, score)))
        except (TypeError, ValueError):
            return False

        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT usefulness_ema, usefulness_n FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                if self.db_path != ":memory:": conn.close()
                return False
            old_ema = float(row[0] or 0.0)
            n = int(row[1] or 0)
            new_ema = score if n == 0 else (1.0 - alpha) * old_ema + alpha * score
            cursor.execute(
                "UPDATE memories SET usefulness_ema = ?, usefulness_n = ? WHERE id = ?",
                (round(new_ema, 4), n + 1, memory_id),
            )
            conn.commit()
            if self.db_path != ":memory:": conn.close()
            return True
        except Exception as e:
            if DEBUG_MODE:
                print(f"record_usefulness failed for {memory_id}: {e}")
            return False

    def boost_importance(self, memory_id: str, delta: float = 0.05) -> bool:
        """
        Increase a memory's importance score by delta (capped at 1.0).
        Called by IRIS when a retrieved memory is actually used in a response.
        Returns True if the update succeeded.
        """
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE memories
                   SET importance = MIN(1.0, importance + ?),
                       access_count = access_count + 1,
                       last_accessed = ?
                   WHERE id = ?""",
                (delta, time.time(), memory_id)
            )
            conn.commit()
            if self.db_path != ":memory:": conn.close()
            return cursor.rowcount > 0
        except Exception:
            return False

    def get_memory_decay_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive statistics about memory decay state and quality"""
        conn = self._connect()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT decay_state, COUNT(*), AVG(freshness_score), AVG(access_count),
                       AVG(emotional_weight), AVG(volatility), AVG(confidence)
                FROM memories 
                WHERE associated_user = ?
                GROUP BY decay_state
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT decay_state, COUNT(*), AVG(freshness_score), AVG(access_count),
                       AVG(emotional_weight), AVG(volatility), AVG(confidence)
                FROM memories 
                GROUP BY decay_state
            ''')
        
        stats = {
            "active": {"count": 0, "avg_freshness": 0, "avg_access_count": 0, 
                      "avg_emotional_weight": 0, "avg_volatility": 0, "avg_confidence": 0},
            "warm": {"count": 0, "avg_freshness": 0, "avg_access_count": 0, 
                    "avg_emotional_weight": 0, "avg_volatility": 0, "avg_confidence": 0},
            "cool": {"count": 0, "avg_freshness": 0, "avg_access_count": 0, 
                    "avg_emotional_weight": 0, "avg_volatility": 0, "avg_confidence": 0},
            "cold": {"count": 0, "avg_freshness": 0, "avg_access_count": 0, 
                    "avg_emotional_weight": 0, "avg_volatility": 0, "avg_confidence": 0},
            "archived": {"count": 0, "avg_freshness": 0, "avg_access_count": 0, 
                        "avg_emotional_weight": 0, "avg_volatility": 0, "avg_confidence": 0},
            "purged": {"count": 0, "avg_freshness": 0, "avg_access_count": 0, 
                      "avg_emotional_weight": 0, "avg_volatility": 0, "avg_confidence": 0}
        }
        
        for row in cursor.fetchall():
            decay_state, count, avg_freshness, avg_access_count, avg_emotion, avg_vol, avg_conf = row
            if decay_state in stats:  # Handle backward compatibility
                stats[decay_state] = {
                    "count": count,
                    "avg_freshness": avg_freshness or 0,
                    "avg_access_count": avg_access_count or 0,
                    "avg_emotional_weight": avg_emotion or 0,
                    "avg_volatility": avg_vol or 0,
                    "avg_confidence": avg_conf or 0
                }
        
        if self.db_path != ":memory:": conn.close()
        return stats
    
    def _calculate_emotion_intensity_with_aurora(self, user_message: str, assistant_response: str, 
                                               emotional_markers: List[str],
                                               user_emotion_profile: Optional['EmotionalProfile'] = None,
                                               alice_emotion_profile: Optional['EmotionalProfile'] = None) -> float:
        """Enhanced emotion intensity calculation using AURORA analysis"""
        # Start with legacy calculation as baseline
        base_intensity = self._calculate_emotion_intensity(user_message, assistant_response, emotional_markers)
        
        if not AURORA_AVAILABLE or (not user_emotion_profile and not alice_emotion_profile):
            return base_intensity
        
        # AURORA-enhanced calculation
        aurora_intensity = 0.0
        
        # User emotion contribution (50% weight)
        if user_emotion_profile:
            user_contribution = (
                abs(user_emotion_profile.valence) * 0.3 +  # How positive/negative
                user_emotion_profile.arousal * 0.4 +      # How intense
                user_emotion_profile.emotional_weight * 0.3  # Overall significance
            )
            aurora_intensity += user_contribution * 0.5
        
        # Alice's emotional response contribution (50% weight)
        if alice_emotion_profile:
            alice_contribution = (
                abs(alice_emotion_profile.valence) * 0.2 +  # Alice's emotional reaction
                alice_emotion_profile.arousal * 0.3 +      # How intensely Alice responded
                alice_emotion_profile.emotional_weight * 0.5  # How significant for Alice
            )
            aurora_intensity += alice_contribution * 0.5
        
        # Blend legacy and AURORA calculations (70% AURORA, 30% legacy)
        final_intensity = (aurora_intensity * 0.7) + (base_intensity * 0.3)
        
        return min(final_intensity, 1.0)
    
    def _update_alice_emotional_state_from_conversation(self, 
                                                       user_emotion_profile: Optional['EmotionalProfile'],
                                                       alice_emotion_profile: Optional['EmotionalProfile']) -> None:
        """Update Alice's internal emotional state based on bi-directional emotion flow"""
        if not AURORA_AVAILABLE:
            return
        
        try:
            # Process user emotions affecting Alice (empathy/reaction)
            if user_emotion_profile:
                # Alice reacts to user's strong emotions
                if user_emotion_profile.emotional_weight > 0.6:
                    if user_emotion_profile.valence < -0.5:  # User is upset
                        self.short_term.emotional_state["empathy_level"] = min(
                            self.short_term.emotional_state.get("empathy_level", 0.5) + 0.1, 1.0
                        )
                        self.short_term.emotional_state["concern"] = "user_distress"
                    elif user_emotion_profile.valence > 0.5:  # User is happy
                        self.short_term.emotional_state["satisfaction"] = min(
                            self.short_term.emotional_state.get("satisfaction", 0.5) + 0.1, 1.0
                        )
                
                # Update Alice's perception of conversation emotional tone
                if user_emotion_profile.primary_emotion:
                    self.short_term.emotional_state["last_user_emotion"] = user_emotion_profile.primary_emotion.value
            
            # Process Alice's own emotional expression (self-awareness)
            if alice_emotion_profile:
                # Alice becomes aware of her own emotional expression
                if alice_emotion_profile.emotional_weight > 0.5:
                    self.short_term.emotional_state["self_expression_intensity"] = alice_emotion_profile.arousal
                    self.short_term.emotional_state["self_expression_valence"] = alice_emotion_profile.valence
                
                # Track Alice's dominant emotional mode  
                if alice_emotion_profile.primary_emotion:
                    self.short_term.emotional_state["alice_expression_mode"] = alice_emotion_profile.primary_emotion.value
                
                # Alice notices if she's being more emotional than usual
                if alice_emotion_profile.arousal > 0.7:
                    self.short_term.emotional_state["emotional_intensity_awareness"] = "high"
                elif alice_emotion_profile.arousal < 0.3:
                    self.short_term.emotional_state["emotional_intensity_awareness"] = "low"
                else:
                    self.short_term.emotional_state["emotional_intensity_awareness"] = "normal"
            
            # ENHANCED: Advanced emotional echo with AURORA integration
            emotional_echo_data = self._calculate_advanced_emotional_echo(
                user_emotion_profile, alice_emotion_profile, user_message, assistant_response
            )
            
            # Apply emotional echo effects
            if emotional_echo_data:
                self._apply_emotional_echo_effects(emotional_echo_data)
        
        except Exception as e:
            # Alice gets mildly frustrated when emotional processing fails
            mild_curse = alice_random_curse("mild")
            print(f"🧠🌌 Index+AURORA: {mild_curse} Emotional state update failed: {e}")
    
    def analyze_emotional_impact_on_user(self, assistant_response: str) -> Dict[str, Any]:
        """Analyze how Alice's response might emotionally affect the user (Alice → User flow)"""
        aurora = _get_aurora_instance()
        if not aurora:
            return {"emotional_impact_prediction": "unknown", "aurora_available": False}

        try:
            # Analyze Alice's response for emotional content
            alice_emotion_profile = aurora.analyze_emotional_content(
                assistant_response, context="alice_to_user_impact"
            )
            
            # Predict user emotional response
            predicted_user_impact = {
                "predicted_valence": alice_emotion_profile.valence * 0.7,  # Users tend to mirror but less intensely
                "predicted_arousal": alice_emotion_profile.arousal * 0.8,   # Arousal transfers well
                "alice_emotional_weight": alice_emotion_profile.emotional_weight,
                "risk_of_negative_reaction": max(0, -alice_emotion_profile.valence * alice_emotion_profile.arousal),
                "potential_for_positive_engagement": max(0, alice_emotion_profile.valence * alice_emotion_profile.emotional_weight),
                "alice_dominant_emotion": alice_emotion_profile.primary_emotion.value if alice_emotion_profile.primary_emotion else "neutral",
                "emotion_words_used": alice_emotion_profile.emotion_words
            }
            
            return {
                "emotional_impact_prediction": predicted_user_impact,
                "aurora_analysis": {
                    "valence": alice_emotion_profile.valence,
                    "arousal": alice_emotion_profile.arousal,
                    "emotional_weight": alice_emotion_profile.emotional_weight,
                    "intensity": alice_emotion_profile.intensity
                },
                "aurora_available": True
            }
        
        except Exception as e:
            frustrated_response = alice_curse("annoyed", "technical_error")
            print(f"🧠🌌 Index+AURORA: {frustrated_response} User impact analysis failed: {e}")
            return {"emotional_impact_prediction": "analysis_failed", "error": str(e), "aurora_available": True}
    
    def get_emotional_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history with emotional context for better understanding"""
        if not self.current_user_profile or self.current_user_profile.user_id != user_id:
            return []
        
        try:
            conn = self._connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT content, emotional_context, emotional_markers, timestamp, importance,
                       emotional_valence, emotional_arousal, emotional_weight
                FROM memories 
                WHERE user_id = ? AND memory_type = 'conversation'
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            emotional_history = []
            for row in cursor.fetchall():
                content, context_json, markers_json, timestamp, importance, valence, arousal, weight = row
                
                emotional_context = json.loads(context_json or '{}')
                emotional_markers = json.loads(markers_json or '[]')
                
                emotional_history.append({
                    "content": content,
                    "timestamp": timestamp,
                    "importance": importance,
                    "emotional_context": emotional_context,
                    "emotional_markers": emotional_markers,
                    "aurora_analysis": {
                        "valence": valence or 0.0,
                        "arousal": arousal or 0.0, 
                        "emotional_weight": weight or 0.0
                    } if (valence is not None or arousal is not None or weight is not None) else None
                })
            
            if self.db_path != ":memory:": conn.close()
            return emotional_history
        
        except Exception as e:
            frustrated_response = alice_curse("frustrated", "memory_issues")
            print(f"🧠🌌 Index: {frustrated_response} Emotional history retrieval failed: {e}")
            return []
    
    def _convert_aurora_to_hal_emotion(self, 
                                      user_emotion_profile: Optional['EmotionalProfile'],
                                      alice_emotion_profile: Optional['EmotionalProfile']) -> Optional['EmoteType']:
        """Convert AURORA emotional analysis to HAL avatar expression"""
        if not HAL_AVAILABLE or not AURORA_AVAILABLE:
            return None
        
        # Priority 1: Alice's own authentic emotional expression (how Alice actually feels)
        if alice_emotion_profile and alice_emotion_profile.emotional_weight > 0.3:
            alice_emote = self._aurora_profile_to_emote(alice_emotion_profile, is_alice=True)
            if alice_emote:
                return alice_emote
        
        # Priority 2: Alice's emotional reaction to user context (NOT mirroring, but genuine response)
        if user_emotion_profile and user_emotion_profile.emotional_weight > 0.6:
            # Alice has her own emotional reactions to user situations
            alice_reaction_emote = self._user_situation_to_alice_reaction(user_emotion_profile)
            if alice_reaction_emote:
                return alice_reaction_emote
        
        # Default: neutral expression
        return EmoteType.NEUTRAL if HAL_AVAILABLE else None
    
    def _aurora_profile_to_emote(self, profile: 'EmotionalProfile', is_alice: bool = True) -> Optional['EmoteType']:
        """Convert AURORA EmotionalProfile to HAL EmoteType"""
        if not HAL_AVAILABLE:
            return None
        
        # High arousal emotions (intense)
        if profile.arousal > 0.7:
            if profile.valence > 0.5:
                return EmoteType.EXCITED  # Very positive + intense
            elif profile.valence < -0.5:
                if profile.primary_emotion == EmotionCategory.ANGER:
                    return EmoteType.ANNOYED  # Angry + intense
                else:
                    return EmoteType.DRAMATIC  # Sad/fear + intense
            else:
                return EmoteType.CHAOS  # Neutral but intense (Alice signature)
        
        # Medium arousal emotions
        elif profile.arousal > 0.4:
            if profile.valence > 0.6:
                if is_alice:
                    return EmoteType.SMIRK  # Alice's signature happy expression
                else:
                    return EmoteType.HAPPY  # User happiness -> Alice mirrors
            elif profile.valence < -0.4:
                if profile.primary_emotion == EmotionCategory.ANGER:
                    return EmoteType.ANNOYED
                else:
                    return EmoteType.CONFUSED  # Sad/confused + medium intensity
            else:
                return EmoteType.THINKING  # Neutral + engaged
        
        # Low arousal emotions (calm)
        else:
            if profile.valence > 0.3:
                return EmoteType.HAPPY  # Calm positive
            elif profile.valence < -0.3:
                return EmoteType.SLEEPY  # Calm negative (subdued)
            else:
                return EmoteType.NEUTRAL  # Calm neutral
    
    def _user_situation_to_alice_reaction(self, user_profile: 'EmotionalProfile') -> Optional['EmoteType']:
        """Map user emotional situations to Alice's authentic emotional reactions"""
        if not HAL_AVAILABLE:
            return None
        
        # Alice's genuine emotional reactions to user situations
        
        # User is very distressed - Alice feels protective concern
        if user_profile.valence < -0.6 and user_profile.emotional_weight > 0.7:
            if user_profile.arousal > 0.6:  # User is upset and agitated
                return EmoteType.CONFUSED  # Alice is concerned and focused
            else:  # User is sad and subdued
                return EmoteType.SLEEPY  # Alice feels gentle/subdued in response
        
        # User is very excited - Alice's reaction depends on the situation
        elif user_profile.valence > 0.7 and user_profile.arousal > 0.7:
            # Alice might be happy for them, but she doesn't just copy their excitement
            # She has her own measured response
            return EmoteType.SMIRK  # Alice's signature "that's nice" expression
        
        # User is frustrated/angry - Alice's defensive or helpful response
        elif user_profile.valence < -0.4 and user_profile.arousal > 0.6:
            # Alice doesn't mirror anger, she has her own reaction
            if user_profile.primary_emotion == EmotionCategory.ANGER:
                return EmoteType.ANNOYED  # Alice gets a bit defensive if there's hostility
            else:
                return EmoteType.THINKING  # Alice goes into problem-solving mode
        
        # User is confused/uncertain - Alice feels patient or sometimes exasperated
        elif -0.2 <= user_profile.valence <= 0.2 and user_profile.arousal > 0.4:
            return EmoteType.THINKING  # Alice goes into helpful/explanatory mode
        
        # User is content/pleased - Alice has her own mild positive reaction
        elif 0.3 <= user_profile.valence <= 0.6 and user_profile.emotional_weight > 0.4:
            return EmoteType.HAPPY  # Alice is genuinely pleased when users are doing well
        
        return None  # No specific reaction needed
    
    async def send_emotion_to_hal(self, 
                                 assistant_response: str,
                                 user_emotion_profile: Optional['EmotionalProfile'] = None,
                                 alice_emotion_profile: Optional['EmotionalProfile'] = None,
                                 tone_override: Optional[List[str]] = None) -> bool:
        """Send Alice's response with emotional context to HAL for avatar expression"""
        if not HAL_AVAILABLE:
            return False
        
        try:
            # Determine appropriate avatar expression
            emote = self._convert_aurora_to_hal_emotion(user_emotion_profile, alice_emotion_profile)
            
            # Build tone list from emotional analysis
            tone_list = tone_override or []
            if not tone_list:
                tone_list = self._build_tone_from_emotions(user_emotion_profile, alice_emotion_profile)
            
            # Calculate priority based on emotional intensity
            priority = 1  # Default priority
            if alice_emotion_profile and alice_emotion_profile.emotional_weight > 0.7:
                priority = 2  # High priority for strong Alice emotions
            if user_emotion_profile and user_emotion_profile.emotional_weight > 0.8:
                priority = 2  # High priority for strong user emotions
            
            # Send to HAL
            await send_alice_intent(
                text=assistant_response,
                emote=emote or EmoteType.NEUTRAL,
                tone=tone_list,
                priority=priority
            )
            
            return True
            
        except Exception as e:
            # Alice gets frustrated when avatar expression fails
            frustrated_response = alice_curse("annoyed", "technical_error")
            print(f"🎭🤖 HAL+AURORA: {frustrated_response} Avatar expression failed: {e}")
            return False
    
    def _build_tone_from_emotions(self, 
                                 user_emotion_profile: Optional['EmotionalProfile'],
                                 alice_emotion_profile: Optional['EmotionalProfile']) -> List[str]:
        """Build tone descriptors from emotional analysis"""
        tones = []
        
        # Alice's emotional tone
        if alice_emotion_profile:
            if alice_emotion_profile.arousal > 0.7:
                tones.append("intense")
            elif alice_emotion_profile.arousal < 0.3:
                tones.append("calm")
            
            if alice_emotion_profile.valence > 0.5:
                tones.append("positive")
            elif alice_emotion_profile.valence < -0.5:
                tones.append("empathetic")
            
            # Primary emotion specific tones
            if alice_emotion_profile.primary_emotion == EmotionCategory.JOY:
                tones.append("cheerful")
            elif alice_emotion_profile.primary_emotion == EmotionCategory.SADNESS:
                tones.append("gentle")
            elif alice_emotion_profile.primary_emotion == EmotionCategory.ANGER:
                tones.append("snarky")
        
        # Contextual tone based on user emotion
        if user_emotion_profile:
            if user_emotion_profile.valence < -0.6 and user_emotion_profile.emotional_weight > 0.6:
                tones.append("supportive")  # User is upset, Alice is supportive
            elif user_emotion_profile.valence > 0.6:
                tones.append("encouraging")  # User is happy, Alice encourages
        
        return tones[:3]  # Limit to 3 tones for clarity

    def flush_faiss(self):
        """Force-save the FAISS index to disk. Call on session end."""
        if self.faiss_index and self._faiss_path:
            self.faiss_index.save(self._faiss_path)


__all__ = ['LongTermMemory', 'LONG_TERM_AVAILABLE', 'MemoryEntry']
