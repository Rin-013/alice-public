# Copyright 2025 Rin - Alice AI System
"""
Index System
=============

Depth-based memory organization using the Index/Akashic/Curator architecture.
Manages memory at different depths (surface, mid, deep, core).

Extracted from legacy index.py - Dec 2025
"""

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def _depth_value(depth) -> str:
    """Memory.__getattribute__ hands enums back as strings (legacy SQLite
    shim), so depth may arrive as either MemoryDepth or str. Accept both."""
    return depth.value if hasattr(depth, "value") else depth

# Import types
from ..types import Memory, MemoryDepth

# Use Memory as IndexMemory for backward compatibility
IndexMemory = Memory

# Import subsystems
from .vector import FAISSMemoryIndex, EmotionalMemoryAccessLayer, FAISS_AVAILABLE, EMBEDDINGS_AVAILABLE

INDEX_SYSTEM_AVAILABLE = True


class IndexSystem:
    """
    Memory v2.0: Alice's subjective lived memory with depth layers
    Enhanced with FAISS indexing for <10ms retrieval
    """

    def __init__(self, db_path: str = "alice/data/databases/alice_memory.db"):
        # Handle SQLite's special :memory: database
        self.db_path = db_path if db_path == ":memory:" else Path(db_path)
        self._memory_conn = None

        # Initialize FAISS memory index for high-performance search
        self.faiss_index = FAISSMemoryIndex() if FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE else None

        # Rebuild FAISS index from persisted memories
        if self.faiss_index:
            self._rebuild_faiss_index()

        # Initialize always-hot emotional memory access layer
        self.emotional_layer = EmotionalMemoryAccessLayer(self)

        # Performance tracking
        self.search_times = []
        self.last_sync_time = time.time()

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from persisted memories in SQLite database"""
        if not self.faiss_index:
            return

        try:
            conn = self._connect()
            cursor = conn.cursor()

            # Get all memories from database
            cursor.execute('''
                SELECT event_id, content,
                       alice_emotional_valence, alice_emotional_arousal, alice_emotional_weight,
                       user_emotional_valence, user_emotional_arousal, user_emotional_weight,
                       influence_short, influence_long, depth
                FROM index_memories
                ORDER BY timestamp ASC
            ''')

            memories = cursor.fetchall()
            if self.db_path != ":memory:":
                conn.close()

            rebuilt_count = 0
            for memory_row in memories:
                event_id = memory_row[0]
                content = memory_row[1]

                # Build emotional_data dict
                emotional_data = {
                    'valence': memory_row[2],
                    'arousal': memory_row[3],
                    'weight': memory_row[4],
                    'user_valence': memory_row[5],
                    'user_arousal': memory_row[6],
                    'user_weight': memory_row[7],
                    'influence_short': memory_row[8],
                    'influence_long': memory_row[9],
                    'depth': memory_row[10]
                }

                # Add to FAISS index
                if self.faiss_index.add_memory(event_id, content, emotional_data):
                    rebuilt_count += 1

            if rebuilt_count > 0:
                print(f"🔄 FAISS index rebuilt: {rebuilt_count} memories indexed from database")

        except Exception as e:
            print(f"⚠️  FAISS rebuild failed: {e}")

    def _connect(self):
        """Get database connection (persistent for :memory:)"""
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._memory_conn
        else:
            return sqlite3.connect(self.db_path)

    def add_memory(self, memory: IndexMemory) -> str:
        """Add subjective memory to Index with depth calculation and FAISS indexing"""
        # Calculate influence scores and determine depth
        memory.calculate_influence_scores()
        memory.depth = memory.determine_depth()

        # Set initial hierarchical context layer
        if memory.context_layer is None or memory.context_layer == "":
            memory.context_layer = "active"
        if memory.layer_timestamp is None:
            memory.layer_timestamp = memory.timestamp

        # Store in SQLite
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO index_memories (
                event_id, timestamp, content, depth, user_id,
                emotional_context, emotional_markers,
                emotion_intensity, repetition_freq, salience, uniqueness,
                contrast, divergence_impact, choice_impact,
                influence_short, influence_long,
                access_count, last_accessed, edit_resistance, decay_floor,
                tags, divergence_flag, divergence_id, review_scheduled,
                user_emotional_valence, user_emotional_arousal, user_emotional_weight,
                alice_emotional_valence, alice_emotional_arousal, alice_emotional_weight,
                emotional_mismatch_score,
                context_layer, layer_timestamp, is_compressed, compression_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.event_id, memory.timestamp, memory.content, _depth_value(memory.depth),
            memory.user_id, json.dumps(memory.emotional_context_data),
            json.dumps(memory.emotional_markers),
            memory.emotion_intensity, memory.repetition_freq, memory.salience,
            memory.uniqueness, memory.contrast, memory.divergence_impact,
            memory.choice_impact, memory.influence_short, memory.influence_long,
            memory.access_count, memory.last_accessed, memory.edit_resistance,
            memory.decay_floor, json.dumps(memory.tags or []),
            memory.divergence_flag, memory.divergence_id, memory.review_scheduled,
            memory.user_emotional_valence, memory.user_emotional_arousal, memory.user_emotional_weight,
            memory.alice_emotional_valence, memory.alice_emotional_arousal, memory.alice_emotional_weight,
            memory.emotional_mismatch_score,
            memory.context_layer, memory.layer_timestamp, memory.is_compressed, memory.compression_summary
        ))
        
        conn.commit()
        if self.db_path != ":memory:": conn.close()
        
        # Add to FAISS index for fast retrieval
        if self.faiss_index:
            emotional_data = {
                'valence': memory.alice_emotional_valence,
                'arousal': memory.alice_emotional_arousal,  
                'weight': memory.alice_emotional_weight,
                'user_valence': memory.user_emotional_valence,
                'user_arousal': memory.user_emotional_arousal,
                'user_weight': memory.user_emotional_weight,
                'influence_short': memory.influence_short,
                'influence_long': memory.influence_long,
                'depth': _depth_value(memory.depth)
            }
            
            success = self.faiss_index.add_memory(
                memory.event_id,
                memory.content,
                emotional_data
            )

            if not success:
                logger.warning(f"Failed to add memory {memory.event_id} to FAISS index")
        
        return memory.event_id
    
    def get_memory(self, event_id: str) -> Optional[IndexMemory]:
        """Get Alice's memory by event ID"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM index_memories WHERE event_id = ?', (event_id,))
        row = cursor.fetchone()
        if self.db_path != ":memory:": conn.close()
        
        if row:
            return self._row_to_index_memory(row)
        return None
    
    def get_memories_by_depth(self, depth: MemoryDepth, user_id: str, 
                             limit: int = 20) -> List[IndexMemory]:
        """Get memories from specific depth layer"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM index_memories 
            WHERE depth = ? AND user_id = ?
            ORDER BY influence_long DESC, timestamp DESC 
            LIMIT ?
        ''', (_depth_value(depth), user_id, limit))
        
        memories = []
        for row in cursor.fetchall():
            memory = self._row_to_index_memory(row)
            memory.access_count += 1
            memory.last_accessed = time.time()
            memories.append(memory)
        
        # Update access tracking
        self._update_access_tracking(memories, cursor)
        conn.commit()
        if self.db_path != ":memory:": conn.close()
        return memories
    
    def get_deep_memories(self, user_id: str, limit: int = 10) -> List[IndexMemory]:
        """Get Alice's deepest, most identity-shaping memories"""
        deep_memories = self.get_memories_by_depth(MemoryDepth.DEEP, user_id, limit//2)
        core_memories = self.get_memories_by_depth(MemoryDepth.CORE, user_id, limit//2)
        return core_memories + deep_memories
    
    def update_salience_decay(self, user_id: str, decay_rate: float = 0.1):
        """Update salience scores with time-based decay"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Get all memories for user
        cursor.execute('''
            SELECT event_id, salience, timestamp FROM index_memories 
            WHERE user_id = ?
        ''', (user_id,))
        
        current_time = time.time()
        updates = []
        
        for event_id, salience, timestamp in cursor.fetchall():
            # Exponential decay based on time
            days_old = (current_time - timestamp) / (24 * 3600)
            new_salience = salience * math.exp(-decay_rate * days_old)
            updates.append((new_salience, event_id))
        
        # Batch update
        cursor.executemany('''
            UPDATE index_memories 
            SET salience = ?
            WHERE event_id = ?
        ''', updates)
        
        conn.commit()
        if self.db_path != ":memory:": conn.close()
        return len(updates)
    
    def fast_search_memories(self, query: str, k: int = 10, 
                           emotional_filter: Optional[Dict[str, Any]] = None,
                           depth_filter: Optional[List[MemoryDepth]] = None) -> List[IndexMemory]:
        """
        High-performance memory search using FAISS index
        Target: <10ms retrieval time
        """
        start_time = time.perf_counter()
        
        # Try FAISS first for semantic search
        if self.faiss_index:
            try:
                # Get event IDs from FAISS
                faiss_results = self.faiss_index.search_memories(
                    query, k * 2, emotional_filter
                )
                
                if faiss_results:
                    # Retrieve full memory objects from SQLite
                    memories = []
                    for event_id, similarity in faiss_results:
                        memory = self.get_memory(event_id)
                        if memory:
                            # Apply depth filter if specified
                            if depth_filter and memory.depth not in depth_filter:
                                continue
                            
                            # Add similarity score for ranking
                            memory.similarity_score = similarity
                            memories.append(memory)
                            
                            if len(memories) >= k:
                                break
                    
                    # Track performance
                    search_time = (time.perf_counter() - start_time) * 1000
                    self.search_times.append(search_time)
                    
                    # Keep only recent search times for rolling average
                    if len(self.search_times) > 100:
                        self.search_times = self.search_times[-100:]
                    
                    if search_time > 10.0:  # Log if over target
                        print(f"⚠️  FAISS search took {search_time:.2f}ms (target: <10ms)")
                    
                    return memories
                
            except Exception as e:
                print(f"⚠️  FAISS search failed, trying IRIS semantic search: {e}")

        # Fallback 1: Try IRIS semantic search (lightweight, no ML)
        if hasattr(self, 'iris_index') and self.iris_index:
            try:
                from .iris import SearchContext
                search_context = SearchContext(
                    query=query,
                    user_id="default",
                    emotional_state=emotional_filter or {},
                    time_preference="recent"
                )
                iris_results = self.iris_index.smart_search(search_context, limit=k)

                if iris_results:
                    memories = []
                    for match in iris_results:
                        memory = self.get_memory(match.memory_id)
                        if memory:
                            if depth_filter and memory.depth not in depth_filter:
                                continue
                            memory.similarity_score = match.relevance_score
                            memories.append(memory)
                            if len(memories) >= k:
                                break

                    if memories:
                        search_time = (time.perf_counter() - start_time) * 1000
                        print(f"✅ IRIS semantic search succeeded in {search_time:.2f}ms")
                        return memories

            except Exception as e:
                print(f"⚠️  IRIS search failed, falling back to SQLite keywords: {e}")

        # Fallback 2: SQLite keyword search (last resort)
        return self._fallback_search(query, k, emotional_filter, depth_filter, start_time)
    
    def _fallback_search(self, query: str, k: int,
                        emotional_filter: Optional[Dict[str, Any]],
                        depth_filter: Optional[List[MemoryDepth]],
                        start_time: float) -> List[IndexMemory]:
        """Fallback keyword search using SQLite with improved matching"""
        conn = self._connect()
        cursor = conn.cursor()

        # Extract key terms from query for better matching
        query_words = re.findall(r'\b\w+\b', query.lower())
        # Remove common words that don't help with matching
        stop_words = {'what', 'is', 'the', 'my', 'your', 'are', 'do', 'does', 'how', 'when', 'where', 'why', 'who', 'can', 'could', 'would', 'should'}
        key_words = [word for word in query_words if word not in stop_words and len(word) > 2]

        # Build query conditions - search for key terms
        if key_words:
            # Search for any of the key words in content
            word_conditions = []
            word_params = []
            for word in key_words:
                word_conditions.append("content LIKE ?")
                word_params.append(f'%{word}%')

            conditions = [f"({' OR '.join(word_conditions)})"]
            params = word_params
        else:
            # Fallback to original query if no key words extracted
            conditions = ["content LIKE ?"]
            params = [f'%{query}%']
        
        if depth_filter:
            depth_placeholders = ','.join('?' * len(depth_filter))
            conditions.append(f"depth IN ({depth_placeholders})")
            params.extend([d.value for d in depth_filter])
        
        if emotional_filter:
            # Add basic emotional filtering
            if 'valence_range' in emotional_filter:
                min_val, max_val = emotional_filter['valence_range']
                conditions.extend([
                    "alice_emotional_valence >= ?",
                    "alice_emotional_valence <= ?"
                ])
                params.extend([min_val, max_val])
        
        where_clause = " AND ".join(conditions)
        
        cursor.execute(f'''
            SELECT * FROM index_memories 
            WHERE {where_clause}
            ORDER BY influence_long DESC, influence_short DESC, timestamp DESC
            LIMIT ?
        ''', params + [k])
        
        memories = []
        for row in cursor.fetchall():
            memory = self._row_to_index_memory(row)
            memory.similarity_score = 0.5  # Default similarity for keyword match
            memories.append(memory)
        
        if self.db_path != ":memory:": conn.close()
        
        # Track fallback performance
        search_time = (time.perf_counter() - start_time) * 1000
        self.search_times.append(search_time)
        
        return memories
    
    def get_search_performance_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        if not self.search_times:
            return {"no_searches": True}
        
        avg_time = sum(self.search_times) / len(self.search_times)
        min_time = min(self.search_times)
        max_time = max(self.search_times)
        
        faiss_stats = self.faiss_index.get_stats() if self.faiss_index else {"available": False}
        
        return {
            "average_search_time_ms": round(avg_time, 2),
            "min_search_time_ms": round(min_time, 2),
            "max_search_time_ms": round(max_time, 2),
            "total_searches": len(self.search_times),
            "target_met": avg_time < 10.0,
            "faiss_index": faiss_stats
        }
    
    def _row_to_index_memory(self, row) -> IndexMemory:
        """Convert database row to IndexMemory object"""
        # Handle both old rows (missing hierarchical fields) and new rows
        context_layer = row[31] if len(row) > 31 else "active"
        layer_timestamp = row[32] if len(row) > 32 else None
        is_compressed = bool(row[33]) if len(row) > 33 else False
        compression_summary = row[34] if len(row) > 34 else None

        return IndexMemory(
            event_id=row[0], timestamp=row[1], content=row[2],
            depth=MemoryDepth(row[3]), user_id=row[4],
            emotional_context_data=json.loads(row[5] or '{}'),
            emotional_markers=json.loads(row[6] or '[]'),
            emotion_intensity=row[7], repetition_freq=row[8],
            salience=row[9], uniqueness=row[10], contrast=row[11],
            divergence_impact=row[12], choice_impact=row[13],
            influence_short=row[14], influence_long=row[15],
            access_count=row[16], last_accessed=row[17],
            edit_resistance=row[18], decay_floor=row[19],
            tags=json.loads(row[20] or '[]'),
            divergence_flag=bool(row[21]), divergence_id=row[22],
            review_scheduled=row[23],
            user_emotional_valence=row[24] if len(row) > 24 else None,
            user_emotional_arousal=row[25] if len(row) > 25 else None,
            user_emotional_weight=row[26] if len(row) > 26 else None,
            alice_emotional_valence=row[27] if len(row) > 27 else None,
            alice_emotional_arousal=row[28] if len(row) > 28 else None,
            alice_emotional_weight=row[29] if len(row) > 29 else None,
            emotional_mismatch_score=row[30] if len(row) > 30 else None,
            context_layer=context_layer,
            layer_timestamp=layer_timestamp,
            is_compressed=is_compressed,
            compression_summary=compression_summary
        )
    
    def _update_access_tracking(self, memories: List[IndexMemory], cursor):
        """Update access tracking for memories"""
        for memory in memories:
            cursor.execute('''
                UPDATE index_memories
                SET access_count = ?, last_accessed = ?
                WHERE event_id = ?
            ''', (memory.access_count, memory.last_accessed, memory.event_id))

    # ============================================================================
    # HIERARCHICAL CONTEXT LAYER MANAGEMENT
    # ============================================================================
    # Methods for managing memory layers (active/recent/archived)
    # Part of hierarchical context window architecture

    def get_memories_by_layer(self, layer: str, user_id: str, limit: int = 20) -> List[IndexMemory]:
        """
        Retrieve memories from a specific context layer

        Args:
            layer: "active", "recent", or "archived"
            user_id: User to get memories for
            limit: Maximum memories to return

        Returns:
            List of memories from specified layer, sorted by timestamp (newest first)
        """
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM index_memories
            WHERE context_layer = ? AND user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (layer, user_id, limit))

        rows = cursor.fetchall()
        if self.db_path != ":memory:": conn.close()

        return [self._row_to_index_memory(row) for row in rows]

    def move_memory_to_layer(self, event_id: str, new_layer: str) -> bool:
        """
        Move a memory to a different context layer

        Args:
            event_id: Memory to move
            new_layer: "active", "recent", or "archived"

        Returns:
            True if successful
        """
        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE index_memories
                SET context_layer = ?, layer_timestamp = ?
                WHERE event_id = ?
            ''', (new_layer, time.time(), event_id))

            conn.commit()
            if self.db_path != ":memory:": conn.close()
            return True

        except Exception as e:
            print(f"⚠️ Failed to move memory {event_id} to layer {new_layer}: {e}")
            if self.db_path != ":memory:": conn.close()
            return False

    def compress_and_archive(self, event_id: str, summary: str) -> bool:
        """
        Compress a memory and move it to archived layer

        Args:
            event_id: Memory to compress
            summary: Compressed summary of the memory

        Returns:
            True if successful
        """
        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE index_memories
                SET context_layer = ?,
                    layer_timestamp = ?,
                    is_compressed = ?,
                    compression_summary = ?
                WHERE event_id = ?
            ''', ("archived", time.time(), True, summary, event_id))

            conn.commit()
            if self.db_path != ":memory:": conn.close()
            return True

        except Exception as e:
            print(f"⚠️ Failed to compress and archive memory {event_id}: {e}")
            if self.db_path != ":memory:": conn.close()
            return False

    def run_temporal_decay(self, user_id: str, active_threshold_minutes: int = 5,
                          recent_threshold_minutes: int = 60,
                          access_count_threshold: int = 3) -> Dict[str, int]:
        """
        Run temporal decay - move old memories between layers based on time AND access patterns

        Considers both time and access frequency:
        - Frequently accessed memories stay in active/recent longer
        - Deep memories (CORE/DEEP) resist decay more
        - High salience memories are "stickier"

        Args:
            user_id: User to run decay for
            active_threshold_minutes: Move to recent after this many minutes (default: 5)
            recent_threshold_minutes: Move to archived after this many minutes (default: 60)
            access_count_threshold: Don't move if accessed this many times (default: 3)

        Returns:
            Dict with counts: {"active_to_recent": N, "recent_to_archived": M}
        """
        conn = self._connect()
        cursor = conn.cursor()
        current_time = time.time()

        active_cutoff = current_time - (active_threshold_minutes * 60)
        recent_cutoff = current_time - (recent_threshold_minutes * 60)

        # Move old active → recent (but NOT if frequently accessed or DEEP/CORE memories)
        cursor.execute('''
            UPDATE index_memories
            SET context_layer = ?, layer_timestamp = ?
            WHERE context_layer = ?
              AND user_id = ?
              AND layer_timestamp < ?
              AND access_count < ?
              AND depth NOT IN ('deep', 'core')
        ''', ("recent", current_time, "active", user_id, active_cutoff, access_count_threshold))
        active_moved = cursor.rowcount

        # Move old recent → archived (but NOT if frequently accessed or CORE memories)
        cursor.execute('''
            UPDATE index_memories
            SET context_layer = ?, layer_timestamp = ?
            WHERE context_layer = ?
              AND user_id = ?
              AND layer_timestamp < ?
              AND access_count < ?
              AND depth != 'core'
        ''', ("archived", current_time, "recent", user_id, recent_cutoff, access_count_threshold))
        recent_moved = cursor.rowcount

        conn.commit()
        if self.db_path != ":memory:": conn.close()

        return {
            "active_to_recent": active_moved,
            "recent_to_archived": recent_moved
        }




__all__ = ['IndexSystem', 'INDEX_SYSTEM_AVAILABLE', 'IndexMemory']
