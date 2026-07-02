# Copyright 2025 Rin - Alice AI System
"""
Vector Memory Storage (FAISS)
=============================

High-performance FAISS-based memory indexing for <10ms semantic retrieval.
Includes emotional memory access layer with background sync.

Extracted from legacy index.py - Dec 2025
"""

import json
import os
import sqlite3
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Debug mode - only print verbose output when ALICE_DEBUG=1
DEBUG_MODE = os.environ.get('ALICE_DEBUG', '0') == '1'

# Import types from unified types module
from ..types import Memory, MemoryDepth

# Use Memory as IndexMemory for backward compatibility
IndexMemory = Memory

# FAISS vector indexing
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    np = None
    FAISS_AVAILABLE = False

# Sentence embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    EMBEDDINGS_AVAILABLE = False

# Shared embedding model
try:
    from ...utils.embedding_utils import get_shared_embedding_model
except ImportError:
    def get_shared_embedding_model():
        if EMBEDDINGS_AVAILABLE:
            return SentenceTransformer('all-MiniLM-L6-v2')
        return None

VECTOR_AVAILABLE = FAISS_AVAILABLE and EMBEDDINGS_AVAILABLE


class FAISSMemoryIndex:
    """
    High-performance FAISS-based memory indexing for <10ms retrieval
    Always-hot emotional memory access layer with background SQLite sync
    """
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

        # Multi-index support: separate indices for different memory types
        self.indices = {}  # index_type -> faiss.Index
        self.index = None  # Legacy: points to 'conversation' index for backward compat

        # Per-index mappings
        self.memory_maps = {}  # index_type -> {faiss_id -> event_id}
        self.event_to_faiss = {}  # event_id -> (index_type, faiss_id)
        self.next_faiss_ids = {}  # index_type -> next_id

        # Legacy single-index mappings (for backward compat)
        self.memory_map = {}  # faiss_id -> event_id mapping

        self.embeddings_cache = {}  # event_id -> embedding cache
        self.emotional_cache = {}  # event_id -> emotional_data cache
        self.query_cache = {}  # query -> embedding cache (NEW: optimization)
        self.query_cache_max = 100  # Limit cache size
        self.lock = threading.RLock()
        self.next_faiss_id = 0  # Legacy

        # Initialize embedding model if available
        self.encoder = None
        self.encoder_available = False
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use local model stored in Alice project (no internet required)
                from pathlib import Path
                # Use shared embedding model (saves 174MB RAM by deduplication!)
                from alice.core.utils.embedding_utils import get_shared_embedding_model
                shared_model = get_shared_embedding_model(lazy_load=False)
                self.encoder = shared_model.get_model()

                self.encoder_available = True

                # Warmup: encode a dummy query to eliminate first-run overhead
                _ = self.encoder.encode(["warmup query"], normalize_embeddings=True)
                print("✅ Encoder warmed up for optimal performance")

            except Exception as e:
                print(f"⚠️  Embedding model failed to load: {e}")
                self.encoder_available = False

        # Initialize FAISS index if available
        if FAISS_AVAILABLE and self.encoder_available:
            self._init_faiss_index()
    
    def _init_faiss_index(self):
        """Initialize FAISS indices for fast semantic search (dual: conversation + fact)"""
        try:
            # Create separate indices for different memory types
            for index_type in ['conversation', 'fact']:
                self.indices[index_type] = faiss.IndexFlatIP(self.embedding_dim)
                self.memory_maps[index_type] = {}  # faiss_id -> event_id
                self.next_faiss_ids[index_type] = 0

            # Legacy: point self.index to conversation index for backward compat
            self.index = self.indices['conversation']

            print(f"✅ FAISS dual indices initialized: {self.embedding_dim}D (conversation + fact)")

        except Exception as e:
            print(f"❌ FAISS initialization failed: {e}")
            self.index = None
            self.indices = {}
    
    def _encode_text(self, text: str, use_cache: bool = True) -> Optional[Any]:
        """
        Convert text to embedding vector with optional caching

        Args:
            text: Text to encode
            use_cache: Whether to use/update query cache (default True for queries)
        """
        if not self.encoder:
            return None

        # Check cache first (for repeated queries)
        if use_cache and text in self.query_cache:
            return self.query_cache[text]

        try:
            # Normalize embeddings for cosine similarity
            embedding = self.encoder.encode([text], normalize_embeddings=True)
            result = embedding[0]

            # Cache query embeddings (not memory embeddings, those are stored separately)
            if use_cache:
                # Limit cache size (FIFO)
                if len(self.query_cache) >= self.query_cache_max:
                    # Remove oldest entry
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                self.query_cache[text] = result

            return result
        except Exception as e:
            print(f"⚠️  Encoding failed: {e}")
            return None
    
    def add_memory(self, event_id: str, content: str, emotional_data: Dict[str, Any],
                   memory_type: str = 'conversation') -> bool:
        """
        Add memory to FAISS index for fast retrieval.

        Args:
            event_id: Unique memory identifier
            content: Text content to embed
            emotional_data: Emotional metadata
            memory_type: 'conversation' or 'fact' - determines which index to use
        """
        # Validate memory_type
        if memory_type not in self.indices:
            memory_type = 'conversation'  # Default fallback

        index = self.indices.get(memory_type)
        if not index or not self.encoder:
            print(f"⚠️  FAISS add_memory failed: index={index is not None}, encoder={self.encoder is not None}")
            return False

        try:
            with self.lock:
                # Skip if already indexed by ID
                if event_id in self.event_to_faiss:
                    return True

                # Generate embedding (don't use query cache for memories)
                embedding = self._encode_text(content, use_cache=False)
                if embedding is None:
                    print(f"⚠️  FAISS add_memory failed: embedding is None for content: {content[:50]}...")
                    return False

                # Get next ID for this index type
                faiss_id = self.next_faiss_ids[memory_type]

                # Add to correct FAISS index
                index.add(embedding.reshape(1, -1))

                # Update per-index mappings
                self.memory_maps[memory_type][faiss_id] = event_id
                self.event_to_faiss[event_id] = (memory_type, faiss_id)
                self.next_faiss_ids[memory_type] += 1

                # Shared caches (content/emotion doesn't depend on index type)
                self.embeddings_cache[event_id] = embedding
                self.emotional_cache[event_id] = emotional_data

                # Legacy mapping (for backward compat with conversation index)
                if memory_type == 'conversation':
                    self.memory_map[faiss_id] = event_id
                    self.next_faiss_id = self.next_faiss_ids['conversation']

                total = sum(idx.ntotal for idx in self.indices.values())
                if DEBUG_MODE:
                    print(f"✅ FAISS indexed [{memory_type}]: {event_id} (total: {total})")
                return True

        except Exception as e:
            if DEBUG_MODE:
                print(f"⚠️  FAISS add failed: {e}")
            return False

    def is_duplicate(self, content: str, memory_type: str = 'conversation',
                     threshold: float = 0.92) -> bool:
        """
        Check if a semantically identical memory already exists.

        Uses inner-product distance on normalized embeddings (equivalent to cosine similarity).
        Returns True if nearest neighbour similarity >= threshold.
        """
        if not FAISS_AVAILABLE or not self.encoder:
            return False

        index_type = memory_type if memory_type in self.indices else 'conversation'
        index = self.indices.get(index_type)
        if index is None or index.ntotal == 0:
            return False

        try:
            embedding = self._encode_text(content, use_cache=False)
            if embedding is None:
                return False
            with self.lock:
                distances, _ = index.search(embedding.reshape(1, -1), k=1)
            sim = float(distances[0][0])
            if sim >= threshold:
                if DEBUG_MODE:
                    print(f"⏭  Dedup skip (sim={sim:.3f}): {content[:60]}")
                return True
            return False
        except Exception:
            return False

    def search_memories(self, query: str, k: int = 10,
                       emotional_filter: Optional[Dict[str, Any]] = None,
                       use_recency_decay: bool = False,
                       index_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Fast semantic search of memories with optional emotional filtering and recency decay.

        Args:
            query: Search query text
            k: Number of results to return
            emotional_filter: Optional emotional criteria for filtering
            use_recency_decay: Apply Alice's personality-based recency decay (OPTIMIZATION 2)
            index_type: 'conversation', 'fact', or None (searches both, weighted)

        Returns:
            List of (event_id, similarity_score) tuples
        """
        if not self.encoder:
            return []

        # Determine which indices to search
        if index_type and index_type in self.indices:
            indices_to_search = [(index_type, self.indices[index_type])]
        else:
            # Search all indices
            indices_to_search = list(self.indices.items())

        if not indices_to_search:
            return []

        try:
            # Encode query
            query_embedding = self._encode_text(query)
            if query_embedding is None:
                return []

            all_results = []

            with self.lock:
                for idx_type, index in indices_to_search:
                    if index.ntotal == 0:
                        continue

                    # Search this FAISS index
                    search_k = k * 3 if (use_recency_decay or emotional_filter) else k * 2
                    similarities, faiss_ids = index.search(
                        query_embedding.reshape(1, -1),
                        min(search_k, index.ntotal)
                    )

                    # Get memory map for this index type
                    memory_map = self.memory_maps.get(idx_type, {})

                    for sim, faiss_id in zip(similarities[0], faiss_ids[0]):
                        if faiss_id == -1:  # No more results
                            break

                        event_id = memory_map.get(faiss_id)
                        if not event_id:
                            # Fallback to legacy map for conversation index
                            if idx_type == 'conversation':
                                event_id = self.memory_map.get(faiss_id)
                            if not event_id:
                                continue

                        # Apply emotional filtering if specified
                        if emotional_filter:
                            emotional_data = self.emotional_cache.get(event_id, {})
                            if not self._matches_emotional_filter(emotional_data, emotional_filter):
                                continue

                        all_results.append((event_id, float(sim)))

                # Sort all results by similarity
                all_results.sort(key=lambda x: x[1], reverse=True)

                # Apply recency decay if requested
                if use_recency_decay:
                    all_results = self.apply_alice_recency_decay(all_results)

                # Return top K
                return all_results[:k]

        except Exception as e:
            print(f"⚠️  FAISS search failed: {e}")
            return []
    
    def _matches_emotional_filter(self, emotional_data: Dict[str, Any], 
                                 filter_criteria: Dict[str, Any]) -> bool:
        """Check if emotional data matches filter criteria"""
        try:
            # Example filters: valence_range, arousal_min, weight_min
            if 'valence_range' in filter_criteria:
                valence = emotional_data.get('valence', 0.0)
                min_val, max_val = filter_criteria['valence_range']
                if not (min_val <= valence <= max_val):
                    return False
            
            if 'arousal_min' in filter_criteria:
                arousal = emotional_data.get('arousal', 0.0)
                if arousal < filter_criteria['arousal_min']:
                    return False
            
            if 'weight_min' in filter_criteria:
                weight = emotional_data.get('weight', 0.0)
                if weight < filter_criteria['weight_min']:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def save(self, path: str) -> bool:
        """
        Persist FAISS indices and mappings to disk.

        Args:
            path: Base path (without extension). Saves:
                  {path}_{index_type}.faiss  - FAISS binary index
                  {path}_mappings.json       - ID mappings and emotional cache
        Returns:
            True on success
        """
        if not FAISS_AVAILABLE or not self.indices:
            return False
        try:
            import pickle
            with self.lock:
                for idx_type, index in self.indices.items():
                    faiss.write_index(index, f"{path}_{idx_type}.faiss")

                mappings = {
                    'memory_maps': self.memory_maps,
                    'event_to_faiss': self.event_to_faiss,
                    'next_faiss_ids': self.next_faiss_ids,
                    'emotional_cache': self.emotional_cache,
                    # Legacy
                    'memory_map': self.memory_map,
                    'next_faiss_id': self.next_faiss_id,
                }
                with open(f"{path}_mappings.pkl", 'wb') as f:
                    pickle.dump(mappings, f)
            return True
        except Exception as e:
            print(f"⚠️ FAISS save failed: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load FAISS indices and mappings from disk.

        Args:
            path: Base path used when save() was called.
        Returns:
            True if loaded successfully
        """
        if not FAISS_AVAILABLE:
            return False
        try:
            import pickle
            import os
            mappings_path = f"{path}_mappings.pkl"
            if not os.path.exists(mappings_path):
                return False

            with self.lock:
                with open(mappings_path, 'rb') as f:
                    mappings = pickle.load(f)

                self.memory_maps = mappings.get('memory_maps', {})
                self.event_to_faiss = mappings.get('event_to_faiss', {})
                self.next_faiss_ids = mappings.get('next_faiss_ids', {})
                self.emotional_cache = mappings.get('emotional_cache', {})
                self.memory_map = mappings.get('memory_map', {})
                self.next_faiss_id = mappings.get('next_faiss_id', 0)

                loaded = 0
                for idx_type in ['conversation', 'fact']:
                    faiss_path = f"{path}_{idx_type}.faiss"
                    if os.path.exists(faiss_path):
                        self.indices[idx_type] = faiss.read_index(faiss_path)
                        loaded += self.indices[idx_type].ntotal
                    else:
                        self.indices[idx_type] = faiss.IndexFlatIP(self.embedding_dim)
                        self.memory_maps.setdefault(idx_type, {})
                        self.next_faiss_ids.setdefault(idx_type, 0)

                # Legacy compat
                self.index = self.indices.get('conversation')

            print(f"✅ FAISS index loaded from disk: {loaded} vectors")
            return True
        except Exception as e:
            print(f"⚠️ FAISS load failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        with self.lock:
            # Per-index stats
            index_stats = {}
            total = 0
            for idx_type, index in self.indices.items():
                count = index.ntotal if index else 0
                index_stats[idx_type] = count
                total += count

            return {
                "available": len(self.indices) > 0,
                "total_memories": total,
                "indices": index_stats,  # {'conversation': N, 'fact': M}
                "embedding_dim": self.embedding_dim,
                "memory_cache_size": len(self.embeddings_cache),
                "emotional_cache_size": len(self.emotional_cache),
                "query_cache_size": len(self.query_cache),
                "query_cache_max": self.query_cache_max
            }

    def remove_memory(self, event_id: str) -> bool:
        """Remove memory from FAISS index"""
        # Note: FAISS doesn't support direct removal, so we'd need to rebuild
        # For now, just remove from caches and mappings
        with self.lock:
            if event_id in self.event_to_faiss:
                mapping = self.event_to_faiss[event_id]
                del self.event_to_faiss[event_id]

                # Handle both old (int) and new (tuple) format
                if isinstance(mapping, tuple):
                    idx_type, faiss_id = mapping
                    if idx_type in self.memory_maps:
                        self.memory_maps[idx_type].pop(faiss_id, None)
                else:
                    # Legacy format: just faiss_id
                    self.memory_map.pop(mapping, None)

            if event_id in self.embeddings_cache:
                del self.embeddings_cache[event_id]

            if event_id in self.emotional_cache:
                del self.emotional_cache[event_id]

            return True

    # ===== EXPERIMENTAL ALICE-SPECIFIC OPTIMIZATIONS =====

    def search_memories_with_emotional_boost(self, query: str,
                                            current_emotion: Optional[Dict[str, Any]] = None,
                                            k: int = 10) -> List[Tuple[str, float]]:
        """
        OPTIMIZATION 1: Emotional Vector Boosting

        Re-rank FAISS search results based on emotional similarity between
        Alice's current emotional state and memory emotions.

        Args:
            query: Search query text
            current_emotion: Alice's current emotional state with keys:
                - primary_emotion: str (e.g., "frustrated", "happy")
                - valence: float (-1.0 to 1.0)
                - arousal: float (0.0 to 1.0)
            k: Number of results to return

        Returns:
            List of (event_id, boosted_score) tuples
        """
        if not current_emotion:
            # Fall back to standard search if no emotional context
            return self.search_memories(query, k=k)

        # Get 3x more results for re-ranking
        base_results = self.search_memories(query, k=k*3)

        if not base_results:
            return []

        current_primary = current_emotion.get('primary_emotion', '').lower()
        current_valence = current_emotion.get('valence', 0.0)
        current_arousal = current_emotion.get('arousal', 0.5)

        boosted_results = []

        for event_id, base_score in base_results:
            emotional_data = self.emotional_cache.get(event_id, {})

            # Start with base score
            boost = 1.0

            # Boost 1: Primary emotion match (strongest signal)
            memory_primary = emotional_data.get('primary_emotion', '').lower()
            if memory_primary and current_primary:
                if memory_primary == current_primary:
                    boost *= 1.3  # 30% boost for exact emotion match

            # Boost 2: Valence alignment (negative/positive match)
            memory_valence = emotional_data.get('valence', 0.0)
            valence_diff = abs(current_valence - memory_valence)
            valence_boost = 1.0 + (1.0 - valence_diff) * 0.15  # Up to 15% boost
            boost *= valence_boost

            # Boost 3: Arousal/intensity alignment
            memory_arousal = emotional_data.get('arousal', 0.5)
            arousal_diff = abs(current_arousal - memory_arousal)
            arousal_boost = 1.0 + (1.0 - arousal_diff) * 0.2  # Up to 20% boost
            boost *= arousal_boost

            boosted_score = base_score * boost
            boosted_results.append((event_id, boosted_score))

        # Sort by boosted scores and return top K
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:k]

    def apply_alice_recency_decay(self, results: List[Tuple[str, float]],
                                  current_time: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        OPTIMIZATION 2: Recency Decay with Personality

        Alice forgets boring memories faster than exciting ones.
        Applies personality-based decay to memory scores based on:
        - Time since memory creation
        - Emotional arousal/excitement level

        Args:
            results: List of (event_id, score) tuples from search
            current_time: Current timestamp (defaults to now)

        Returns:
            List of (event_id, decayed_score) tuples sorted by score
        """
        import time as time_module

        if current_time is None:
            current_time = time_module.time()

        decayed_results = []

        for event_id, score in results:
            # Get memory metadata
            emotional_data = self.emotional_cache.get(event_id, {})

            # Need to get timestamp from somewhere
            # For now, use a simple approach - check if we can get it from metadata
            # TODO: May need to store timestamp in emotional_cache or fetch from DB
            memory_timestamp = emotional_data.get('timestamp', current_time)

            # Calculate age in hours
            age_hours = max(0, (current_time - memory_timestamp) / 3600)

            # Get emotional arousal (excitement level)
            excitement = emotional_data.get('arousal', 0.5)

            # Alice's personality-based decay
            if excitement < 0.3:  # Boring/low arousal
                # Fast decay: boring stuff fades quickly
                decay_factor = 0.95 ** (age_hours * 3)
            elif excitement > 0.7:  # Exciting/high arousal
                # Slow decay: exciting memories stick around
                decay_factor = 0.99 ** (age_hours * 0.5)
            else:  # Normal arousal
                # Medium decay: standard forgetting curve
                decay_factor = 0.97 ** age_hours

            decayed_score = score * decay_factor
            decayed_results.append((event_id, decayed_score))

        # Sort by decayed scores
        decayed_results.sort(key=lambda x: x[1], reverse=True)
        return decayed_results

    def prefetch_by_momentum(self, current_emotion: Optional[str] = None,
                            momentum_direction: Optional[str] = None,
                            tiredness_level: float = 0.0):
        """
        OPTIMIZATION 3: Emotional Momentum-Aware Caching

        Pre-cache likely queries based on Alice's emotional momentum from thought trains.
        Runs asynchronously to prepare embeddings for instant retrieval.

        Args:
            current_emotion: Alice's current primary emotion ("frustrated", "happy", etc.)
            momentum_direction: "rising", "declining", "stable"
            tiredness_level: 0.0-1.0 (higher = more tired)
        """
        if not current_emotion or not momentum_direction:
            return

        # Predict likely queries based on emotional state + momentum
        likely_queries = []

        # Frustrated + declining → looking for past frustrations/solutions
        if current_emotion == "frustrated" and momentum_direction == "declining":
            likely_queries.extend([
                "user frustrated",
                "helped user with frustration",
                "debugging problems",
                "when things went wrong"
            ])

        # Happy + rising → looking for exciting/positive memories
        elif current_emotion == "happy" and momentum_direction == "rising":
            likely_queries.extend([
                "user excited",
                "successful outcomes",
                "when things worked",
                "user happy about results"
            ])

        # Tired → looking for simple/easy solutions
        if tiredness_level > 0.5:
            likely_queries.extend([
                "previous simple tasks",
                "easy solutions",
                "quick fixes"
            ])

        # Bored/low momentum → looking for interesting past conversations
        if momentum_direction == "stable" and tiredness_level < 0.3:
            likely_queries.extend([
                "interesting conversations",
                "creative ideas",
                "fun interactions"
            ])

        # Pre-encode predictions (adds to query cache)
        for query in likely_queries:
            if query not in self.query_cache:
                # This will cache the embedding for instant retrieval later
                _ = self._encode_text(query, use_cache=True)

    def extract_key_concepts(self, text: str, max_concepts: int = 5) -> List[str]:
        """
        Helper for multi-hop chaining: Extract key concepts from memory content.
        Simple keyword extraction based on word frequency and length.
        """
        # Simple approach: extract longer words (likely nouns/concepts)
        words = text.lower().split()
        # Filter: length > 4, not common words
        common_words = {'the', 'and', 'with', 'that', 'this', 'have', 'from', 'they', 'been', 'were', 'what', 'when', 'where'}
        concepts = [w.strip('.,!?;:') for w in words
                   if len(w) > 4 and w not in common_words]

        # Return unique concepts, limited to max_concepts
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append(concept)
                if len(unique_concepts) >= max_concepts:
                    break

        return unique_concepts

    def get_memory_chain(self, initial_query: str,
                        max_hops: int = 3,
                        decay_per_hop: float = 0.8) -> List[Tuple[str, float, int]]:
        """
        OPTIMIZATION 4: Multi-Hop Memory Chaining

        Follow memory associations like thought trains.
        Builds a chain of related memories by following concept associations.

        Args:
            initial_query: Starting search query
            max_hops: Maximum number of association hops
            decay_per_hop: Score multiplier for each hop distance

        Returns:
            List of (event_id, score, hop_number) tuples forming a memory chain
        """
        chain = []
        visited = set()

        # Hop 1: Initial search
        initial_results = self.search_memories(initial_query, k=5)
        if not initial_results:
            return []

        # Add top result to chain
        event_id, score = initial_results[0]
        chain.append((event_id, score, 0))  # hop 0
        visited.add(event_id)

        # Hops 2-N: Follow associations
        for hop in range(1, max_hops):
            if not chain:
                break

            # Get last memory's content
            last_event_id = chain[-1][0]

            # Need to retrieve memory content (would need to add this capability)
            # For now, use emotional cache as proxy
            emotional_data = self.emotional_cache.get(last_event_id, {})

            # Extract concepts from emotion words as simple approach
            # In full implementation, would get actual memory content
            emotion_words = emotional_data.get('emotion_words', [])
            if not emotion_words:
                break

            # Search for related memories
            concept_query = " ".join(emotion_words[:3])  # Use top 3 emotion words
            related_results = self.search_memories(concept_query, k=3)

            # Find first unvisited memory
            added = False
            for event_id, score in related_results:
                if event_id not in visited:
                    decayed_score = score * (decay_per_hop ** hop)
                    chain.append((event_id, decayed_score, hop))
                    visited.add(event_id)
                    added = True
                    break

            # If no new memories found, stop chaining
            if not added:
                break

        return chain

    def search_with_sass_detection(self, query: str,
                                   current_response: str,
                                   k: int = 10) -> List[Tuple[str, float]]:
        """
        OPTIMIZATION 6: Sarcasm/Sass-Aware Search

        When Alice is sassy, find memories where she was also sassy.
        Matches tone/personality level in retrieval.

        Args:
            query: Search query
            current_response: Alice's current response (to detect sass level)
            k: Number of results

        Returns:
            List of (event_id, score) tuples ranked by sass-level matching
        """
        # Detect sass level in current response
        sass_indicators = ["ugh", "obviously", "literally", "whatever", "baka",
                          "seriously", "tbh", "honestly", "fine", "guess"]

        current_sass = sum(1 for word in sass_indicators
                          if word in current_response.lower())

        # Get 2x results for re-ranking
        base_results = self.search_memories(query, k=k*2)

        if not base_results:
            return []

        # Re-rank by sass level matching
        sassy_results = []

        for event_id, base_score in base_results:
            # Calculate memory sass level from cached content
            # (In full implementation, would check actual memory content)
            emotional_data = self.emotional_cache.get(event_id, {})
            emotion_words = emotional_data.get('emotion_words', [])

            memory_sass = sum(1 for word in sass_indicators
                            if any(word in ew for ew in emotion_words))

            # Boost if sass levels match
            sass_diff = abs(current_sass - memory_sass)
            sass_boost = 1.0 + (0.1 * min(3 - sass_diff, 3))  # Up to 30% boost

            boosted_score = base_score * sass_boost
            sassy_results.append((event_id, boosted_score))

        # Sort and return top K
        sassy_results.sort(key=lambda x: x[1], reverse=True)
        return sassy_results[:k]


class EmotionalMemoryAccessLayer:
    """
    Always-hot emotional memory access layer with background FAISS ↔ SQLite sync
    Keeps emotionally significant memories cached in memory for instant access
    """
    
    def __init__(self, index_system, max_cache_size: int = 1000):
        self.index_system = index_system
        self.max_cache_size = max_cache_size
        
        # Always-hot emotional memory cache
        self.emotional_cache = {}  # event_id -> IndexMemory
        self.cache_priority = []   # Ordered by emotional significance
        self.cache_lock = threading.RLock()
        
        # Background sync management
        self.sync_thread = None
        self.sync_interval = 30.0  # seconds
        self.stop_sync = threading.Event()
        self.sync_stats = {
            "syncs_completed": 0,
            "last_sync_time": 0,
            "sync_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Start background sync
        self._start_background_sync()
    
    def _start_background_sync(self):
        """Start background thread for FAISS ↔ SQLite synchronization"""
        if self.sync_thread and self.sync_thread.is_alive():
            return
        
        self.sync_thread = threading.Thread(
            target=self._background_sync_loop,
            daemon=True,
            name="EmotionalMemorySync"
        )
        self.sync_thread.start()
        print("✅ Background emotional memory sync started")
    
    def _background_sync_loop(self):
        """Background loop for keeping emotional memories hot"""
        while not self.stop_sync.wait(self.sync_interval):
            try:
                self._sync_emotional_memories()
                self._cleanup_cache()
                self.sync_stats["syncs_completed"] += 1
                self.sync_stats["last_sync_time"] = time.time()
                
            except Exception as e:
                self.sync_stats["sync_errors"] += 1
                print(f"⚠️  Emotional memory sync error: {e}")
    
    def _sync_emotional_memories(self):
        """Sync high-emotion memories between SQLite and cache"""
        if not self.index_system:
            return
        
        # Get highly emotional memories from SQLite
        conn = sqlite3.connect(self.index_system.db_path)
        cursor = conn.cursor()
        
        # Query for emotionally significant memories
        cursor.execute('''
            SELECT * FROM index_memories 
            WHERE (ABS(alice_emotional_valence) > 0.7 
                   OR alice_emotional_weight > 0.8
                   OR influence_long > 0.7
                   OR depth IN ('deep', 'core'))
            ORDER BY influence_long DESC, alice_emotional_weight DESC
            LIMIT ?
        ''', (self.max_cache_size,))
        
        high_emotion_memories = []
        for row in cursor.fetchall():
            memory = self.index_system._row_to_index_memory(row)
            high_emotion_memories.append(memory)

        if self.index_system.db_path != ":memory:": conn.close()
        
        # Update cache with emotional memories
        with self.cache_lock:
            # Clear old cache
            self.emotional_cache.clear()
            self.cache_priority.clear()
            
            # Add high-emotion memories to cache
            for memory in high_emotion_memories:
                self.emotional_cache[memory.event_id] = memory
                
                # Calculate priority score for cache ordering
                priority = self._calculate_emotional_priority(memory)
                self.cache_priority.append((memory.event_id, priority))
            
            # Sort by priority (highest first)
            self.cache_priority.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top memories if over limit
            if len(self.cache_priority) > self.max_cache_size:
                excess_memories = self.cache_priority[self.max_cache_size:]
                self.cache_priority = self.cache_priority[:self.max_cache_size]
                
                # Remove excess from cache
                for event_id, _ in excess_memories:
                    self.emotional_cache.pop(event_id, None)
    
    def _calculate_emotional_priority(self, memory: IndexMemory) -> float:
        """Calculate priority score for emotional memory caching"""
        priority = 0.0
        
        # Alice's emotional weight (how much she cares)
        if memory.alice_emotional_weight:
            priority += memory.alice_emotional_weight * 0.3
        
        # Emotional intensity (absolute valence)
        if memory.alice_emotional_valence:
            priority += abs(memory.alice_emotional_valence) * 0.2
        
        # Long-term influence
        priority += memory.influence_long * 0.3
        
        # Memory depth significance
        depth_weights = {
            MemoryDepth.SURFACE: 0.0,
            MemoryDepth.MID: 0.05,
            MemoryDepth.DEEP: 0.1,
            MemoryDepth.CORE: 0.15
        }
        priority += depth_weights.get(memory.depth, 0.0)
        
        return priority
    
    def _cleanup_cache(self):
        """Clean up old or low-priority cached memories"""
        with self.cache_lock:
            if len(self.emotional_cache) <= self.max_cache_size:
                return
            
            # Keep only top priority memories
            keep_count = int(self.max_cache_size * 0.9)  # Keep 90% to prevent thrashing
            to_keep = self.cache_priority[:keep_count]
            to_remove = self.cache_priority[keep_count:]
            
            # Remove low-priority memories
            for event_id, _ in to_remove:
                self.emotional_cache.pop(event_id, None)
            
            self.cache_priority = to_keep
    
    def get_emotional_memory(self, event_id: str) -> Optional[IndexMemory]:
        """Get memory from emotional cache (always-hot access)"""
        with self.cache_lock:
            if event_id in self.emotional_cache:
                self.sync_stats["cache_hits"] += 1
                return self.emotional_cache[event_id]
            else:
                self.sync_stats["cache_misses"] += 1
                return None
    
    def search_emotional_memories(self, query: str, k: int = 5) -> List[IndexMemory]:
        """Search cached emotional memories for instant access"""
        results = []
        
        with self.cache_lock:
            # Simple keyword search in cached memories
            query_lower = query.lower()
            
            for memory in self.emotional_cache.values():
                if query_lower in memory.content.lower():
                    # Calculate simple relevance score
                    relevance = memory.influence_long + (memory.alice_emotional_weight or 0)
                    memory.similarity_score = relevance
                    results.append(memory)
            
            # Sort by relevance and return top k
            results.sort(key=lambda m: m.similarity_score or 0, reverse=True)
            return results[:k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emotional memory access layer statistics"""
        with self.cache_lock:
            cache_size = len(self.emotional_cache)
            hit_rate = 0.0
            
            total_requests = self.sync_stats["cache_hits"] + self.sync_stats["cache_misses"]
            if total_requests > 0:
                hit_rate = self.sync_stats["cache_hits"] / total_requests
        
        return {
            "cache_size": cache_size,
            "max_cache_size": self.max_cache_size,
            "cache_hit_rate": round(hit_rate, 3),
            "background_sync": {
                "active": self.sync_thread.is_alive() if self.sync_thread else False,
                "syncs_completed": self.sync_stats["syncs_completed"],
                "sync_errors": self.sync_stats["sync_errors"],
                "last_sync": self.sync_stats["last_sync_time"]
            }
        }
    
    def stop(self):
        """Stop background sync and cleanup"""
        self.stop_sync.set()
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)




__all__ = [
    'FAISSMemoryIndex', 
    'EmotionalMemoryAccessLayer',
    'FAISS_AVAILABLE',
    'EMBEDDINGS_AVAILABLE', 
    'VECTOR_AVAILABLE',
    'IndexMemory'
]
