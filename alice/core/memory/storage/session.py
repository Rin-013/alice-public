# Copyright 2025 Rin - Alice AI System
"""
Session Memory Store
====================

In-memory FAISS index for the current session only.
Cleared on start_session(), flushed to total memory on end_session().

This is Tier 2 in the memory hierarchy:
  Tier 1 - In-context    (conversation_history[] in the prompt)
  Tier 2 - Session       (this class — searchable, current session only)
  Tier 3 - Total         (LongTermMemory — persistent across all sessions)

Why a separate session store?
- Small index = fast search (<1ms vs ~10ms for large total index)
- Inherently high signal — everything here is from the current conversation
- Clean separation: nothing is written to disk until end_session() flush
"""

import time
import uuid
import threading
from typing import List, Tuple, Optional, Dict, Any

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    np = None
    FAISS_AVAILABLE = False

try:
    from ....utils.embedding_utils import get_shared_embedding_model
except ImportError:
    def get_shared_embedding_model():
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return None


class SessionMemoryStore:
    """
    Lightweight in-memory FAISS index for the current session.

    Stores (content, memory_id) pairs. Pure in-memory — nothing is written
    to disk. Cleared on start_session(), flushed to total on end_session().
    """

    DIM = 384  # all-MiniLM-L6-v2 output dimension

    def __init__(self):
        self._lock = threading.Lock()
        self._encoder = None
        self._index = None
        self._id_to_content: Dict[int, Tuple[str, str]] = {}  # faiss_id → (memory_id, content)
        self._next_id = 0
        self._entries: List[Dict[str, Any]] = []  # ordered list of raw entries

        self._init_index()

    def _init_index(self):
        """Initialize (or re-initialize) the FAISS index."""
        if not FAISS_AVAILABLE:
            return
        # IndexFlatIP = inner product on normalized vectors = cosine similarity
        self._index = faiss.IndexFlatIP(self.DIM)
        self._id_to_content = {}
        self._next_id = 0
        self._entries = []

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = get_shared_embedding_model()
        return self._encoder

    def _encode(self, text: str):
        enc = self._get_encoder()
        if enc is None or np is None:
            return None
        try:
            vec = enc.encode([text], normalize_embeddings=True, show_progress_bar=False)
            return vec.astype(np.float32)
        except Exception:
            return None

    # ── Write ──────────────────────────────────────────────────────────────────

    def add(self, content: str, memory_id: Optional[str] = None,
            metadata: Optional[Dict] = None) -> str:
        """
        Add a piece of content to the session store.

        Args:
            content:   Text to store and index
            memory_id: Optional stable ID; generated if not given
            metadata:  Optional extra data to store alongside content

        Returns:
            The memory_id used
        """
        memory_id = memory_id or str(uuid.uuid4())

        entry = {
            "memory_id": memory_id,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        if FAISS_AVAILABLE and self._index is not None:
            vec = self._encode(content)
            if vec is not None:
                with self._lock:
                    faiss_id = self._next_id
                    self._index.add(vec)
                    self._id_to_content[faiss_id] = (memory_id, content)
                    self._next_id += 1

        self._entries.append(entry)
        return memory_id

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over session memories.

        Returns list of dicts: {memory_id, content, score, timestamp, metadata}
        Sorted by similarity descending.
        """
        if not FAISS_AVAILABLE or self._index is None or self._index.ntotal == 0:
            return []

        vec = self._encode(query)
        if vec is None:
            return []

        k = min(k, self._index.ntotal)
        with self._lock:
            distances, indices = self._index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            memory_id, content = self._id_to_content.get(int(idx), (None, None))
            if memory_id is None:
                continue
            # Find original entry for timestamp/metadata
            entry = next((e for e in self._entries if e["memory_id"] == memory_id), {})
            results.append({
                "memory_id": memory_id,
                "content": content,
                "score": float(dist),
                "timestamp": entry.get("timestamp", 0),
                "metadata": entry.get("metadata", {}),
                "source": "session",
            })

        return results

    # ── Bulk access ────────────────────────────────────────────────────────────

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all session entries in insertion order (for flush to total)."""
        return list(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def clear(self):
        """Reset the session store (call on start_session)."""
        with self._lock:
            self._init_index()

    def __repr__(self):
        return f"<SessionMemoryStore entries={self.size}>"


__all__ = ["SessionMemoryStore"]
