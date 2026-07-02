# Copyright 2025 Rin - Alice AI System
"""
Memory Storage Layer
====================

Handles persistence of memories across different storage backends.

Modules (all fully extracted from legacy - Dec 2025):
- short_term: Session/conversation memory
- long_term: SQLite persistence (1,269 lines)
- akashic: Factual truth records
- vector: FAISS embeddings + emotional layer
- index_system: Depth-based organization
"""

from .short_term import ShortTermMemory
from .long_term import LongTermMemory, LONG_TERM_AVAILABLE
from .akashic import AkashicRecords, AkashicRecord, AKASHIC_AVAILABLE
from .vector import FAISSMemoryIndex, EmotionalMemoryAccessLayer, VECTOR_AVAILABLE
from .index_system import IndexSystem, INDEX_SYSTEM_AVAILABLE

__all__ = [
    # Core storage
    'ShortTermMemory',
    'LongTermMemory',
    'AkashicRecords',
    'AkashicRecord',
    'FAISSMemoryIndex',
    'EmotionalMemoryAccessLayer',
    'IndexSystem',
    # Availability flags
    'LONG_TERM_AVAILABLE',
    'AKASHIC_AVAILABLE',
    'VECTOR_AVAILABLE',
    'INDEX_SYSTEM_AVAILABLE',
]
