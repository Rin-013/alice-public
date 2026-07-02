# Copyright 2025 Rin - Alice AI System
"""
Alice Memory System
===================

IRIS is the single memory interface. All memory operations go through IRIS.

Usage:
    from alice.core.memory import IRIS, get_iris
    from alice.core.memory import Memory, MemoryType, SearchType

    iris = get_iris()
    iris.start_session("rin", "Rin")
    results = iris.search("query")
"""

# Core types
from .types import (
    Memory,
    MemoryType,
    MemoryDepth,
    DecayState,
    SearchQuery,
    SearchResult,
    SearchType,
    AkashicRecord,
    ChoiceRecord,
    UserProfile,
    MemoryEntry,
    IndexMemory,
)

# IRIS - THE memory interface
from .iris import IRIS, IRISConfig, get_iris

# Supporting modules (used internally by IRIS or externally)
from .importance import score_importance
from .recall_gate import MemoryRecallGate

# Reintegrated subsystems
try:
    from .smart_facts import SmartFactExtractor, FactType, StructuredFact
    SMART_FACTS_AVAILABLE = True
except ImportError as e:
    SMART_FACTS_AVAILABLE = False
    print(f"   Failed in {__file__}: {e}")

try:
    from .divergence import DivergenceDetector
    DIVERGENCE_AVAILABLE = True
except ImportError as e:
    DIVERGENCE_AVAILABLE = False
    print(f"   Failed in {__file__}: {e}")

try:
    from .trauma import TraumaQuarantine
    TRAUMA_AVAILABLE = True
except ImportError as e:
    TRAUMA_AVAILABLE = False
    print(f"   Failed in {__file__}: {e}")

# Backward compatibility aliases
AliceMemorySystem = IRIS
UnifiedMemorySystem = IRIS
get_alice_memory = get_iris

from .memory import reset_memory_system

__all__ = [
    # Types
    'Memory', 'MemoryType', 'MemoryDepth', 'DecayState',
    'SearchQuery', 'SearchResult', 'SearchType',
    'AkashicRecord', 'ChoiceRecord', 'UserProfile',
    'MemoryEntry', 'IndexMemory',
    # IRIS
    'IRIS', 'IRISConfig', 'get_iris',
    # Supporting
    'score_importance', 'MemoryRecallGate',
    # Reintegrated subsystems
    'SmartFactExtractor', 'FactType', 'StructuredFact',
    'DivergenceDetector', 'TraumaQuarantine',
    # Backward compat
    'AliceMemorySystem', 'UnifiedMemorySystem', 'get_alice_memory',
    'reset_memory_system',
]
