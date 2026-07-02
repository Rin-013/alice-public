# Copyright 2025 Rin - Alice AI System
"""
Backward-compatibility shim.

AliceMemorySystem is now IRIS. This module re-exports for code that
imports AliceMemorySystem or get_alice_memory directly.
"""

from .iris import IRIS as AliceMemorySystem, get_iris as get_alice_memory
from .iris.search import _reset_iris


def reset_memory_system():
    """Reset the memory system singleton (for testing)."""
    _reset_iris()


__all__ = [
    'AliceMemorySystem',
    'get_alice_memory',
    'reset_memory_system',
]
