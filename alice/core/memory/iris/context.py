# Copyright 2025 Rin - Alice AI System
"""
Context Prefetch Module
=======================

Entity detection and context prefetch for conversation.
Absorbed from Oracle.

Note: Currently wraps legacy oracle.py implementation.
"""

# Re-export from legacy
try:
    from ..oracle import Oracle
    CONTEXT_AVAILABLE = True

    def get_oracle(memory_system=None):
        """Factory function for Oracle."""
        return Oracle(memory_system) if memory_system else Oracle()

except ImportError as e:
    CONTEXT_AVAILABLE = False
    print(f"⚠️ Oracle not available: {e}")

    class Oracle:
        """Stub"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Requires legacy oracle.py")

    def get_oracle(*args, **kwargs):
        return None


__all__ = ['Oracle', 'get_oracle', 'CONTEXT_AVAILABLE']
