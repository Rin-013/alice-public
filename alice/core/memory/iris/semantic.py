# Copyright 2025 Rin - Alice AI System
"""
Semantic Search Module
======================

Semantic/vector-based search using emotional clusters and Alice-specific rules.

Note: Currently wraps legacy iris.py implementation.
"""

# Re-export from legacy. The legacy IRIS (smart_search) was archived to
# master_archive/memory/_legacy/ in the Dec 2025 unification; the modern
# search path (cosine usefulness, depth layer) is authoritative and
# get_iris() is always called with legacy_iris=None. This import is
# expected to fail now — the stubs below keep the symbols importable.
# Not an error; only surfaced under ALICE_DEBUG. To restore, move the
# legacy module back.
import os as _os

try:
    from .._legacy.iris import IRIS as LegacyIRIS, SemanticMatch, SearchContext
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    SEMANTIC_AVAILABLE = False
    if _os.environ.get("ALICE_DEBUG") == "1":
        print(f"   Legacy IRIS not wired (archived): {e}")

    class LegacyIRIS:
        """Stub"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Requires legacy iris.py")

    class SemanticMatch:
        """Stub"""
        pass

    class SearchContext:
        """Stub"""
        pass


__all__ = ['LegacyIRIS', 'SemanticMatch', 'SearchContext', 'SEMANTIC_AVAILABLE']
