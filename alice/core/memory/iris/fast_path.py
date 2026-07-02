# Copyright 2025 Rin - Alice AI System
"""
Fast Path Memory Retrieval
===========================

Quick FAISS-based retrieval for <100ms response times.

Note: Currently wraps legacy fast_path.py implementation.
"""

# Re-export from legacy. The legacy fast_path was archived to
# master_archive/memory/ in the Dec 2025 unification; the modern IRIS
# search path doesn't use it (get_iris() is always called with
# fast_path=None). This import is expected to fail now — the stub below
# keeps `FastPathRetrieval` importable. Not an error; only surfaced
# under ALICE_DEBUG. To restore, move the legacy module back.
import os as _os

try:
    from ..fast_path import FastPathRetrieval
    FAST_PATH_AVAILABLE = True
except ImportError as e:
    FAST_PATH_AVAILABLE = False
    if _os.environ.get("ALICE_DEBUG") == "1":
        print(f"   FastPathRetrieval not wired (legacy archived): {e}")

    class FastPathRetrieval:
        """Stub"""
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Requires legacy fast_path.py")


__all__ = ['FastPathRetrieval', 'FAST_PATH_AVAILABLE']
