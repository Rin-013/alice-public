# Copyright 2025 Rin - Alice AI System
"""
IRIS - Intelligent Retrieval and Indexing System
=================================================

THE single memory interface for Alice.

Usage:
    from alice.core.memory.iris import IRIS, get_iris

    iris = get_iris()
    iris.start_session("rin", "Rin")
    iris.add_conversation("hello", "hi there!")
    results = iris.search("what did we talk about?")
"""

from .search import IRIS, IRISConfig, get_iris, _reset_iris

# Also export types needed for search
from ..types import SearchQuery, SearchResult, SearchType

# Wrapper modules
from .semantic import LegacyIRIS, SemanticMatch, SearchContext, SEMANTIC_AVAILABLE
from .context import Oracle, get_oracle, CONTEXT_AVAILABLE
from .fast_path import FastPathRetrieval, FAST_PATH_AVAILABLE

__all__ = [
    # Main unified interface
    'IRIS',
    'IRISConfig',
    'get_iris',
    '_reset_iris',
    # Search types
    'SearchQuery',
    'SearchResult',
    'SearchType',
    # Legacy wrappers
    'LegacyIRIS',
    'SemanticMatch',
    'SearchContext',
    'Oracle',
    'get_oracle',
    'FastPathRetrieval',
    # Availability flags
    'SEMANTIC_AVAILABLE',
    'CONTEXT_AVAILABLE',
    'FAST_PATH_AVAILABLE',
]
