"""
Alice Utils - Shared Utilities Package
======================================

Shared utility classes and functions used across Alice systems:
- embedding_utils: Shared SentenceTransformer model (saves 174MB RAM)
- gliner_utils: GLiNER-based fact extraction
- llm_utils: Lightweight LLM for quick inference tasks
- spacy_utils: spaCy NER for fact extraction (83% RAM savings vs GLiNER)

Usage:
    from alice.core.utils import get_shared_embedding_model
    from alice.core.utils import get_spacy_utility
"""

# Embedding utilities (shared model - saves 174MB RAM)
from .embedding_utils import (
    SharedEmbeddingModel,
    get_shared_embedding_model,
    unload_shared_embedding_model,
)

# GLiNER utilities (fact extraction)
from .gliner_utils import (
    GLiNERUtility,
    get_gliner_utility,
    unload_gliner,
)

# LLM utilities (quick inference)
from .llm_utils import (
    LLMUtility,
    get_llm_utility,
)

# spaCy utilities (lightweight NER)
from .spacy_utils import (
    SpaCyUtility,
    get_spacy_utility,
    unload_spacy,
)

__all__ = [
    # Embedding
    'SharedEmbeddingModel',
    'get_shared_embedding_model',
    'unload_shared_embedding_model',
    # GLiNER
    'GLiNERUtility',
    'get_gliner_utility',
    'unload_gliner',
    # LLM
    'LLMUtility',
    'get_llm_utility',
    # spaCy
    'SpaCyUtility',
    'get_spacy_utility',
    'unload_spacy',
]
