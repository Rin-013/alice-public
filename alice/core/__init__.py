"""
Alice Core Systems
==================

Core personality, inference, and memory systems for Alice.

This __init__.py provides backward compatibility by re-exporting all systems
at the alice.core level, so old imports still work.
"""

import os

# TTS worker subprocess — skip all heavy imports, only needs tts library directly
if os.environ.get('_ALICE_TTS_WORKER') == '1':
    # Worker process: nothing to export
    __all__ = []
else:
    # Full Alice process: load everything

    # Quiet mode helper
    def alice_print(*args, **kwargs):
        """Print only if not in quiet mode"""
        if os.environ.get('ALICE_QUIET_MODE') != '1':
            print(*args, **kwargs)

    import builtins
    builtins.alice_quiet = (os.environ.get('ALICE_QUIET_MODE') == '1')

    # Config
    from .config import AliceConfig, default_config

    # Fairy — content filter (TOS-compliance) + input injection guard
    from .fairy import (
        FairyProtection,
        PromptInjectionGuard,
    )

    # Memory Systems
    from .memory import (
        AliceMemorySystem, IndexMemory, AkashicRecord, MemoryDepth,
        IRIS,
    )

    # Mind — v4 cognitive architecture (Mind, Inner Voices, Proposals Buffer, Scheduler)
    from .mind import Mind, ProposalsBuffer, parse_mind_output

    # System Package
    from .system import (
        SystemRegistry,
        SystemState,
        SystemInfo,
        get_registry,
        initialize_all_systems,
        SystemCoordinator,
        SystemType,
    )

    # Utils Package
    from .utils import (
        SharedEmbeddingModel,
        get_shared_embedding_model,
        SpaCyUtility,
        get_spacy_utility,
    )

    # Legacy aliases
    alice_memory = None
    alice_personality = None

    __all__ = [
        'AliceConfig', 'default_config',
        'FairyProtection', 'PromptInjectionGuard',
        'AliceMemorySystem', 'IndexMemory', 'AkashicRecord', 'MemoryDepth',
        'IRIS',
        'Mind', 'ProposalsBuffer', 'parse_mind_output',
        'SystemRegistry', 'SystemState', 'SystemInfo',
        'get_registry', 'initialize_all_systems',
        'SystemCoordinator', 'SystemType',
        'SharedEmbeddingModel', 'get_shared_embedding_model',
        'SpaCyUtility', 'get_spacy_utility',
        'alice_memory', 'alice_personality',
    ]
