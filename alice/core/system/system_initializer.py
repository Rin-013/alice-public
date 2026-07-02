"""
Alice System Initializer - Dependency Injection Entry Point
============================================================

THIS IS THE ONLY FILE THAT IMPORTS AND CREATES SYSTEM INSTANCES.

All other files receive dependencies via parameters.

Architecture:
- ONE place creates instances (this file)
- Everything else receives dependencies (via __init__)
- ZERO module-level cross-folder imports elsewhere
- SystemRegistry tracks all instances

v4 cleanup: March 2026 — removed all archived/deleted system registrations.
Only live systems remain.
"""

import time
import os
from typing import Dict, Any, Optional
from .system_registry import get_registry


def initialize_all_systems(config: Optional[Dict[str, Any]] = None, memory_db_path: str = "alice/data/databases/alice_memory.db"):
    """
    Initialize all live Alice systems in dependency order.

    Returns:
        SystemRegistry with all loaded systems

    Usage:
        from alice.core.system.system_initializer import initialize_all_systems
        registry = initialize_all_systems()
        fairy = registry.get('fairy')
        iris = registry.get('iris')
    """
    registry = get_registry()
    registry.clear()

    print("🚀 Starting Alice system initialization...")
    start_time = time.time()

    config = config or {}

    # ============================================================
    # LEVEL 0: Protection
    # ============================================================

    try:
        from ..fairy.fairy import FairyProtection
        fairy = FairyProtection()
        registry.register('fairy', fairy, dependencies=[])
        print("✅ Fairy protection loaded")
    except ImportError as e:
        print(f"⚠️ Fairy not available: {e}")

    # ============================================================
    # LEVEL 2: Memory + Scripting
    # ============================================================

    # IRIS — the single memory interface
    try:
        from ..memory import get_iris
        iris = get_iris()
        registry.register('iris', iris, dependencies=[])
        registry.register('memory', iris, dependencies=[])  # backward compat
        print("✅ IRIS memory system loaded")
    except ImportError as e:
        print(f"⚠️ IRIS not available: {e}")

    # ============================================================
    # LEVEL 3: System Coordinator
    # ============================================================

    try:
        from .system_coordinator import alice_coordinator
        registry.register('system_coordinator', alice_coordinator, dependencies=[])
        print("✅ System Coordinator loaded")
    except ImportError as e:
        print(f"⚠️ System Coordinator not available: {e}")

    # ============================================================
    # Complete
    # ============================================================
    total_time = time.time() - start_time
    stats = registry.get_stats()

    print(f"\n✅ Alice initialization complete!")
    print(f"   Systems loaded: {stats['ready']}/{stats['total_systems']}")
    print(f"   Time: {total_time:.2f}s")

    if stats['failed'] > 0:
        print(f"\n⚠️  Failed systems:")
        for name, info in stats['systems'].items():
            if info['state'] == 'failed':
                print(f"   - {name}: {info.get('error', 'Unknown error')}")

    return registry


# Backward compatibility
def __getattr__(name):
    """Lazy attribute access for backward compat."""
    registry = get_registry()

    name_mapping = {
        'alice_memory': 'memory',
        'system_coordinator': 'system_coordinator',
        'fairy_protection_check': 'fairy',
    }

    if name in name_mapping:
        key = name_mapping[name]
        return registry.get(key) if registry.has(key) else None

    if name == 'SYSTEM_COORDINATOR_AVAILABLE':
        return registry.has('system_coordinator')

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
