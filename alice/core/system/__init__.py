"""
Alice System - System Management Package
=========================================

Core system management including:
- SystemRegistry: Single source of truth for all system instances
- SystemInitializer: Dependency injection entry point
- SystemCoordinator: Master orchestrator for all psychological systems

Usage:
    from alice.core.system import get_registry, initialize_all_systems
    from alice.core.system import SystemCoordinator, SystemType
"""

# System Registry (single source of truth)
from .system_registry import (
    SystemRegistry,
    SystemState,
    SystemInfo,
    get_registry,
)

# System Initializer (dependency injection)
from .system_initializer import (
    initialize_all_systems,
)

# System Coordinator (orchestrates psychological systems)
from .system_coordinator import (
    SystemCoordinator,
    SystemType,
    SystemPriority,
    SystemConfig,
    SystemTrigger,
    TriggerResult,
    # Convenience functions
    trigger_system,
    process_queue,
    coordinate_message,
    generate_meta_awareness,
    get_state,
    get_stats,
    get_system_status,
    # Global instance
    alice_coordinator,
)

__all__ = [
    # Registry
    'SystemRegistry',
    'SystemState',
    'SystemInfo',
    'get_registry',
    # Initializer
    'initialize_all_systems',
    # Coordinator
    'SystemCoordinator',
    'SystemType',
    'SystemPriority',
    'SystemConfig',
    'SystemTrigger',
    'TriggerResult',
    'trigger_system',
    'process_queue',
    'coordinate_message',
    'generate_meta_awareness',
    'get_state',
    'get_stats',
    'get_system_status',
    'alice_coordinator',
]
