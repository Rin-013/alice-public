"""
Alice System Registry - Single Source of Truth for System Instances
====================================================================

This registry tracks ALL Alice system instances and prevents duplicate initialization.

Core Principles:
1. ONE registry instance (singleton pattern)
2. Systems registered exactly ONCE
3. Dependency tracking for initialization order
4. Health monitoring and diagnostics

Usage:
    from alice.core.system_registry import get_registry

    registry = get_registry()
    registry.register('wheatley', wheatley_instance, dependencies=[])
    wheatley = registry.get('wheatley')
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time


class SystemState(Enum):
    """System lifecycle states"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class SystemInfo:
    """
    Information about a registered system.

    Attributes:
        name: Unique system identifier
        instance: The actual system object
        state: Current system state
        dependencies: List of system names this depends on
        load_time_ms: Time taken to initialize (milliseconds)
        error: Error message if state == FAILED
    """
    name: str
    instance: Any
    state: SystemState
    dependencies: List[str] = field(default_factory=list)
    load_time_ms: float = 0.0
    error: Optional[str] = None


class SystemRegistry:
    """
    Singleton registry for all Alice systems.

    Prevents duplicate initialization and provides dependency tracking.

    This is THE central point for system management. All systems must be
    registered here before use.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._systems: Dict[str, SystemInfo] = {}
        self._initialized = True
        print("📋 System Registry initialized")

    def register(self, name: str, instance: Any, dependencies: Optional[List[str]] = None):
        """
        Register a system instance.

        Args:
            name: Unique identifier for this system
            instance: The system object to register
            dependencies: List of system names this depends on

        Raises:
            ValueError: If system already registered (prevents duplicates)

        Example:
            registry.register('wheatley', wheatley_core, dependencies=[])
            registry.register('cne', cne_system, dependencies=['wheatley', 'memory'])
        """
        if name in self._systems:
            existing = self._systems[name]
            raise ValueError(
                f"❌ System '{name}' already registered!\n"
                f"   Existing instance: {type(existing.instance).__name__}\n"
                f"   New instance: {type(instance).__name__}\n"
                f"   This means the system was initialized twice.\n"
                f"   Fix: Remove duplicate initialization in system_initializer.py"
            )

        # Validate dependencies exist (optional - can defer to runtime)
        deps = dependencies or []
        missing_deps = [d for d in deps if d not in self._systems]
        if missing_deps:
            print(f"⚠️  Warning: '{name}' depends on {missing_deps} which aren't registered yet")
            print(f"   This may be OK if they load later. Otherwise it will fail at runtime.")

        self._systems[name] = SystemInfo(
            name=name,
            instance=instance,
            state=SystemState.READY,
            dependencies=deps
        )

        print(f"✅ Registered: {name}")

    def get(self, name: str) -> Any:
        """
        Get a system instance.

        Args:
            name: System identifier

        Returns:
            The system instance

        Raises:
            KeyError: If system not registered
            RuntimeError: If system not in READY state

        Example:
            wheatley = registry.get('wheatley')
        """
        if name not in self._systems:
            available = list(self._systems.keys())
            raise KeyError(
                f"❌ System '{name}' not found in registry!\n"
                f"   Available systems: {available}\n"
                f"   Fix: Make sure system is registered in system_initializer.py"
            )

        info = self._systems[name]
        if info.state != SystemState.READY:
            raise RuntimeError(
                f"❌ System '{name}' is not ready (state: {info.state.value})\n"
                f"   Error: {info.error}\n"
                f"   Fix: Check initialization logs for errors"
            )

        return info.instance

    def has(self, name: str) -> bool:
        """
        Check if system is registered and ready.

        Args:
            name: System identifier

        Returns:
            True if system exists and is ready, False otherwise

        Example:
            if registry.has('cne'):
                cne = registry.get('cne')
        """
        return name in self._systems and self._systems[name].state == SystemState.READY

    def get_stats(self) -> Dict[str, Any]:
        """
        Get loading statistics and dependency graph.

        Returns:
            Dictionary with:
                - total_systems: Number of registered systems
                - ready: Number in READY state
                - failed: Number in FAILED state
                - total_load_time_ms: Sum of all load times
                - systems: Per-system details

        Example:
            stats = registry.get_stats()
            print(f"Loaded {stats['ready']}/{stats['total_systems']} systems")
            print(f"Total load time: {stats['total_load_time_ms']:.2f}ms")
        """
        return {
            "total_systems": len(self._systems),
            "ready": sum(1 for s in self._systems.values() if s.state == SystemState.READY),
            "failed": sum(1 for s in self._systems.values() if s.state == SystemState.FAILED),
            "total_load_time_ms": sum(s.load_time_ms for s in self._systems.values()),
            "systems": {
                name: {
                    "state": info.state.value,
                    "dependencies": info.dependencies,
                    "load_time_ms": info.load_time_ms,
                    "type": type(info.instance).__name__,
                    "error": info.error
                }
                for name, info in self._systems.items()
            }
        }

    def validate_dependencies(self) -> List[str]:
        """
        Validate all system dependencies are satisfied.

        Returns:
            List of error messages (empty if all valid)

        Example:
            errors = registry.validate_dependencies()
            if errors:
                for error in errors:
                    print(error)
        """
        errors = []

        for name, info in self._systems.items():
            for dep in info.dependencies:
                if dep not in self._systems:
                    errors.append(
                        f"❌ System '{name}' depends on '{dep}' which is not registered"
                    )
                elif self._systems[dep].state != SystemState.READY:
                    errors.append(
                        f"❌ System '{name}' depends on '{dep}' which is not ready "
                        f"(state: {self._systems[dep].state.value})"
                    )

        return errors

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get complete dependency graph.

        Returns:
            Dictionary mapping system names to their dependencies

        Example:
            graph = registry.get_dependency_graph()
            print(f"CNE depends on: {graph['cne']}")
        """
        return {
            name: info.dependencies
            for name, info in self._systems.items()
        }

    def get_load_order(self) -> List[List[str]]:
        """
        Get systems grouped by dependency level.

        Returns:
            List of levels, each containing systems that can load in parallel

        Example:
            levels = registry.get_load_order()
            # [[Level 0 systems], [Level 1 systems], [Level 2 systems], ...]
        """
        # Build dependency graph
        graph = self.get_dependency_graph()

        # Find systems with no dependencies (Level 0)
        levels = []
        remaining = set(self._systems.keys())
        loaded = set()

        while remaining:
            # Find systems whose dependencies are all loaded
            level = []
            for system in remaining:
                deps = graph[system]
                if all(d in loaded for d in deps):
                    level.append(system)

            if not level:
                # Circular dependency detected
                print(f"⚠️  Warning: Possible circular dependencies in remaining systems: {remaining}")
                levels.append(list(remaining))
                break

            levels.append(level)
            loaded.update(level)
            remaining -= set(level)

        return levels

    def clear(self):
        """
        Clear all registered systems.

        WARNING: Only use for testing! This removes all systems.
        """
        self._systems.clear()
        print("🧹 Registry cleared")

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"<SystemRegistry: {stats['ready']}/{stats['total_systems']} ready, "
            f"{stats['failed']} failed>"
        )


# Global registry instance
_registry = SystemRegistry()


def get_registry() -> SystemRegistry:
    """
    Get the global system registry.

    Returns:
        The singleton SystemRegistry instance

    Example:
        from alice.core.system_registry import get_registry

        registry = get_registry()
        registry.register('my_system', instance, dependencies=[])
    """
    return _registry
