#!/usr/bin/env python3
"""
Health Check System for Alice

Provides health status for monitoring and load balancing:
- Database connectivity
- Memory usage
- Disk space
- System resources

Usage:
    from alice.utils.healthcheck import get_health_status

    health = get_health_status()
    if health["status"] == "healthy":
        print("All systems operational")
"""

import os
import sys
import psutil
from datetime import datetime
from typing import Dict, Any
from pathlib import Path


def check_database(db_path: str = "alice/data/databases/alice_memory.db") -> Dict[str, Any]:
    """
    Check if database is accessible and healthy

    Returns:
        dict: Status, size, and connection info
    """
    try:
        db_file = Path(db_path)

        if not db_file.exists():
            return {
                "status": "unhealthy",
                "error": "Database file not found",
                "path": str(db_path)
            }

        # Check if writable
        if not os.access(db_path, os.W_OK):
            return {
                "status": "unhealthy",
                "error": "Database not writable",
                "path": str(db_path)
            }

        # Get size
        size_mb = db_file.stat().st_size / (1024 * 1024)

        # Try to connect
        import sqlite3
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]
        conn.close()

        return {
            "status": "healthy",
            "size_mb": round(size_mb, 2),
            "tables": table_count,
            "path": str(db_path)
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "path": str(db_path)
        }


def check_memory_usage(warning_threshold_pct: float = 80.0) -> Dict[str, Any]:
    """
    Check system memory usage

    Args:
        warning_threshold_pct: Percentage to trigger warning status

    Returns:
        dict: Memory status and usage info
    """
    try:
        mem = psutil.virtual_memory()
        process = psutil.Process()
        process_mem = process.memory_info()

        status = "healthy"
        if mem.percent >= warning_threshold_pct:
            status = "warning"
        if mem.percent >= 95.0:
            status = "unhealthy"

        return {
            "status": status,
            "system_memory_pct": round(mem.percent, 2),
            "system_memory_available_gb": round(mem.available / (1024**3), 2),
            "system_memory_total_gb": round(mem.total / (1024**3), 2),
            "process_memory_mb": round(process_mem.rss / (1024**2), 2),
            "warning_threshold_pct": warning_threshold_pct
        }

    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e)
        }


def check_disk_space(path: str = ".", warning_threshold_pct: float = 80.0) -> Dict[str, Any]:
    """
    Check disk space availability

    Args:
        path: Path to check disk space for
        warning_threshold_pct: Percentage to trigger warning status

    Returns:
        dict: Disk status and usage info
    """
    try:
        disk = psutil.disk_usage(path)

        status = "healthy"
        if disk.percent >= warning_threshold_pct:
            status = "warning"
        if disk.percent >= 95.0:
            status = "unhealthy"

        return {
            "status": status,
            "disk_usage_pct": round(disk.percent, 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "path": str(Path(path).absolute()),
            "warning_threshold_pct": warning_threshold_pct
        }

    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e)
        }


def check_system_resources() -> Dict[str, Any]:
    """
    Check overall system resources (CPU, threads, etc.)

    Returns:
        dict: System resource status
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = os.cpu_count() or 1
        process = psutil.Process()

        status = "healthy"
        if cpu_percent >= 80.0:
            status = "warning"
        if cpu_percent >= 95.0:
            status = "unhealthy"

        return {
            "status": status,
            "cpu_percent": round(cpu_percent, 2),
            "cpu_count": cpu_count,
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "uptime_seconds": round(datetime.now().timestamp() - process.create_time(), 2)
        }

    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e)
        }


def get_health_status(
    db_path: str = "alice/data/databases/alice_memory.db",
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Get comprehensive health status for Alice

    Args:
        db_path: Path to database file
        detailed: Include detailed system metrics

    Returns:
        dict: Complete health status

    Example:
        {
            "status": "healthy",
            "timestamp": "2025-11-06T12:00:00",
            "version": "1.0.0",
            "checks": {
                "database": {...},
                "memory": {...},
                "disk": {...},
                "system": {...}
            }
        }
    """
    # Run all health checks
    db_check = check_database(db_path)
    mem_check = check_memory_usage()
    disk_check = check_disk_space()
    system_check = check_system_resources()

    # Determine overall status
    all_checks = [db_check, mem_check, disk_check, system_check]
    statuses = [check.get("status", "unknown") for check in all_checks]

    if any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    elif any(s == "warning" for s in statuses):
        overall_status = "warning"
    elif any(s == "unknown" for s in statuses):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Build response
    response = {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",  # TODO: Load from package version
        "environment": os.getenv("ALICE_ENVIRONMENT", "development")
    }

    if detailed:
        response["checks"] = {
            "database": db_check,
            "memory": mem_check,
            "disk": disk_check,
            "system": system_check
        }

    return response


def print_health_status():
    """
    Print health status to console (for CLI usage)
    """
    health = get_health_status()

    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "unhealthy": "❌",
        "degraded": "⚠️",
        "unknown": "❓"
    }

    print(f"\n{status_emoji.get(health['status'], '❓')} Alice Health Status: {health['status'].upper()}")
    print(f"Timestamp: {health['timestamp']}")
    print(f"Version: {health['version']}")
    print(f"Environment: {health['environment']}")

    if "checks" in health:
        print("\nDetailed Checks:")
        for check_name, check_result in health["checks"].items():
            check_status = check_result.get("status", "unknown")
            emoji = status_emoji.get(check_status, "❓")
            print(f"  {emoji} {check_name.capitalize()}: {check_status}")

            # Show key metrics
            if check_name == "database" and check_status == "healthy":
                print(f"     Tables: {check_result.get('tables', 0)}")
                print(f"     Size: {check_result.get('size_mb', 0)} MB")
            elif check_name == "memory":
                print(f"     System: {check_result.get('system_memory_pct', 0)}%")
                print(f"     Process: {check_result.get('process_memory_mb', 0)} MB")
            elif check_name == "disk":
                print(f"     Usage: {check_result.get('disk_usage_pct', 0)}%")
                print(f"     Free: {check_result.get('disk_free_gb', 0)} GB")
            elif check_name == "system":
                print(f"     CPU: {check_result.get('cpu_percent', 0)}%")
                print(f"     Threads: {check_result.get('thread_count', 0)}")

            # Show errors
            if "error" in check_result:
                print(f"     Error: {check_result['error']}")

    print()


if __name__ == "__main__":
    # CLI usage: python -m alice.utils.healthcheck
    print_health_status()

    # Exit with appropriate code for CI/CD
    health = get_health_status(detailed=False)
    if health["status"] == "unhealthy":
        sys.exit(1)
    elif health["status"] in ["warning", "degraded"]:
        sys.exit(2)
    else:
        sys.exit(0)
