"""
Cron task storage — JSON-backed, thread-safe read/write for the scheduler.

Schema:
  {
    "id": "nightly_compact",
    "trigger": {
      "type": "interval" | "daily" | "once",
      "seconds": 3600,                 # interval only
      "time": "03:00",                 # daily only (HH:MM 24h local)
      "at": "2026-04-17T10:00:00"      # once only (ISO-8601 local)
    },
    "payload": "Check memory usage...", # prompt injected into Mind
    "permanent": false,                 # exempt from 30d auto-expire
    "jitter_seconds": 0,
    "last_fired_at": "2026-04-16T...",
    "created_at":   "2026-04-16T..."
  }

Fail-open: corrupt file → treated as empty list, not a crash.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_PATH = Path("alice/data/scheduled_tasks.json")
_FILE_LOCK = threading.Lock()

AUTO_EXPIRE_DAYS = 30  # user crons expire after this unless permanent=True


@dataclass
class CronTask:
    id: str
    trigger: Dict[str, Any]
    payload: str
    permanent: bool = False
    jitter_seconds: int = 0
    last_fired_at: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CronTask":
        return cls(
            id=str(d.get("id", "")),
            trigger=d.get("trigger", {}) or {},
            payload=str(d.get("payload", "")),
            permanent=bool(d.get("permanent", False)),
            jitter_seconds=int(d.get("jitter_seconds", 0) or 0),
            last_fired_at=d.get("last_fired_at"),
            created_at=d.get("created_at"),
        )

    def is_valid(self) -> bool:
        if not self.id or not self.payload:
            return False
        t = self.trigger or {}
        kind = t.get("type")
        if kind == "interval":
            try:
                secs = int(t.get("seconds", 0) or 0)
            except (TypeError, ValueError):
                return False
            return secs > 0
        if kind == "daily":
            s = str(t.get("time", "") or "")
            if len(s) < 4 or ":" not in s:
                return False
            try:
                hh, mm = s.split(":", 1)
                h = int(hh); m = int(mm)
                return 0 <= h < 24 and 0 <= m < 60
            except ValueError:
                return False
        if kind == "once":
            try:
                datetime.fromisoformat(str(t.get("at", "") or ""))
                return True
            except ValueError:
                return False
        return False


def load_tasks(path: Optional[Path] = None) -> List[CronTask]:
    p = path or _DEFAULT_PATH
    with _FILE_LOCK:
        if not p.exists():
            return []
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    if not isinstance(raw, list):
        return []
    tasks: List[CronTask] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        t = CronTask.from_dict(entry)
        if t.is_valid():
            tasks.append(t)
    return tasks


def save_tasks(tasks: List[CronTask], path: Optional[Path] = None) -> None:
    p = path or _DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [t.to_dict() for t in tasks]
    with _FILE_LOCK:
        # Atomic write: tmp file + rename
        fd, tmp = tempfile.mkstemp(prefix=".tasks.", dir=str(p.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def add_task(task: CronTask, path: Optional[Path] = None) -> bool:
    """Add or replace a task by id. Returns False if task is invalid."""
    if not task.is_valid():
        return False
    if not task.created_at:
        task.created_at = datetime.now().isoformat(timespec="seconds")
    tasks = load_tasks(path)
    tasks = [t for t in tasks if t.id != task.id]
    tasks.append(task)
    save_tasks(tasks, path)
    return True


def remove_task(task_id: str, path: Optional[Path] = None) -> bool:
    tasks = load_tasks(path)
    before = len(tasks)
    tasks = [t for t in tasks if t.id != task_id]
    if len(tasks) == before:
        return False
    save_tasks(tasks, path)
    return True


def expire_stale(tasks: List[CronTask], now: Optional[datetime] = None) -> List[CronTask]:
    """Drop non-permanent tasks older than AUTO_EXPIRE_DAYS. Returns the
    filtered list."""
    now = now or datetime.now()
    cutoff = now - timedelta(days=AUTO_EXPIRE_DAYS)
    out: List[CronTask] = []
    for t in tasks:
        if t.permanent:
            out.append(t)
            continue
        if not t.created_at:
            out.append(t)
            continue
        try:
            created = datetime.fromisoformat(t.created_at)
        except ValueError:
            out.append(t)
            continue
        if created >= cutoff:
            out.append(t)
    return out
