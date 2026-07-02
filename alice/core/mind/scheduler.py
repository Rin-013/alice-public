"""
Mind Scheduler — background thread that fires scheduled prompts into Mind.

Design (ported pattern from Claude Code's cronScheduler, simplified):
  - 1s poll loop, jittered fires
  - Supports three trigger types: interval / daily / once
  - Missed-fire recovery: on start, checks last_fired_at vs now and
    fires any interval/daily task whose window was crossed while Alice
    was off (once-shots also fire if their `at` has passed)
  - Permanent flag exempts built-in tasks from 30d auto-expire
  - onFire callback injects payload as a proposal (or a Mind
    conversation message) — decoupled from storage

Env:
  ALICE_SCHEDULER=0      disable the scheduler entirely (default on)
  ALICE_SCHEDULER_POLL_S tick interval in seconds (default 1.0)

Fail-open everywhere — a bad task never kills the thread.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

from .cron_tasks import (
    CronTask,
    load_tasks,
    save_tasks,
    expire_stale,
    AUTO_EXPIRE_DAYS,
)

logger = logging.getLogger(__name__)

OnFireCallback = Callable[[CronTask], None]


def _now() -> datetime:
    return datetime.now()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def should_fire(task: CronTask, now: Optional[datetime] = None) -> bool:
    """Return True if `task` should fire at `now` given its last_fired_at."""
    now = now or _now()
    trigger = task.trigger or {}
    kind = trigger.get("type")
    last = _parse_iso(task.last_fired_at)

    if kind == "interval":
        try:
            secs = int(trigger.get("seconds", 0) or 0)
        except (TypeError, ValueError):
            return False
        if secs <= 0:
            return False
        if last is None:
            return True
        return (now - last).total_seconds() >= secs

    if kind == "daily":
        s = str(trigger.get("time", "") or "")
        if ":" not in s:
            return False
        try:
            hh, mm = s.split(":", 1)
            target_h = int(hh); target_m = int(mm)
        except ValueError:
            return False
        # Fire when "now" has passed today's target and we haven't already
        # fired today (or ever).
        target_today = now.replace(hour=target_h, minute=target_m,
                                   second=0, microsecond=0)
        if now < target_today:
            return False
        if last is None:
            return True
        return last < target_today

    if kind == "once":
        at = _parse_iso(trigger.get("at"))
        if at is None:
            return False
        if last is not None:
            return False  # already fired
        return now >= at

    return False


class Scheduler:
    """Background scheduler thread that fires scheduled prompts."""

    def __init__(self,
                 on_fire: OnFireCallback,
                 path: Optional[Path] = None,
                 poll_seconds: Optional[float] = None):
        self._on_fire = on_fire
        self._path = path
        self._poll = poll_seconds or float(
            os.environ.get("ALICE_SCHEDULER_POLL_S", "1.0") or 1.0)

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._wake = threading.Event()

        self.stats = {
            "fires": 0,
            "missed_fires_recovered": 0,
            "once_expired": 0,
            "failures": 0,
        }

    def enabled(self) -> bool:
        return os.environ.get("ALICE_SCHEDULER", "1").strip() in ("1", "true", "True")

    def start(self) -> None:
        if self._running or not self.enabled():
            return
        self._running = True
        self._recover_missed_fires()
        self._thread = threading.Thread(target=self._loop,
                                        daemon=True, name="MindScheduler")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ----- internals -----

    def _recover_missed_fires(self) -> None:
        """On startup, scan for tasks whose fire window was crossed while
        Alice was off. Fire them once, catching up rather than floods."""
        try:
            tasks = load_tasks(self._path)
        except Exception as e:
            logger.warning(f"Scheduler: load_tasks failed on startup: {e}")
            return
        now = _now()
        changed = False
        for t in tasks:
            try:
                if should_fire(t, now):
                    self._fire_once(t, now)
                    self.stats["missed_fires_recovered"] += 1
                    changed = True
            except Exception as e:
                self.stats["failures"] += 1
                logger.warning(f"Scheduler: missed-fire recovery for {t.id}: {e}")
        if changed:
            self._persist(tasks)

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
            except Exception as e:
                self.stats["failures"] += 1
                logger.warning(f"Scheduler tick failed: {e}")
            # Sleep but allow early wake on stop()
            self._wake.wait(timeout=self._poll)
            self._wake.clear()

    def _tick(self) -> None:
        tasks = load_tasks(self._path)
        now = _now()
        changed = False
        for t in tasks:
            try:
                if should_fire(t, now):
                    self._fire_once(t, now)
                    self.stats["fires"] += 1
                    changed = True
            except Exception as e:
                self.stats["failures"] += 1
                logger.warning(f"Scheduler: fire {t.id} failed: {e}")

        # Drop once-tasks that already fired
        before = len(tasks)
        tasks = [t for t in tasks if not (
            t.trigger.get("type") == "once" and t.last_fired_at)]
        self.stats["once_expired"] += before - len(tasks)
        if before != len(tasks):
            changed = True

        # Expire stale non-permanent tasks
        before = len(tasks)
        tasks = expire_stale(tasks, now)
        if before != len(tasks):
            changed = True

        if changed:
            self._persist(tasks)

    def _fire_once(self, task: CronTask, now: datetime) -> None:
        # Apply jitter so back-to-back tasks don't stampede
        jitter = max(0, int(task.jitter_seconds or 0))
        if jitter:
            time.sleep(random.uniform(0, jitter))
        try:
            self._on_fire(task)
        except Exception as e:
            logger.warning(f"Scheduler on_fire callback for {task.id}: {e}")
        task.last_fired_at = now.isoformat(timespec="seconds")

    def _persist(self, tasks: List[CronTask]) -> None:
        try:
            save_tasks(tasks, self._path)
        except Exception as e:
            logger.warning(f"Scheduler: save_tasks failed: {e}")
