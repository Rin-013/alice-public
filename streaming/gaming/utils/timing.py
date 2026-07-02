"""
Timing Utilities
================

FrameTimer for maintaining consistent loop rates,
Cooldown for rate-limiting actions.
"""

import time
from typing import Optional


class FrameTimer:
    """
    Maintains a consistent tick rate for game loops.

    Usage:
        timer = FrameTimer(target_fps=60)
        while running:
            timer.tick()
            # ... do work ...
            timer.wait()  # sleep until next frame
    """

    def __init__(self, target_fps: float = 60.0):
        self.target_fps = target_fps
        self.target_dt = 1.0 / target_fps
        self._last_tick = time.perf_counter()
        self._frame_count = 0
        self._fps_window_start = time.perf_counter()
        self._fps_window_frames = 0
        self._current_fps = 0.0

    def tick(self):
        """Mark the start of a new frame."""
        now = time.perf_counter()
        self._last_tick = now
        self._frame_count += 1
        self._fps_window_frames += 1

        # Update FPS every second
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._current_fps = self._fps_window_frames / elapsed
            self._fps_window_start = now
            self._fps_window_frames = 0

    def wait(self):
        """Sleep until the next frame boundary."""
        elapsed = time.perf_counter() - self._last_tick
        remaining = self.target_dt - elapsed
        if remaining > 0.001:  # Don't bother sleeping for <1ms
            time.sleep(remaining)

    @property
    def fps(self) -> float:
        return self._current_fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def dt(self) -> float:
        """Actual time since last tick."""
        return time.perf_counter() - self._last_tick


class Cooldown:
    """
    Simple cooldown timer for rate-limiting actions.

    Usage:
        cd = Cooldown(seconds=2.0)
        if cd.ready():
            cd.trigger()
            do_thing()
    """

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last_trigger: Optional[float] = None

    def ready(self) -> bool:
        """Check if cooldown has elapsed."""
        if self._last_trigger is None:
            return True
        return (time.perf_counter() - self._last_trigger) >= self.seconds

    def trigger(self):
        """Reset the cooldown."""
        self._last_trigger = time.perf_counter()

    def remaining(self) -> float:
        """Seconds remaining on cooldown."""
        if self._last_trigger is None:
            return 0.0
        elapsed = time.perf_counter() - self._last_trigger
        return max(0.0, self.seconds - elapsed)

    def reset(self):
        """Clear the cooldown (makes it ready immediately)."""
        self._last_trigger = None
