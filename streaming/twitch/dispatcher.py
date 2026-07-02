"""
Alice's tool dispatcher.

This is the layer between Alice's LLM output and actual tool execution.
Flow:
    LLM produces tool call -> Dispatcher validates -> Handler executes ->
    Result formatted -> Returned to Alice's context for her next turn.

Design principles:
  - Validation is strict. Bad calls return an error to Alice, not crash.
  - Execution is async-friendly. Twitch API calls shouldn't block her loop.
  - Errors are explained back to Alice in natural language so she can react.
  - All calls are logged for debugging and post-stream review.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Awaitable, Callable, Optional
from enum import Enum

from .tool_schemas import ALICE_TOOLS, TOOL_NAMES, PERCEPTION_TOOLS, ACTION_TOOLS

log = logging.getLogger("alice.dispatcher")


# =============================================================================
# Result types
# =============================================================================

class ToolStatus(str, Enum):
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"  # bad args from Alice
    EXECUTION_ERROR = "execution_error"    # Twitch API or network failure
    NOT_FOUND = "not_found"                # tool name doesn't exist
    RATE_LIMITED = "rate_limited"          # Alice fired too fast (see below)
    DENIED = "denied"                      # tool exists but blocked right now


@dataclass
class ToolResult:
    """What gets returned to Alice's context after a tool call."""
    tool_name: str
    status: ToolStatus
    data: Any = None              # the actual result payload on success
    message: str = ""             # natural-language summary for Alice to read
    error: Optional[str] = None   # technical error for logs
    duration_ms: float = 0.0

    def to_alice_context(self) -> str:
        """
        Format this result the way it gets injected into Alice's next turn.
        Stays terse - she doesn't need a wall of text, just enough to react.
        """
        if self.status == ToolStatus.SUCCESS:
            return f"[tool: {self.tool_name}] {self.message}"
        else:
            return f"[tool: {self.tool_name} failed] {self.message}"

    def to_log(self) -> dict:
        return asdict(self)


@dataclass
class CallRecord:
    """Audit log entry for every tool call attempted."""
    timestamp: float
    tool_name: str
    args: dict
    result: ToolResult


# =============================================================================
# Rate limiting (per-tool, per-stream)
# =============================================================================

@dataclass
class RateLimit:
    """
    Per-tool rate limits to prevent Alice from going feral.

    Twitch's API has its own rate limits, but these are policy limits to
    keep Alice from spamming polls every 30 seconds, which would be obnoxious
    to viewers even if technically allowed.
    """
    max_per_minute: int = 60
    max_per_stream: Optional[int] = None
    cooldown_seconds: float = 0.0  # min time between consecutive calls


# Tool-specific limits. Tweak these based on actual stream behavior.
RATE_LIMITS = {
    # Perception is cheap, near-unlimited
    "read_recent_chat": RateLimit(max_per_minute=120),
    "get_sub_count": RateLimit(max_per_minute=12),
    "read_recent_superchats": RateLimit(max_per_minute=30),
    "read_recent_gift_subs": RateLimit(max_per_minute=30),

    # Mod actions - measured
    "timeout_user": RateLimit(max_per_minute=10, cooldown_seconds=2.0),
    "ban_user": RateLimit(max_per_minute=3, max_per_stream=20, cooldown_seconds=5.0),

    # Audience engagement - rare events, big impact
    "create_poll": RateLimit(max_per_minute=2, max_per_stream=15, cooldown_seconds=30.0),
    "create_prediction": RateLimit(max_per_minute=1, max_per_stream=10, cooldown_seconds=60.0),
    "pin_chat_message": RateLimit(max_per_minute=4, cooldown_seconds=10.0),
    "request_clip": RateLimit(max_per_minute=4, cooldown_seconds=15.0),
    "update_stream_info": RateLimit(max_per_minute=2, cooldown_seconds=30.0),
}


class RateTracker:
    """Tracks tool call timestamps for rate limiting."""

    def __init__(self):
        self._calls: dict[str, list[float]] = {}
        self._stream_totals: dict[str, int] = {}
        self._stream_started_at: float = time.time()

    def reset_stream(self):
        """Call when a new stream starts."""
        self._calls.clear()
        self._stream_totals.clear()
        self._stream_started_at = time.time()

    def check(self, tool_name: str) -> Optional[str]:
        """
        Returns None if call is allowed, or an error message if rate limited.
        """
        limit = RATE_LIMITS.get(tool_name)
        if limit is None:
            return None  # no limit configured

        now = time.time()
        history = self._calls.setdefault(tool_name, [])

        # Prune entries older than 60s
        history[:] = [t for t in history if now - t < 60.0]

        # Check per-minute
        if len(history) >= limit.max_per_minute:
            return (
                f"You're firing {tool_name} too fast - "
                f"max {limit.max_per_minute}/min. Wait a bit."
            )

        # Check cooldown
        if history and limit.cooldown_seconds > 0:
            elapsed = now - history[-1]
            if elapsed < limit.cooldown_seconds:
                wait = limit.cooldown_seconds - elapsed
                return f"Cooldown on {tool_name} - {wait:.1f}s left."

        # Check stream total
        if limit.max_per_stream is not None:
            total = self._stream_totals.get(tool_name, 0)
            if total >= limit.max_per_stream:
                return (
                    f"You've used {tool_name} {total} times this stream - "
                    f"that's the cap ({limit.max_per_stream}). Save it for "
                    f"next time."
                )

        return None

    def record(self, tool_name: str):
        """Log a successful call for rate tracking."""
        now = time.time()
        self._calls.setdefault(tool_name, []).append(now)
        self._stream_totals[tool_name] = self._stream_totals.get(tool_name, 0) + 1


# =============================================================================
# Dispatcher
# =============================================================================

# Type for handler functions. They take validated args, return raw result data.
# Handlers can be sync or async. Dispatcher handles both.
ToolHandler = Callable[[dict], Any]


class ToolDispatcher:
    """
    Central dispatcher for Alice's tool calls.

    Usage:
        dispatcher = ToolDispatcher()
        dispatcher.register("read_recent_chat", handle_read_chat)
        ...

        result = await dispatcher.dispatch("timeout_user", {
            "username": "annoying_chatter_42",
            "duration_seconds": 60,
            "reason": "spamming the same emote",
        })

        alice_context.append(result.to_alice_context())
    """

    def __init__(self, extra_schemas: Optional[list[dict]] = None):
        self._handlers: dict[str, ToolHandler] = {}
        all_schemas = list(ALICE_TOOLS) + (extra_schemas or [])
        self._schemas: dict[str, dict] = {t["name"]: t for t in all_schemas}
        self._rate_tracker = RateTracker()
        self._call_log: list[CallRecord] = []
        self._enabled: dict[str, bool] = {name: True for name in self._schemas}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(self, tool_name: str, handler: ToolHandler):
        """Wire a handler function to a tool name."""
        if tool_name not in self._schemas:
            raise ValueError(
                f"Cannot register handler for unknown tool: {tool_name}. "
                f"Known tools: {TOOL_NAMES}"
            )
        self._handlers[tool_name] = handler
        log.info(f"Registered handler for {tool_name}")

    def unregister(self, tool_name: str):
        self._handlers.pop(tool_name, None)

    # -------------------------------------------------------------------------
    # Runtime control (Rin can disable tools mid-stream)
    # -------------------------------------------------------------------------

    def disable(self, tool_name: str):
        """Block this tool from being called. Useful for emergencies."""
        self._enabled[tool_name] = False
        log.warning(f"Tool DISABLED: {tool_name}")

    def enable(self, tool_name: str):
        self._enabled[tool_name] = True
        log.info(f"Tool ENABLED: {tool_name}")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self, tool_name: str, args: dict) -> Optional[str]:
        """
        Returns None if valid, else a validation error message.
        Light validation - the handlers do deeper checks. This catches
        obvious schema violations before we waste an HTTP call.
        """
        schema = self._schemas.get(tool_name)
        if schema is None:
            return f"Unknown tool: {tool_name}"

        params = schema["parameters"]
        required = params.get("required", [])
        properties = params.get("properties", {})

        # Required fields present?
        missing = [r for r in required if r not in args]
        if missing:
            return f"Missing required parameter(s): {missing}"

        # Unknown fields?
        unknown = [k for k in args if k not in properties]
        if unknown:
            return f"Unknown parameter(s): {unknown}"

        # Type / constraint checks (light)
        for key, value in args.items():
            spec = properties[key]
            expected = spec.get("type")
            if expected == "integer" and not isinstance(value, int):
                return f"{key} must be an integer, got {type(value).__name__}"
            if expected == "string" and not isinstance(value, str):
                return f"{key} must be a string"
            if expected == "array" and not isinstance(value, list):
                return f"{key} must be an array"

            if "minimum" in spec and value < spec["minimum"]:
                return f"{key} must be >= {spec['minimum']}, got {value}"
            if "maximum" in spec and value > spec["maximum"]:
                return f"{key} must be <= {spec['maximum']}, got {value}"
            if "maxLength" in spec and len(value) > spec["maxLength"]:
                return f"{key} too long (max {spec['maxLength']} chars)"
            if "minItems" in spec and len(value) < spec["minItems"]:
                return f"{key} needs at least {spec['minItems']} items"
            if "maxItems" in spec and len(value) > spec["maxItems"]:
                return f"{key} can have at most {spec['maxItems']} items"

        return None

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------

    async def dispatch(self, tool_name: str, args: dict) -> ToolResult:
        """
        Main entry point. Validates, rate-limits, executes, logs.
        """
        start = time.time()

        # Tool exists?
        if tool_name not in self._schemas:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.NOT_FOUND,
                message=(
                    f"There's no tool called '{tool_name}'. Check the list."
                ),
            )
            self._record_call(tool_name, args, result, start)
            return result

        # Tool enabled?
        if not self._enabled.get(tool_name, True):
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.DENIED,
                message=f"{tool_name} is disabled right now.",
            )
            self._record_call(tool_name, args, result, start)
            return result

        # Handler registered?
        handler = self._handlers.get(tool_name)
        if handler is None:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.EXECUTION_ERROR,
                message=f"{tool_name} isn't wired up yet.",
                error="No handler registered",
            )
            self._record_call(tool_name, args, result, start)
            return result

        # Validate args
        validation_error = self._validate(tool_name, args)
        if validation_error:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.VALIDATION_ERROR,
                message=validation_error,
            )
            self._record_call(tool_name, args, result, start)
            return result

        # Rate limit check
        rate_error = self._rate_tracker.check(tool_name)
        if rate_error:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.RATE_LIMITED,
                message=rate_error,
            )
            self._record_call(tool_name, args, result, start)
            return result

        # Execute
        try:
            raw = handler(args)
            # Support async handlers
            if isinstance(raw, Awaitable):
                raw = await raw

            self._rate_tracker.record(tool_name)
            result = self._format_success(tool_name, args, raw, start)

        except Exception as e:
            log.exception(f"Handler for {tool_name} raised")
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.EXECUTION_ERROR,
                message=(
                    f"Something broke trying to run {tool_name}. "
                    f"Twitch API or network issue."
                ),
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        self._record_call(tool_name, args, result, start)
        return result

    # -------------------------------------------------------------------------
    # Result formatting (turn raw API responses into Alice-readable summaries)
    # -------------------------------------------------------------------------

    def _format_success(
        self, tool_name: str, args: dict, raw: Any, start: float
    ) -> ToolResult:
        """
        Convert the handler's raw return value into a clean message Alice
        can read in her next context turn.
        """
        msg = ""

        if tool_name == "read_recent_chat":
            n = len(raw) if isinstance(raw, list) else 0
            msg = f"Got {n} recent chat messages."
        elif tool_name == "get_sub_count":
            if isinstance(raw, dict):
                msg = (
                    f"Sub count: {raw.get('total', '?')} "
                    f"(+{raw.get('gained_this_stream', 0)} this stream). "
                    f"Latest: {raw.get('latest_subscriber', 'none')}."
                )
            else:
                msg = "Got sub count."
        elif tool_name == "read_recent_superchats":
            n = len(raw) if isinstance(raw, list) else 0
            msg = f"Got {n} recent cheers/superchats."
        elif tool_name == "read_recent_gift_subs":
            n = len(raw) if isinstance(raw, list) else 0
            msg = f"Got {n} recent gift sub events."
        elif tool_name == "timeout_user":
            msg = (
                f"Timed out {args['username']} for "
                f"{args['duration_seconds']}s. Reason logged."
            )
        elif tool_name == "ban_user":
            msg = f"Banned {args['username']}."
        elif tool_name == "create_poll":
            msg = f"Poll '{args['title']}' is live."
        elif tool_name == "create_prediction":
            msg = f"Prediction '{args['title']}' is open for bets."
        elif tool_name == "pin_chat_message":
            msg = "Message pinned."
        elif tool_name == "request_clip":
            msg = "Clip request sent to mods."
        elif tool_name == "update_stream_info":
            msg = f"Stream title updated to: {args['title']}"
        else:
            msg = "Done."

        return ToolResult(
            tool_name=tool_name,
            status=ToolStatus.SUCCESS,
            data=raw,
            message=msg,
            duration_ms=(time.time() - start) * 1000,
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _record_call(
        self, tool_name: str, args: dict, result: ToolResult, start: float
    ):
        record = CallRecord(
            timestamp=start,
            tool_name=tool_name,
            args=args,
            result=result,
        )
        self._call_log.append(record)
        log.info(
            f"[{result.status.value}] {tool_name}({args}) -> {result.message}"
        )

    def get_call_log(self) -> list[CallRecord]:
        return list(self._call_log)

    def get_stream_stats(self) -> dict:
        """Quick summary of tool usage this stream. Useful for post-stream review."""
        stats = {}
        for record in self._call_log:
            name = record.tool_name
            if name not in stats:
                stats[name] = {"total": 0, "success": 0, "failed": 0}
            stats[name]["total"] += 1
            if record.result.status == ToolStatus.SUCCESS:
                stats[name]["success"] += 1
            else:
                stats[name]["failed"] += 1
        return stats


# =============================================================================
# Quick demo / smoke test
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Mock handlers for testing
    def mock_read_chat(args):
        n = args.get("count", 20)
        return [
            {"user": f"chatter_{i}", "message": f"hello {i}", "id": f"msg_{i}"}
            for i in range(n)
        ]

    def mock_timeout(args):
        return {"success": True}

    async def main():
        d = ToolDispatcher()
        d.register("read_recent_chat", mock_read_chat)
        d.register("timeout_user", mock_timeout)

        # Valid call
        r1 = await d.dispatch("read_recent_chat", {"count": 5})
        print(f"1. {r1.to_alice_context()}")

        # Validation failure
        r2 = await d.dispatch("timeout_user", {"username": "bob"})
        print(f"2. {r2.to_alice_context()}")

        # Valid timeout
        r3 = await d.dispatch("timeout_user", {
            "username": "bob",
            "duration_seconds": 60,
            "reason": "vibing wrong",
        })
        print(f"3. {r3.to_alice_context()}")

        # Unknown tool
        r4 = await d.dispatch("delete_twitch", {})
        print(f"4. {r4.to_alice_context()}")

        # Unregistered handler
        r5 = await d.dispatch("create_poll", {
            "title": "test",
            "choices": ["a", "b"],
        })
        print(f"5. {r5.to_alice_context()}")

        print(f"\nStream stats: {d.get_stream_stats()}")

    asyncio.run(main())
