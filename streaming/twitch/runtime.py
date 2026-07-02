"""
Twitch runtime — background thread that owns the IRC connection, dispatcher,
and inbound message queue.

Lives outside chat.py so the entry point stays thin. chat.py calls
`start_twitch(...)` once during init and then talks to the returned
`TwitchRuntime` handle:

    runtime = start_twitch(fairy=fairy, classifier=mind_classifier)
    # In the main loop:
    msg = runtime.poll_chat_input(timeout=0.05)         # cross-thread queue read
    runtime.send_chat("hello chat")                     # cross-thread send
    result = runtime.dispatch_tool("ban_user", {...})   # cross-thread Helix call

Threading:
  - Main thread: Alice's chat.py loop (synchronous).
  - Twitch thread: this module's asyncio loop. Owns the IRC socket and
    Helix HTTPX client. Never blocks the main thread.

Cross-thread:
  - In-bound chat → main: `queue.PriorityQueue` (thread-safe stdlib).
  - Main → Twitch async API: `asyncio.run_coroutine_threadsafe`.

Classifier:
  - Plug a callable: `(ChatMessage) -> Optional[int]` returning a priority
    score 0-100, or None to drop the message.
  - Default classifier (when None): drops everything except @mentions and
    sub/cheer events. This keeps the integration usable before the Mind
    classifier exists; Mind plugs in via task #8.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue as _queue_mod
import threading
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .auth import load_auth
from .client import ChatMessage, TwitchClient
from .dispatcher import ToolDispatcher, ToolResult
from .tools import register_all

logger = logging.getLogger("alice.twitch.runtime")


# Priority bands. Higher = sooner.
PRIO_DIRECT_MENTION = 80
PRIO_SUB_OR_CHEER = 90
PRIO_MIND_INTERESTING = 50
PRIO_DEFAULT_DROP = -1


@dataclass(order=True)
class _QueueItem:
    """PriorityQueue sorts by (-priority, monotonic ts) — newest of equal priority first."""
    sort_key: tuple = field(init=False)
    priority: int = 0
    timestamp: float = 0.0
    message: Optional[ChatMessage] = field(default=None, compare=False)

    def __post_init__(self):
        # Negate priority so higher numbers come out first from the min-heap.
        self.sort_key = (-self.priority, -self.timestamp)


# Type for the priority classifier. None → drop.
ChatClassifier = Callable[[ChatMessage], Optional[int]]


def _default_classifier(bot_username: str) -> ChatClassifier:
    """
    Heuristic-only classifier used when no smarter one is provided.

    Rules:
      - Cheers (bits > 0): always pass at sub/cheer priority.
      - @<bot_username> or "<bot_username>" mention: pass at mention priority.
      - Else: drop (returns None).

    Sub/gift events come through `event_raw_usernotice` in client.py rather
    than chat messages; this classifier only sees PRIVMSG-style ChatMessages.
    Sub-event handling lives in the runtime itself (see _on_sub_event).
    """
    target = bot_username.lower().lstrip("@")

    def classify(msg: ChatMessage) -> Optional[int]:
        if msg.bits and msg.bits > 0:
            return PRIO_SUB_OR_CHEER
        text = (msg.text or "").lower()
        if f"@{target}" in text:
            return PRIO_DIRECT_MENTION
        # Loose mention — bare name in the text.
        if target and target in text.split():
            return PRIO_DIRECT_MENTION
        return None

    return classify


def make_mind_classifier(mind, bot_username: str, threshold: int = 35) -> ChatClassifier:
    """
    Heuristic + Mind-LLM combined classifier.

    Cheers and direct mentions bypass Mind (cheap, always interesting).
    Everything else gets a Mind score 0-100; messages below `threshold`
    are dropped, the rest enter the queue at the Mind score.

    Args:
      mind: alice.core.mind.Mind instance with `classify_chat_message`.
      bot_username: the bot's twitch login (used by the heuristic prefilter).
      threshold: Mind scores below this are dropped (default 35 → keeps ~10-20%
                 of normal chat traffic depending on Mind's calibration).
    """
    heuristic = _default_classifier(bot_username)

    def classify(msg: ChatMessage) -> Optional[int]:
        # Fast path — cheers, mentions never need a Mind call.
        h = heuristic(msg)
        if h is not None:
            return h
        # Slow path — ask Mind. Failure (None) or low score → drop.
        try:
            score = mind.classify_chat_message(text=msg.text, username=msg.username)
        except Exception as e:
            logger.warning(f"mind classifier raised: {e}")
            return None
        if score is None or score < threshold:
            return None
        return int(score)

    return classify


# =============================================================================
# Runtime
# =============================================================================

class TwitchRuntime:
    """
    Owns the Twitch background thread + asyncio loop. Created via
    `start_twitch(...)`. All public methods are safe to call from the main
    thread; they marshal onto the asyncio loop as needed.
    """

    def __init__(
        self,
        client: TwitchClient,
        dispatcher: ToolDispatcher,
        classifier: ChatClassifier,
        max_queue: int = 200,
    ):
        self.client = client
        self.dispatcher = dispatcher
        self._classifier = classifier
        self._queue: _queue_mod.PriorityQueue[_QueueItem] = _queue_mod.PriorityQueue(maxsize=max_queue)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._stopped = threading.Event()

    # -- Public API -----------------------------------------------------------

    def poll_chat_input(self, timeout: float = 0.0) -> Optional[ChatMessage]:
        """
        Pop the highest-priority pending chat message, or None if none ready
        within `timeout` seconds. Call from the main thread.
        """
        try:
            item = self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
            return item.message
        except _queue_mod.Empty:
            return None

    def send_chat(self, text: str, truncate: int = 480) -> bool:
        """
        Send `text` to chat. Truncates to `truncate` chars (Twitch hard-cap is
        500). Returns True on success, False on failure (logged but not raised).
        Call from the main thread.
        """
        if self._loop is None or not self._loop.is_running():
            logger.warning("send_chat called before runtime is ready")
            return False
        if len(text) > truncate:
            text = text[: truncate - 1] + "…"
        try:
            fut = asyncio.run_coroutine_threadsafe(self.client.send_chat(text), self._loop)
            fut.result(timeout=8.0)
            return True
        except Exception as e:
            logger.warning(f"send_chat failed: {e}")
            return False

    def dispatch_tool(self, tool_name: str, args: dict, timeout: float = 12.0) -> ToolResult:
        """
        Run a tool via the dispatcher. Blocks the main thread until the tool
        completes or `timeout` elapses (which surfaces as an EXECUTION_ERROR).
        """
        if self._loop is None or not self._loop.is_running():
            from .dispatcher import ToolStatus
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.EXECUTION_ERROR,
                message="twitch runtime not connected",
                error="runtime not connected",
            )
        try:
            fut = asyncio.run_coroutine_threadsafe(self.dispatcher.dispatch(tool_name, args), self._loop)
            return fut.result(timeout=timeout)
        except Exception as e:
            from .dispatcher import ToolStatus
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.EXECUTION_ERROR,
                message=f"tool dispatch crashed: {e}",
                error=str(e),
            )

    @property
    def connected(self) -> bool:
        return self._ready.is_set() and self.client.connected

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background thread to disconnect and exit."""
        self._stop.set()
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)  # nudge the loop
        self._stopped.wait(timeout=timeout)
        if self._thread and self._thread.is_alive():
            logger.warning("twitch runtime thread did not exit cleanly")

    # -- Internal: thread + loop lifecycle ------------------------------------

    def _start_thread(self, ready_timeout: float = 15.0) -> None:
        """Spawn the background thread and wait for the connection to come up."""
        self._thread = threading.Thread(target=self._run, name="twitch-runtime", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=ready_timeout):
            raise RuntimeError(f"twitch runtime did not connect within {ready_timeout}s")

    def _run(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main_async())
        except Exception:
            logger.exception("twitch runtime crashed")
        finally:
            try:
                if self._loop is not None:
                    self._loop.close()
            except Exception:
                pass
            self._stopped.set()

    async def _main_async(self) -> None:
        # Wire on_message before connect — twitchio fires events as soon as it can.
        self.client.on_message = self._on_message
        await self.client.connect()
        self._ready.set()
        logger.info("twitch runtime connected and ready")

        # Idle until told to stop. asyncio's stop_event is cheaper than busy-wait.
        stop_future: asyncio.Future = asyncio.Future()

        def _check_stop():
            if self._stop.is_set() and not stop_future.done():
                stop_future.set_result(None)
            else:
                self._loop.call_later(0.5, _check_stop)

        self._loop.call_soon(_check_stop)
        await stop_future
        await self.client.disconnect()
        logger.info("twitch runtime disconnected")

    async def _on_message(self, msg: ChatMessage) -> None:
        """
        Inbound chat callback (runs on the twitch thread's loop).
        Classify the message; enqueue if the classifier says so.
        """
        try:
            prio = self._classifier(msg)
        except Exception as e:
            logger.warning(f"classifier raised: {e}")
            prio = None
        if prio is None or prio < 0:
            return
        try:
            self._queue.put_nowait(_QueueItem(
                priority=prio,
                timestamp=msg.timestamp,
                message=msg,
            ))
        except _queue_mod.Full:
            # Drop oldest low-prio entry to make room — better than wedging.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(_QueueItem(prio, msg.timestamp, msg))
            except Exception:
                pass


# =============================================================================
# start_twitch — entry point for chat.py
# =============================================================================

def start_twitch(
    *,
    fairy=None,
    mind=None,
    classifier: Optional[ChatClassifier] = None,
    tavily_api_key: Optional[str] = None,
    ready_timeout: float = 15.0,
) -> Optional[TwitchRuntime]:
    """
    Boot the Twitch integration in a background thread.

    Returns a TwitchRuntime handle on success, or None if Twitch is not
    configured (missing env vars). chat.py should treat None as "Twitch
    disabled this session" — same code path as `tts is None`.

    Args:
      fairy: pass alice's Fairy so search_web filters queries + results.
      mind:  alice.core.mind.Mind — if provided AND ALICE_MIND_CLASSIFIER=1
             (default 0), incoming chat is classified by Mind. Otherwise the
             heuristic default (mentions + cheers only) is used.
      classifier: explicit override; if set, used as-is and `mind` is ignored.
      tavily_api_key: web-search backend; defaults to env TAVILY_API_KEY.
      ready_timeout: how long to wait for the IRC connection before failing.
    """
    if os.environ.get("ALICE_TWITCH", "1") == "0":
        logger.info("twitch runtime disabled via ALICE_TWITCH=0")
        return None

    try:
        auth = load_auth()
    except RuntimeError as e:
        logger.info(f"twitch runtime not started — credentials missing: {e}")
        return None

    client = TwitchClient(auth)
    dispatcher = ToolDispatcher()
    register_all(
        dispatcher,
        client,
        tavily_api_key=tavily_api_key or os.environ.get("TAVILY_API_KEY"),
        fairy=fairy,
    )

    if classifier is None:
        use_mind = mind is not None and os.environ.get("ALICE_MIND_CLASSIFIER", "0") != "0"
        if use_mind:
            classifier = make_mind_classifier(mind, auth.bot_username)
            logger.info("twitch chat classifier: Mind + heuristic")
        else:
            classifier = _default_classifier(auth.bot_username)
            logger.info("twitch chat classifier: heuristic-only (set ALICE_MIND_CLASSIFIER=1 for Mind)")

    runtime = TwitchRuntime(client=client, dispatcher=dispatcher, classifier=classifier)
    try:
        runtime._start_thread(ready_timeout=ready_timeout)
    except Exception as e:
        logger.error(f"twitch runtime failed to start: {e}")
        return None

    logger.info(f"twitch runtime live as {auth.bot_username} on channel {auth.bot_username}")
    return runtime


__all__ = ["TwitchRuntime", "start_twitch", "ChatClassifier"]
