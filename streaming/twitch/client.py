"""
TwitchClient — thin async wrapper around twitchio.

Owns:
  - The IRC chat connection (receive messages, post replies)
  - A Helix REST client for the action tools (timeout/ban/poll/prediction/etc.)
  - A rolling local buffer of recent chat messages (for read_recent_chat)
  - Subs/cheers event log (for read_recent_superchats / read_recent_gift_subs)

Designed to run inside Alice's existing async loop. Tool handlers in
`tools/` call into this client; they don't import twitchio directly so
the dependency stays in one place.

Lifecycle:
    auth = load_auth()
    client = TwitchClient(auth)
    await client.connect()
    # ... Alice runs ...
    await client.disconnect()
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, List, Optional

from .auth import TwitchAuth, ensure_valid

logger = logging.getLogger("alice.twitch.client")

# Lazy import — twitchio is only loaded when the client is actually used.
# Keeps import-time cheap on machines without twitchio installed (e.g. dev box
# without Twitch credentials).
_twitchio = None


def _import_twitchio():
    global _twitchio
    if _twitchio is None:
        try:
            import twitchio  # type: ignore
            _twitchio = twitchio
        except ImportError as e:
            raise RuntimeError(
                "twitchio not installed. `pip install twitchio` to use TwitchClient."
            ) from e
    return _twitchio


# =============================================================================
# Buffered chat / sub-event records
# =============================================================================

@dataclass
class ChatMessage:
    msg_id: str
    username: str
    display_name: str
    text: str
    timestamp: float
    is_mod: bool = False
    is_sub: bool = False
    bits: int = 0
    badges: tuple = ()


@dataclass
class SubEvent:
    """Unified record for cheer / sub / gift_sub events."""
    event_type: str  # "cheer", "sub", "gift_sub"
    username: str
    timestamp: float
    amount: int = 0          # bits for cheer, months for sub, count for gift_sub
    tier: str = ""           # "1000", "2000", "3000" for subs
    recipient: str = ""      # for gift_sub: who received it
    message: str = ""        # the user's resub/cheer message


# =============================================================================
# TwitchClient
# =============================================================================

class TwitchClient:
    """
    Async Twitch client. Connects to chat IRC, exposes Helix endpoints,
    keeps rolling buffers for the perception tools.
    """

    def __init__(
        self,
        auth: TwitchAuth,
        chat_buffer_size: int = 500,
        sub_event_buffer_size: int = 200,
        on_message: Optional[Callable[[ChatMessage], Awaitable[None]]] = None,
    ):
        self.auth = auth
        self.on_message = on_message  # external callback (e.g. feed Alice)
        self._chat_buffer: Deque[ChatMessage] = deque(maxlen=chat_buffer_size)
        self._sub_buffer: Deque[SubEvent] = deque(maxlen=sub_event_buffer_size)
        self._bot = None  # twitchio.Client
        self._connected = False

    # -- Chat buffer accessors -------------------------------------------------

    def recent_chat(
        self,
        count: int = 20,
        filter_username: Optional[str] = None,
        since_seconds: Optional[int] = None,
    ) -> List[ChatMessage]:
        """Filter the rolling chat buffer. Newest last."""
        msgs = list(self._chat_buffer)
        if filter_username:
            target = filter_username.lower()
            msgs = [m for m in msgs if m.username.lower() == target]
        if since_seconds is not None:
            cutoff = time.time() - since_seconds
            msgs = [m for m in msgs if m.timestamp >= cutoff]
        return msgs[-count:]

    def recent_sub_events(
        self,
        count: int = 10,
        event_type: str = "all",
    ) -> List[SubEvent]:
        events = list(self._sub_buffer)
        if event_type != "all":
            events = [e for e in events if e.event_type == event_type]
        return events[-count:]

    # -- Connection lifecycle --------------------------------------------------

    async def connect(self) -> None:
        """Open the IRC connection and start consuming events."""
        twitchio = _import_twitchio()
        await ensure_valid(self.auth)

        # twitchio's Client wants the OAuth token prefixed with "oauth:" for IRC
        irc_token = self.auth.user_token
        if not irc_token.startswith("oauth:"):
            irc_token = f"oauth:{irc_token}"

        self._bot = twitchio.Client(
            token=irc_token,
            initial_channels=[self.auth.bot_username],
        )

        client_self = self

        @self._bot.event()
        async def event_message(message):
            if message.echo:
                return
            cm = ChatMessage(
                msg_id=str(message.tags.get("id", "")) if message.tags else "",
                username=message.author.name,
                display_name=message.author.display_name,
                text=message.content,
                timestamp=time.time(),
                is_mod=bool(getattr(message.author, "is_mod", False)),
                is_sub=bool(getattr(message.author, "is_subscriber", False)),
                bits=int(message.tags.get("bits", 0)) if message.tags else 0,
                badges=tuple((message.author.badges or {}).keys()),
            )
            client_self._chat_buffer.append(cm)
            if cm.bits > 0:
                client_self._sub_buffer.append(SubEvent(
                    event_type="cheer",
                    username=cm.username,
                    timestamp=cm.timestamp,
                    amount=cm.bits,
                    message=cm.text,
                ))
            if client_self.on_message:
                try:
                    await client_self.on_message(cm)
                except Exception as e:
                    logger.warning(f"on_message callback raised: {e}")

        # Sub / gift events come through USERNOTICE — twitchio raises event_raw_usernotice
        @self._bot.event()
        async def event_raw_usernotice(channel, tags):
            t = tags.get("msg-id", "")
            ts = time.time()
            if t in ("sub", "resub"):
                client_self._sub_buffer.append(SubEvent(
                    event_type="sub",
                    username=tags.get("login", ""),
                    timestamp=ts,
                    amount=int(tags.get("msg-param-cumulative-months", 1)),
                    tier=tags.get("msg-param-sub-plan", "1000"),
                    message=tags.get("system-msg", ""),
                ))
            elif t in ("subgift", "anonsubgift"):
                client_self._sub_buffer.append(SubEvent(
                    event_type="gift_sub",
                    username=tags.get("login", "anonymous"),
                    timestamp=ts,
                    amount=1,
                    tier=tags.get("msg-param-sub-plan", "1000"),
                    recipient=tags.get("msg-param-recipient-user-name", ""),
                ))

        # Start the IRC loop in the background
        self._task = asyncio.create_task(self._bot.start())
        # Tiny wait so connection is established before tools fire
        await asyncio.sleep(2.0)
        self._connected = True
        logger.info(f"twitch connected as {self.auth.bot_username}")

    async def disconnect(self) -> None:
        if self._bot is not None:
            try:
                await self._bot.close()
            except Exception as e:
                logger.warning(f"twitch close error: {e}")
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    # -- Chat send -------------------------------------------------------------

    async def send_chat(self, text: str) -> None:
        """Post a message to chat. Rate-limited by Twitch (don't spam)."""
        if not self._bot:
            raise RuntimeError("TwitchClient not connected.")
        channel = self._bot.get_channel(self.auth.bot_username)
        if channel is None:
            raise RuntimeError(f"channel {self.auth.bot_username} not joined")
        await channel.send(text)

    # -- Helix request ---------------------------------------------------------

    async def helix(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> dict:
        """
        Make a raw Helix request. Auto-refreshes on 401. Returns parsed JSON
        on 2xx; raises on persistent failure.
        """
        import httpx
        await ensure_valid(self.auth)

        url = f"https://api.twitch.tv/helix{path}"
        headers = {
            "Client-Id": self.auth.client_id,
            "Authorization": f"Bearer {self.auth.user_token}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=15.0) as http:
            r = await http.request(method, url, params=params, json=json_body, headers=headers)
            if r.status_code == 401:
                # Token expired between validate and request — refresh once and retry
                from .auth import refresh
                if await refresh(self.auth):
                    headers["Authorization"] = f"Bearer {self.auth.user_token}"
                    r = await http.request(method, url, params=params, json=json_body, headers=headers)
            if not r.is_success:
                raise RuntimeError(f"Helix {method} {path} {r.status_code}: {r.text[:300]}")
            if r.status_code == 204 or not r.content:
                return {}
            return r.json()

    async def get_user_id(self, username: str) -> Optional[str]:
        """Helix lookup of a username → numeric user_id. Cached per-process."""
        if not hasattr(self, "_user_id_cache"):
            self._user_id_cache: dict[str, str] = {}
        u = username.lstrip("@").lower()
        if u in self._user_id_cache:
            return self._user_id_cache[u]
        data = await self.helix("GET", "/users", params={"login": u})
        items = data.get("data", [])
        if not items:
            return None
        uid = items[0]["id"]
        self._user_id_cache[u] = uid
        return uid


__all__ = ["TwitchClient", "ChatMessage", "SubEvent"]
