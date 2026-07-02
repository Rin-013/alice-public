"""
Read-only tool handlers — read_recent_chat, read_recent_superchats,
get_sub_count, read_recent_gift_subs.

`read_recent_chat` and the sub-event readers serve from the local rolling
buffers maintained by TwitchClient (no API call). `get_sub_count` hits
Helix `GET /helix/subscriptions` and counts.

Required scopes:
  channel:read:subscriptions   — for get_sub_count
  bits:read                    — for cheer events (already populated by IRC)
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..client import TwitchClient

log = logging.getLogger("alice.twitch.reading")


def _serialize_chat(msgs):
    return [
        {
            "msg_id": m.msg_id,
            "username": m.username,
            "display_name": m.display_name,
            "text": m.text,
            "timestamp": m.timestamp,
            "is_mod": m.is_mod,
            "is_sub": m.is_sub,
            "bits": m.bits,
        }
        for m in msgs
    ]


def _serialize_subs(events):
    return [
        {
            "event_type": e.event_type,
            "username": e.username,
            "timestamp": e.timestamp,
            "amount": e.amount,
            "tier": e.tier,
            "recipient": e.recipient,
            "message": e.message,
        }
        for e in events
    ]


async def read_recent_chat(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    count = int(args.get("count", 20))
    count = max(1, min(count, 50))
    filter_username = args.get("filter_username") or None
    since_seconds = args.get("since_seconds")
    if since_seconds is not None:
        since_seconds = int(since_seconds)

    msgs = client.recent_chat(count=count, filter_username=filter_username, since_seconds=since_seconds)
    return {"ok": True, "count": len(msgs), "messages": _serialize_chat(msgs)}


async def read_recent_superchats(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    count = int(args.get("count", 10))
    count = max(1, min(count, 20))
    event_type = args.get("event_type", "all")
    if event_type not in ("cheer", "sub", "gift_sub", "all"):
        event_type = "all"
    events = client.recent_sub_events(count=count, event_type=event_type)
    return {"ok": True, "count": len(events), "events": _serialize_subs(events)}


async def read_recent_gift_subs(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    count = int(args.get("count", 5))
    count = max(1, min(count, 20))
    events = client.recent_sub_events(count=count, event_type="gift_sub")
    return {"ok": True, "count": len(events), "events": _serialize_subs(events)}


async def get_sub_count(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Count current subscribers by paging Helix /subscriptions until we run
    out. Twitch returns up to 100 per page; cache result for 60s to avoid
    spamming.
    """
    import time
    cache = getattr(client, "_sub_count_cache", None)
    if cache and (time.time() - cache["ts"]) < 60.0:
        return {"ok": True, "sub_count": cache["count"], "cached": True}

    total = 0
    cursor = None
    while True:
        params = {"broadcaster_id": client.auth.broadcaster_id, "first": 100}
        if cursor:
            params["after"] = cursor
        try:
            data = await client.helix("GET", "/subscriptions", params=params)
        except RuntimeError as e:
            log.warning(f"get_sub_count failed: {e}")
            return {"ok": False, "error": str(e)}
        page = data.get("data", [])
        total += len(page)
        cursor = data.get("pagination", {}).get("cursor")
        if not cursor or not page:
            break

    client._sub_count_cache = {"count": total, "ts": time.time()}  # type: ignore
    return {"ok": True, "sub_count": total, "cached": False}


__all__ = ["read_recent_chat", "read_recent_superchats", "read_recent_gift_subs", "get_sub_count"]
