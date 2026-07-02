"""
Engagement tool handlers — create_poll, create_prediction, pin_chat_message,
request_clip.

Required scopes:
  channel:manage:polls           — polls
  channel:manage:predictions     — predictions
  moderator:manage:chat_messages — pin (announcement endpoint)
  clips:edit                     — request_clip
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..client import TwitchClient

log = logging.getLogger("alice.twitch.engagement")


async def create_poll(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    title = args["title"][:60]
    choices = [{"title": str(c)[:25]} for c in args["choices"][:5]]
    if len(choices) < 2:
        return {"ok": False, "error": "need at least 2 choices"}
    duration = int(args.get("duration_seconds", 60))
    duration = max(15, min(duration, 1800))

    payload = {
        "broadcaster_id": client.auth.broadcaster_id,
        "title": title,
        "choices": choices,
        "duration": duration,
    }
    try:
        result = await client.helix("POST", "/polls", json_body=payload)
    except RuntimeError as e:
        log.warning(f"create_poll failed: {e}")
        return {"ok": False, "error": str(e)}
    poll = result.get("data", [{}])[0] if result.get("data") else {}
    return {"ok": True, "poll_id": poll.get("id"), "title": title, "choices": [c["title"] for c in choices], "duration_seconds": duration}


async def create_prediction(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    title = args["title"][:45]
    outcomes = [{"title": str(o)[:25]} for o in args["outcomes"][:10]]
    if len(outcomes) < 2:
        return {"ok": False, "error": "need at least 2 outcomes"}
    duration = int(args.get("duration_seconds", 120))
    duration = max(30, min(duration, 1800))

    payload = {
        "broadcaster_id": client.auth.broadcaster_id,
        "title": title,
        "outcomes": outcomes,
        "prediction_window": duration,
    }
    try:
        result = await client.helix("POST", "/predictions", json_body=payload)
    except RuntimeError as e:
        log.warning(f"create_prediction failed: {e}")
        return {"ok": False, "error": str(e)}
    pred = result.get("data", [{}])[0] if result.get("data") else {}
    return {"ok": True, "prediction_id": pred.get("id"), "title": title, "outcomes": [o["title"] for o in outcomes], "duration_seconds": duration}


async def pin_chat_message(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pin a chat message OR post a pinned announcement.

    Twitch Helix has no direct "pin existing message by ID" endpoint, but it
    does have /chat/announcements which posts a sticky-styled message that
    behaves like a pin. If `text` is provided we use that; if `message_id`
    is provided we look up the message in the buffer and re-announce its
    content (best we can do via Helix today).
    """
    message_id = args.get("message_id", "").strip()
    text = args.get("text", "").strip()

    if not text and message_id:
        # Look up the message in the buffer and use its content
        msg = next((m for m in client.recent_chat(count=500) if m.msg_id == message_id), None)
        if msg is None:
            return {"ok": False, "error": f"message_id '{message_id}' not in buffer"}
        text = f"📌 {msg.display_name}: {msg.text}"

    if not text:
        return {"ok": False, "error": "need either message_id or text"}

    text = text[:500]
    payload = {"message": text}
    params = {
        "broadcaster_id": client.auth.broadcaster_id,
        "moderator_id": client.auth.broadcaster_id,
    }
    try:
        await client.helix("POST", "/chat/announcements", params=params, json_body=payload)
    except RuntimeError as e:
        log.warning(f"pin_chat_message failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "pinned_text": text}


async def request_clip(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    has_delay = bool(args.get("has_delay", False))
    params = {
        "broadcaster_id": client.auth.broadcaster_id,
        "has_delay": str(has_delay).lower(),
    }
    try:
        result = await client.helix("POST", "/clips", params=params)
    except RuntimeError as e:
        log.warning(f"request_clip failed: {e}")
        return {"ok": False, "error": str(e)}
    clip = result.get("data", [{}])[0] if result.get("data") else {}
    return {
        "ok": True,
        "clip_id": clip.get("id"),
        "edit_url": clip.get("edit_url"),
        "has_delay": has_delay,
    }


__all__ = ["create_poll", "create_prediction", "pin_chat_message", "request_clip"]
