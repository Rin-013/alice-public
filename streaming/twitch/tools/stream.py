"""
Stream-info tool handler — update_stream_info.

Helix `PATCH /helix/channels?broadcaster_id=...`
Required scope: channel:manage:broadcast

At least one of `title`, `game_id`, or `tags` must be provided.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..client import TwitchClient

log = logging.getLogger("alice.twitch.stream")


async def update_stream_info(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if "title" in args and args["title"]:
        body["title"] = str(args["title"])[:140]
    if "game_id" in args and args["game_id"]:
        body["game_id"] = str(args["game_id"])
    if "tags" in args and args["tags"]:
        body["tags"] = [str(t)[:25] for t in args["tags"][:10]]
    if not body:
        return {"ok": False, "error": "need at least one of: title, game_id, tags"}

    params = {"broadcaster_id": client.auth.broadcaster_id}
    try:
        await client.helix("PATCH", "/channels", params=params, json_body=body)
    except RuntimeError as e:
        log.warning(f"update_stream_info failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "updated": body}


__all__ = ["update_stream_info"]
