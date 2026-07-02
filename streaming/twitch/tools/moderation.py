"""
Moderation tool handlers — timeout_user, ban_user.

Both wrap Helix `POST /helix/moderation/bans`. Difference is whether
`duration` is included (timeout) or omitted (permanent ban).

Required scope: moderator:manage:banned_users
The user token must be from a user who is a moderator on the channel
(the broadcaster themselves works).
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from ..client import TwitchClient

log = logging.getLogger("alice.twitch.moderation")


async def timeout_user(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    """Timeout a chatter for `duration_seconds`. Returns the Helix payload."""
    username = args["username"]
    duration = int(args["duration_seconds"])
    reason = args.get("reason", "")[:500]

    user_id = await client.get_user_id(username)
    if not user_id:
        return {"ok": False, "error": f"user '{username}' not found"}

    payload = {
        "data": {
            "user_id": user_id,
            "duration": duration,
            "reason": reason,
        }
    }
    params = {
        "broadcaster_id": client.auth.broadcaster_id,
        "moderator_id": client.auth.broadcaster_id,
    }
    try:
        result = await client.helix("POST", "/moderation/bans", params=params, json_body=payload)
    except RuntimeError as e:
        log.warning(f"timeout_user failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "username": username, "duration_seconds": duration, "result": result}


async def ban_user(client: TwitchClient, args: Dict[str, Any]) -> Dict[str, Any]:
    """Permanently ban a chatter."""
    username = args["username"]
    reason = args["reason"][:500]

    user_id = await client.get_user_id(username)
    if not user_id:
        return {"ok": False, "error": f"user '{username}' not found"}

    payload = {"data": {"user_id": user_id, "reason": reason}}
    params = {
        "broadcaster_id": client.auth.broadcaster_id,
        "moderator_id": client.auth.broadcaster_id,
    }
    try:
        result = await client.helix("POST", "/moderation/bans", params=params, json_body=payload)
    except RuntimeError as e:
        log.warning(f"ban_user failed: {e}")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "username": username, "result": result}


__all__ = ["timeout_user", "ban_user"]
