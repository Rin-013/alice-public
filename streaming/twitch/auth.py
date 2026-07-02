"""
Twitch OAuth token management.

Loads credentials from `.env` (or environment variables directly) and
refreshes the user access token when it's about to expire. Tokens are
cached in `alice/data/twitch_tokens.json` (gitignored) so we don't have
to re-prompt every restart.

Environment variables expected (see .env.example):
  TWITCH_CLIENT_ID        — from dev console app
  TWITCH_CLIENT_SECRET    — from dev console app
  TWITCH_USER_TOKEN       — initial user access token (broadcaster scope)
  TWITCH_REFRESH_TOKEN    — initial refresh token
  TWITCH_BROADCASTER_ID   — numeric user ID of the channel Alice streams on
  TWITCH_BOT_USERNAME     — login name Alice posts under (often the broadcaster)

Required scopes on the user token:
  chat:read chat:edit channel:moderate
  moderator:manage:banned_users moderator:manage:chat_messages
  channel:manage:polls channel:manage:predictions
  clips:edit channel:manage:broadcast
  bits:read channel:read:subscriptions

If the token expires, refresh via:
  POST https://id.twitch.tv/oauth2/token
    grant_type=refresh_token
    refresh_token=<...>
    client_id=<...>
    client_secret=<...>
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("alice.twitch.auth")

REPO_ROOT = Path(__file__).resolve().parents[3]
TOKEN_CACHE_PATH = REPO_ROOT / "alice" / "data" / "twitch_tokens.json"

OAUTH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"
OAUTH_VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"


@dataclass
class TwitchAuth:
    client_id: str
    client_secret: str
    user_token: str
    refresh_token: str
    broadcaster_id: str
    bot_username: str
    expires_at: float = 0.0  # unix timestamp; 0 means unknown


def _load_env() -> TwitchAuth:
    """Read credentials from environment. Raises if any required field is missing."""
    required = {
        "TWITCH_CLIENT_ID": "client_id",
        "TWITCH_CLIENT_SECRET": "client_secret",
        "TWITCH_USER_TOKEN": "user_token",
        "TWITCH_REFRESH_TOKEN": "refresh_token",
        "TWITCH_BROADCASTER_ID": "broadcaster_id",
        "TWITCH_BOT_USERNAME": "bot_username",
    }
    values = {}
    missing = []
    for env_key, attr in required.items():
        v = os.environ.get(env_key, "").strip()
        if not v:
            missing.append(env_key)
        values[attr] = v
    if missing:
        raise RuntimeError(
            f"Missing Twitch env vars: {', '.join(missing)}. "
            f"See alice/core/twitch/README.md for setup."
        )
    return TwitchAuth(**values)


def _load_cached_tokens() -> Optional[dict]:
    if not TOKEN_CACHE_PATH.exists():
        return None
    try:
        with TOKEN_CACHE_PATH.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"twitch token cache unreadable, ignoring: {e}")
        return None


def _save_cached_tokens(auth: TwitchAuth) -> None:
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOKEN_CACHE_PATH.open("w") as f:
        json.dump(
            {
                "user_token": auth.user_token,
                "refresh_token": auth.refresh_token,
                "expires_at": auth.expires_at,
            },
            f,
        )


def load_auth() -> TwitchAuth:
    """Load credentials. Prefers cached tokens (newer) over env defaults."""
    auth = _load_env()
    cached = _load_cached_tokens()
    if cached:
        auth.user_token = cached.get("user_token", auth.user_token)
        auth.refresh_token = cached.get("refresh_token", auth.refresh_token)
        auth.expires_at = float(cached.get("expires_at", 0.0))
    return auth


def is_expired(auth: TwitchAuth, skew_seconds: int = 60) -> bool:
    """True if the token is expired or within `skew_seconds` of expiring."""
    if auth.expires_at == 0.0:
        return False  # unknown — assume valid until validate proves otherwise
    return time.time() >= (auth.expires_at - skew_seconds)


async def validate(auth: TwitchAuth) -> Optional[dict]:
    """
    Hit /oauth2/validate. Returns the validation payload (login, scopes,
    expires_in) on success, or None on failure.
    """
    headers = {"Authorization": f"OAuth {auth.user_token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(OAUTH_VALIDATE_URL, headers=headers)
            if r.status_code == 200:
                data = r.json()
                # Update expires_at from the response so we know when to refresh
                if "expires_in" in data:
                    auth.expires_at = time.time() + float(data["expires_in"])
                return data
            logger.warning(f"twitch validate {r.status_code}: {r.text[:200]}")
            return None
        except httpx.HTTPError as e:
            logger.warning(f"twitch validate failed: {e}")
            return None


async def refresh(auth: TwitchAuth) -> bool:
    """
    Refresh the user access token. Updates `auth` in place and writes the
    new tokens to the cache file. Returns True on success.
    """
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": auth.refresh_token,
        "client_id": auth.client_id,
        "client_secret": auth.client_secret,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(OAUTH_TOKEN_URL, data=payload)
        except httpx.HTTPError as e:
            logger.error(f"twitch refresh transport failed: {e}")
            return False
    if r.status_code != 200:
        logger.error(f"twitch refresh {r.status_code}: {r.text[:300]}")
        return False
    data = r.json()
    auth.user_token = data["access_token"]
    auth.refresh_token = data.get("refresh_token", auth.refresh_token)
    expires_in = float(data.get("expires_in", 3600))
    auth.expires_at = time.time() + expires_in
    _save_cached_tokens(auth)
    logger.info(f"twitch token refreshed; expires in {expires_in:.0f}s")
    return True


async def ensure_valid(auth: TwitchAuth) -> bool:
    """
    Validate-then-refresh-if-needed. Call before any Helix request that
    can't tolerate a 401. Returns True if the token is valid after this.
    """
    if not is_expired(auth):
        info = await validate(auth)
        if info is not None:
            return True
    return await refresh(auth)


__all__ = ["TwitchAuth", "load_auth", "validate", "refresh", "ensure_valid", "is_expired"]
