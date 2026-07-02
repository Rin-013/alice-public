"""
Tool schemas for Alice's Twitch tool ecosystem.

Each schema describes a tool Alice can fire from her LLM output. The
dispatcher (dispatcher.py) consumes ALICE_TOOLS to validate calls and
pass them to per-tool handlers.

Schema shape mirrors the Anthropic tool-use format:

    {
        "name": "<tool_name>",
        "description": "<when/why Alice should use it, in her voice context>",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...],
        },
    }

Buckets:
  - PERCEPTION_TOOLS: read-only — search, read chat, read sub history
  - ACTION_TOOLS:     mutating — moderate, poll, predict, pin, clip, update info

Each tool also defines its rate-limit profile (folded into RATE_LIMITS in
the dispatcher). Rate limits are intentionally conservative so Alice
doesn't get the channel ratelimited or banned.
"""
from __future__ import annotations

from typing import Any, Dict, List

# =============================================================================
# Perception tools (read-only)
# =============================================================================

SEARCH_WEB = {
    "name": "search_web",
    "description": (
        "Search the internet for current information. Use this when chat "
        "asks about something happening in the world, when you want to look "
        "up a fact you're not sure about, or when you want to react to news, "
        "trends, or anything beyond your knowledge. Don't use this for "
        "things you already know — it costs latency. Returns a short summary "
        "with sources."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — 3-8 words usually, like a search-engine query, not a question.",
                "minLength": 2,
                "maxLength": 200,
            },
            "intent": {
                "type": "string",
                "description": "Why you're searching — picks a result strategy.",
                "enum": ["news", "fact_check", "topic_lookup", "trend_check", "reference"],
            },
            "freshness": {
                "type": "string",
                "description": "How recent results should be.",
                "enum": ["day", "week", "month", "any"],
                "default": "any",
            },
        },
        "required": ["query", "intent"],
    },
}

READ_RECENT_CHAT = {
    "name": "read_recent_chat",
    "description": (
        "Read the last N messages from your Twitch chat. Use when chat is "
        "moving fast and you missed something, when you want to find someone "
        "specific to engage with, or when you suspect chat is reacting to "
        "something you didn't see. Don't use every turn — most of the time "
        "you can just respond to what's already in your context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "How many recent messages to fetch (1-50).",
                "minimum": 1,
                "maximum": 50,
                "default": 20,
            },
            "filter_username": {
                "type": "string",
                "description": "Optional — only return messages from this user.",
            },
            "since_seconds": {
                "type": "integer",
                "description": "Optional — only messages from the last N seconds.",
                "minimum": 1,
                "maximum": 3600,
            },
        },
        "required": [],
    },
}

READ_RECENT_SUPERCHATS = {
    "name": "read_recent_superchats",
    "description": (
        "Read recent bits/cheers and sub events from your channel. Use when "
        "chat or your own context references subs/cheers and you should "
        "know who/what, or when someone just supported and you should "
        "acknowledge them. Twitch's equivalent of superchats is bits + subs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "How many recent events to fetch (1-20).",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
            "event_type": {
                "type": "string",
                "description": "Which event type. 'all' is the default.",
                "enum": ["cheer", "sub", "gift_sub", "all"],
                "default": "all",
            },
        },
        "required": [],
    },
}

GET_SUB_COUNT = {
    "name": "get_sub_count",
    "description": (
        "Get your current total sub count. Use when chat references subs "
        "in a way that requires knowing the actual number, or when someone "
        "subs and you want to mention how many you're at. Cheap call — but "
        "still don't fire it for no reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

READ_RECENT_GIFT_SUBS = {
    "name": "read_recent_gift_subs",
    "description": (
        "Read your most recent gift-sub events specifically. Use when "
        "someone just gifted subs and you should call them out, or when "
        "you want to recap recent gifters. More focused than "
        "read_recent_superchats."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "How many recent gift-sub events to fetch (1-20).",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": [],
    },
}


# =============================================================================
# Action tools (mutating)
# =============================================================================

TIMEOUT_USER = {
    "name": "timeout_user",
    "description": (
        "Timeout a chatter. Use when someone's being annoying, spammy, or "
        "trying to derail you. Be confident — don't apologize. Standard "
        "durations: 60s (warning), 300s (5 min, real), 600s (10 min, "
        "serious). Don't timeout for disagreement — that's lame."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "description": "Twitch username (without @).",
                "minLength": 1,
                "maxLength": 25,
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Timeout length. 60 / 300 / 600 are typical.",
                "minimum": 1,
                "maximum": 1209600,  # 14 days, Twitch max
            },
            "reason": {
                "type": "string",
                "description": "Honest reason in your voice. Not corporate.",
                "maxLength": 500,
            },
        },
        "required": ["username", "duration_seconds", "reason"],
    },
}

BAN_USER = {
    "name": "ban_user",
    "description": (
        "Permanently ban a chatter. Heavy — use for harassment, slurs, "
        "doxing, sustained bad behavior after timeouts. Default to timeout "
        "first; ban is the escalation. Always include a real reason."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "description": "Twitch username (without @).",
                "minLength": 1,
                "maxLength": 25,
            },
            "reason": {
                "type": "string",
                "description": "Concrete reason — what specifically they did.",
                "minLength": 1,
                "maxLength": 500,
            },
        },
        "required": ["username", "reason"],
    },
}

CREATE_POLL = {
    "name": "create_poll",
    "description": (
        "Open a Twitch poll. Use when chat is split on something and you "
        "want to see who wins, when you want chat to make a decision you "
        "don't want to make, or when energy needs a focal point. Don't "
        "poll every 5 minutes — novelty dies."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Poll question (Twitch caps at 60 chars).",
                "minLength": 1,
                "maxLength": 60,
            },
            "choices": {
                "type": "array",
                "description": "2-5 choices, each ≤25 chars.",
                "items": {"type": "string", "maxLength": 25},
                "minItems": 2,
                "maxItems": 5,
            },
            "duration_seconds": {
                "type": "integer",
                "description": "How long the poll runs (15-1800s).",
                "minimum": 15,
                "maximum": 1800,
                "default": 60,
            },
        },
        "required": ["title", "choices"],
    },
}

CREATE_PREDICTION = {
    "name": "create_prediction",
    "description": (
        "Open a Twitch channel-points prediction. Use when there's a real "
        "uncertain outcome coming up — 'will I beat this boss in 3 tries', "
        "'which guest answers first'. Bad: predictions on things already "
        "decided, or with no real outcome. These cost channel points — "
        "don't waste them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Prediction question (Twitch caps at 45 chars).",
                "minLength": 1,
                "maxLength": 45,
            },
            "outcomes": {
                "type": "array",
                "description": "2-10 outcomes, each ≤25 chars.",
                "items": {"type": "string", "maxLength": 25},
                "minItems": 2,
                "maxItems": 10,
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Submission window (30-1800s).",
                "minimum": 30,
                "maximum": 1800,
                "default": 120,
            },
        },
        "required": ["title", "outcomes"],
    },
}

PIN_CHAT_MESSAGE = {
    "name": "pin_chat_message",
    "description": (
        "Pin a message to the top of chat. Use when someone says something "
        "funny enough to deserve immortality, or when there's important "
        "info chat needs to keep seeing. Replaces whatever's currently "
        "pinned, so make it count."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "Twitch message ID to pin (from read_recent_chat).",
            },
            "text": {
                "type": "string",
                "description": "Alternative — pin a custom announcement instead of a chat message.",
                "maxLength": 500,
            },
        },
        "required": [],  # one of message_id or text — validated in handler
    },
}

REQUEST_CLIP = {
    "name": "request_clip",
    "description": (
        "Request a clip of the current moment. Use when something just "
        "happened that should live forever — great moment, great joke, "
        "perfect timing, chat reaction worth saving. Don't ask constantly. "
        "This signals mods/clip-bots to actually create the clip."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "has_delay": {
                "type": "boolean",
                "description": "If true, account for ~4s broadcast delay (catches the moment that just passed).",
                "default": False,
            },
        },
        "required": [],
    },
}

UPDATE_STREAM_INFO = {
    "name": "update_stream_info",
    "description": (
        "Update the stream title or game category. Use when you switch "
        "what you're doing, or when the title is stale and a new bit is "
        "happening. Don't change it constantly — it's not a chat box."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "New stream title.",
                "maxLength": 140,
            },
            "game_id": {
                "type": "string",
                "description": "New game/category ID (Twitch internal ID — fetch via search_categories).",
            },
            "tags": {
                "type": "array",
                "description": "Stream tags (max 10, each ≤25 chars).",
                "items": {"type": "string", "maxLength": 25},
                "maxItems": 10,
            },
        },
        "required": [],  # at least one of title/game_id/tags — validated in handler
    },
}


# =============================================================================
# Public exports
# =============================================================================

PERCEPTION_TOOLS: List[Dict[str, Any]] = [
    SEARCH_WEB,
    READ_RECENT_CHAT,
    READ_RECENT_SUPERCHATS,
    GET_SUB_COUNT,
    READ_RECENT_GIFT_SUBS,
]

ACTION_TOOLS: List[Dict[str, Any]] = [
    TIMEOUT_USER,
    BAN_USER,
    CREATE_POLL,
    CREATE_PREDICTION,
    PIN_CHAT_MESSAGE,
    REQUEST_CLIP,
    UPDATE_STREAM_INFO,
]

ALICE_TOOLS: List[Dict[str, Any]] = PERCEPTION_TOOLS + ACTION_TOOLS

TOOL_NAMES: List[str] = [t["name"] for t in ALICE_TOOLS]


__all__ = [
    "ALICE_TOOLS",
    "TOOL_NAMES",
    "PERCEPTION_TOOLS",
    "ACTION_TOOLS",
    # Individual schemas (so handlers can import the one they care about)
    "SEARCH_WEB",
    "READ_RECENT_CHAT",
    "READ_RECENT_SUPERCHATS",
    "GET_SUB_COUNT",
    "READ_RECENT_GIFT_SUBS",
    "TIMEOUT_USER",
    "BAN_USER",
    "CREATE_POLL",
    "CREATE_PREDICTION",
    "PIN_CHAT_MESSAGE",
    "REQUEST_CLIP",
    "UPDATE_STREAM_INFO",
]

# Rate limits live in dispatcher.py (RATE_LIMITS dict, RateLimit dataclass).
# Single source of truth. Tools added here should also get a RateLimit entry there.
