"""
Tool handler registry.

`register_all(dispatcher, twitch_client, tavily_api_key=None, fairy=None)`
wires every handler in this package into the dispatcher.

Two flavors:
  - perception/action handlers in this package take `(client, args)` and
    return a dict. They get adapted to the dispatcher's ToolHandler signature
    (which takes a single `args` dict and returns a result).
  - search_web is special: it has its own SearchHandler with a pluggable
    Fairy filter (TOS + security stack) and runs against
    Tavily, not Helix. Built here.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from ..client import TwitchClient
from . import moderation, engagement, stream, reading
from .search import SearchHandler, TavilyBackend, PassthroughFilter, BasicFairyStub, SearchResponse

log = logging.getLogger("alice.twitch.tools")


# Map of tool_name → handler factory. Each factory takes the bound resources
# (client, search_handler) and returns the per-tool async handler the
# dispatcher expects: `async def(args: dict) -> dict`.
def _bind_helix_tool(client: TwitchClient, fn: Callable[[TwitchClient, Dict], Awaitable[Dict]]):
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        return await fn(client, args)
    return handler


def _bind_search(search_handler: SearchHandler):
    async def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        resp: SearchResponse = await search_handler.search(
            query=args["query"],
            intent=args["intent"],
            freshness=args.get("freshness", "any"),
        )
        return {
            "ok": not resp.blocked,
            "summary": resp.summary,
            "sources": resp.sources_for_attribution,
            "result_count": len(resp.results),
            "blocked": resp.blocked,
            "blocked_reason": resp.blocked_reason,
            "duration_ms": resp.duration_ms,
        }
    return handler


def register_all(
    dispatcher,
    client: TwitchClient,
    *,
    tavily_api_key: Optional[str] = None,
    fairy=None,
    summarizer: Optional[Callable] = None,
) -> None:
    """
    Register every Twitch tool handler with the dispatcher.

    `dispatcher` should be a Dispatcher instance from `..dispatcher`.
    `client` is a connected TwitchClient.
    `tavily_api_key` enables real web search; if None, search_web returns
    a clean error result.
    `fairy` is the unified Fairy filter (TOS + security stack)
    for search (see tools/search.py for the Protocol). PassthroughFilter
    is used by default so everything works during dev.
    """
    # Helix tools (8)
    dispatcher.register("timeout_user", _bind_helix_tool(client, moderation.timeout_user))
    dispatcher.register("ban_user", _bind_helix_tool(client, moderation.ban_user))
    dispatcher.register("create_poll", _bind_helix_tool(client, engagement.create_poll))
    dispatcher.register("create_prediction", _bind_helix_tool(client, engagement.create_prediction))
    dispatcher.register("pin_chat_message", _bind_helix_tool(client, engagement.pin_chat_message))
    dispatcher.register("request_clip", _bind_helix_tool(client, engagement.request_clip))
    dispatcher.register("update_stream_info", _bind_helix_tool(client, stream.update_stream_info))
    dispatcher.register("get_sub_count", _bind_helix_tool(client, reading.get_sub_count))

    # Local-buffer tools (3)
    dispatcher.register("read_recent_chat", _bind_helix_tool(client, reading.read_recent_chat))
    dispatcher.register("read_recent_superchats", _bind_helix_tool(client, reading.read_recent_superchats))
    dispatcher.register("read_recent_gift_subs", _bind_helix_tool(client, reading.read_recent_gift_subs))

    # Search tool — has its own filter pipeline
    if tavily_api_key:
        search_handler = SearchHandler(
            api_key=tavily_api_key,
            fairy=fairy or PassthroughFilter(),
            summarizer=summarizer,
        )
        dispatcher.register("search_web", _bind_search(search_handler))
    else:
        async def disabled(args):
            return {
                "ok": False,
                "error": "search_web disabled — TAVILY_API_KEY not set",
            }
        dispatcher.register("search_web", disabled)

    log.info(f"registered {len(dispatcher._handlers)} tool handlers")


__all__ = ["register_all"]
