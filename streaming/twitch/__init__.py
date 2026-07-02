"""
Twitch — Alice's tool ecosystem for Twitch streaming.

Public surface:
    TwitchClient          — IRC + Helix wrapper (twitchio under the hood)
    TwitchAuth, load_auth — OAuth credential mgmt
    ToolDispatcher        — tool-call validator + router (from dispatcher.py)
    register_all          — wire every tool handler into a Dispatcher
    build_system_prompt   — assemble Alice's tool-aware system prompt
    ALICE_TOOLS, TOOL_NAMES, PERCEPTION_TOOLS, ACTION_TOOLS — schemas

Wiring example (in chat.py or a stream entry script):

    from streaming.twitch import (
        load_auth, TwitchClient, Dispatcher, register_all,
        build_system_prompt, ALICE_TOOLS,
    )

    auth = load_auth()
    client = TwitchClient(auth, on_message=alice_on_chat_message)
    await client.connect()

    dispatcher = ToolDispatcher()  # see dispatcher.py
    register_all(
        dispatcher,
        client,
        tavily_api_key=os.environ.get("TAVILY_API_KEY"),
        # fairy=registry.get("fairy"),  # plug Alice's real Fairy in (TOS + security)
    )

    system_prompt = build_system_prompt(runtime_state="...")
    # Pass ALICE_TOOLS to Alice's LLM as the tool list.

See README.md in this directory for the full setup (Twitch dev console
app, scopes, .env vars).
"""
from .auth import TwitchAuth, load_auth, ensure_valid
from .client import TwitchClient, ChatMessage, SubEvent
from .dispatcher import ToolDispatcher, ToolResult, ToolStatus
from .system_prompt import build_system_prompt, ALICE_CORE, TOOL_AWARENESS, TOOL_GUIDANCE, FEW_SHOT_EXAMPLES
from .tool_schemas import (
    ALICE_TOOLS,
    TOOL_NAMES,
    PERCEPTION_TOOLS,
    ACTION_TOOLS,
)
from .tools import register_all

__all__ = [
    # Auth
    "TwitchAuth",
    "load_auth",
    "ensure_valid",
    # Client
    "TwitchClient",
    "ChatMessage",
    "SubEvent",
    # Dispatcher
    "ToolDispatcher",
    "ToolResult",
    "ToolStatus",
    "register_all",
    # System prompt
    "build_system_prompt",
    "ALICE_CORE",
    "TOOL_AWARENESS",
    "TOOL_GUIDANCE",
    "FEW_SHOT_EXAMPLES",
    # Schemas
    "ALICE_TOOLS",
    "TOOL_NAMES",
    "PERCEPTION_TOOLS",
    "ACTION_TOOLS",
]
