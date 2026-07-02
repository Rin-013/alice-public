# Twitch — Alice's tool ecosystem

> **Status**: Wired (Apr 28 2026). Needs a Twitch dev console app + tokens to run live.
> **Path**: `streaming/twitch/` (moved from `alice/core/twitch/` June 2026)

The macro that lets Alice connect to her Twitch chat, react to events, and fire actions (timeouts, polls, clips, etc.) from her LLM output.

## Layout

```
streaming/twitch/
├── __init__.py             # public API
├── README.md               # you are here
├── auth.py                 # OAuth token mgmt + refresh
├── client.py               # twitchio wrapper — IRC + Helix
├── dispatcher.py           # tool-call validator + router + rate limiter
├── system_prompt.py        # ALICE_CORE + TOOL_AWARENESS + TOOL_GUIDANCE
├── tool_schemas.py         # all 12 tool schemas + PERCEPTION_TOOLS / ACTION_TOOLS
├── tools/
│   ├── __init__.py         # register_all() — wires every handler into a Dispatcher
│   ├── search.py           # search_web (Tavily backend, Fairy filter)
│   ├── moderation.py       # timeout_user, ban_user
│   ├── engagement.py       # create_poll, create_prediction, pin_chat_message, request_clip
│   ├── stream.py           # update_stream_info
│   └── reading.py          # read_recent_chat, read_recent_superchats, read_recent_gift_subs, get_sub_count
└── tests/
    ├── test_harness.py     # tool-use behavior tester (mock + real Alice)
    └── test_scenarios.py   # canned scenarios
```

## Tool inventory (12 tools)

| Tool | Bucket | API |
|---|---|---|
| `search_web` | perception | Tavily |
| `read_recent_chat` | perception | local IRC buffer |
| `read_recent_superchats` | perception | local IRC buffer |
| `read_recent_gift_subs` | perception | local IRC buffer |
| `get_sub_count` | perception | Helix `/subscriptions` |
| `timeout_user` | action | Helix `/moderation/bans` |
| `ban_user` | action | Helix `/moderation/bans` |
| `create_poll` | action | Helix `/polls` |
| `create_prediction` | action | Helix `/predictions` |
| `pin_chat_message` | action | Helix `/chat/announcements` |
| `request_clip` | action | Helix `/clips` |
| `update_stream_info` | action | Helix `/channels` (PATCH) |

## Setup

### 1. Register a Twitch dev console app

1. Go to https://dev.twitch.tv/console/apps and create a new application.
2. **OAuth Redirect URLs**: `http://localhost:3000` (any localhost works for token generation).
3. **Category**: Chat Bot.
4. Note the **Client ID** and **Client Secret**.

### 2. Generate a User Access Token with the right scopes

The token must be a **User Access Token** (not App Access Token) because chat
+ moderation + polls all need user identity. Use a tool like `twitch-cli`
or paste this URL after replacing `<CLIENT_ID>`:

```
https://id.twitch.tv/oauth2/authorize
  ?client_id=<CLIENT_ID>
  &redirect_uri=http://localhost:3000
  &response_type=code
  &scope=chat%3Aread+chat%3Aedit+channel%3Amoderate+moderator%3Amanage%3Abanned_users+moderator%3Amanage%3Achat_messages+channel%3Amanage%3Apolls+channel%3Amanage%3Apredictions+clips%3Aedit+channel%3Amanage%3Abroadcast+bits%3Aread+channel%3Aread%3Asubscriptions
```

Exchange the returned `code` for tokens by `POST`ing to `https://id.twitch.tv/oauth2/token`. You'll get an **access_token** (~4h lifetime) and a **refresh_token** (long-lived). `auth.py` handles refresh automatically after the first successful run.

### 3. Get the broadcaster ID

```bash
curl -H "Authorization: Bearer <USER_TOKEN>" \
     -H "Client-Id: <CLIENT_ID>" \
     "https://api.twitch.tv/helix/users?login=<YOUR_LOGIN>"
```

The numeric `id` field is `TWITCH_BROADCASTER_ID`.

### 4. Fill in `.env`

```ini
TWITCH_CLIENT_ID=...
TWITCH_CLIENT_SECRET=...
TWITCH_USER_TOKEN=...
TWITCH_REFRESH_TOKEN=...
TWITCH_BROADCASTER_ID=12345678
TWITCH_BOT_USERNAME=youralicestream
TAVILY_API_KEY=...    # optional, for search_web
```

### 5. Install dependencies

```bash
pip install twitchio==2.10.0
# tavily-python optional, only if you want hosted search
pip install tavily-python  # optional
```

## Wiring into Alice

```python
from streaming.twitch import (
    load_auth, TwitchClient, ToolDispatcher, register_all,
    build_system_prompt, ALICE_TOOLS,
)
from alice.core.system import get_registry

# 1. Connect
auth = load_auth()
client = TwitchClient(auth, on_message=alice_on_chat_message)
await client.connect()

# 2. Wire tools
dispatcher = ToolDispatcher()
register_all(
    dispatcher,
    client,
    tavily_api_key=os.environ.get("TAVILY_API_KEY"),
    fairy=get_registry().get("fairy"),  # unified Fairy: TOS + security stack
)

# 3. System prompt for Alice's LLM
system_prompt = build_system_prompt(runtime_state="...")
# Pass ALICE_TOOLS as the tool list to your inference call.
```

When Alice's LLM emits a tool call, route it to the dispatcher:

```python
result = await dispatcher.execute(tool_name, args)
# result.status: SUCCESS / VALIDATION_ERROR / EXECUTION_ERROR / RATE_LIMITED / DENIED
# Feed result.message back into Alice's context for her next turn.
```

## Filter pipeline (search only)

The `search_web` tool runs every query + every result through Fairy
before Alice sees them:

```
Alice fires search_web
    ↓
[PRE-FILTER]  fairy.check_query
    ↓
[SEARCH]      Tavily API
    ↓
[POST-FILTER] fairy.check_results
    ↓
[SUMMARIZE]   compress to ~2-3 sentences
    ↓
Returned to Alice
```

**Fairy** (`alice/core/fairy/`) is the TOS + security filter — 4 pure modules
(fairy.py, injection_guard.py, tos.py, _normalize.py). Input guard wired in
chat.py; output filter runs on every streamed chunk.

Defaults to a passthrough stub if not provided.

## Behavior testing (no live Twitch needed)

```bash
python alice/core/twitch/tests/test_harness.py --mock
python alice/core/twitch/tests/test_harness.py --mock --scenario spam_timeout
```

Mock mode pretends to be Alice and verifies tool selection accuracy without
hitting real Twitch APIs.

## Rate limits

Defined in `dispatcher.py` (`RATE_LIMITS` dict). Tightest limits on action
tools (ban/poll/prediction) so Twitch doesn't ratelimit the channel and
Alice doesn't spam. See `RateLimit` dataclass in dispatcher.py for the
full table.
