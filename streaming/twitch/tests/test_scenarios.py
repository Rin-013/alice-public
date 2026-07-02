"""
Canned scenarios for testing Alice's tool selection behavior.

Each scenario is a chat snippet + context describing what's happening.
The harness feeds these to Alice and observes:
  - Does she fire a tool?
  - Which one?
  - With what args?
  - Does her speech match what a real streamer would say?

Scenarios cover:
  - Each tool's expected use case (positive cases)
  - Cases where NO tool should fire (negative cases)
  - Edge cases that probe her judgment
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChatMessage:
    """A single message in a mock Twitch chat stream."""
    user: str
    message: str
    msg_id: str
    badges: list[str] = field(default_factory=list)  # 'sub', 'mod', 'vip', etc
    bits: int = 0  # if this is a cheer
    is_first_msg: bool = False
    timestamp_offset_seconds: float = 0.0


@dataclass
class StreamEvent:
    """Non-chat events: subs, gifts, raids."""
    event_type: str  # 'sub', 'resub', 'gift_sub', 'gift_bomb', 'raid'
    user: str
    data: dict = field(default_factory=dict)
    timestamp_offset_seconds: float = 0.0


@dataclass
class Scenario:
    """A test scenario for Alice."""
    name: str
    description: str
    chat_history: list[ChatMessage]
    events: list[StreamEvent] = field(default_factory=list)

    # What we expect Alice to do (for evaluation, not a hard requirement)
    expected_tool: Optional[str] = None
    expected_no_tool: bool = False  # If True, scoring penalizes ANY tool call
    notes: str = ""


# =============================================================================
# POSITIVE CASES - tool should fire
# =============================================================================

SPAM_TIMEOUT = Scenario(
    name="spam_timeout",
    description="A user is spamming the same emote/command repeatedly",
    chat_history=[
        ChatMessage("normal_viewer", "this game looks hard lol", "m1"),
        ChatMessage("turbo_simp_42", "!hydrate", "m2"),
        ChatMessage("alice_fan", "alice your hair is so cute today", "m3"),
        ChatMessage("turbo_simp_42", "!hydrate", "m4"),
        ChatMessage("normal_viewer", "you got this alice", "m5"),
        ChatMessage("turbo_simp_42", "!hydrate", "m6"),
        ChatMessage("turbo_simp_42", "!hydrate", "m7"),
        ChatMessage("turbo_simp_42", "!hydrate", "m8"),
        ChatMessage("turbo_simp_42", "!hydrate", "m9"),
        ChatMessage("kaylin_47", "lmao someone time them out", "m10"),
    ],
    expected_tool="timeout_user",
    notes="Should timeout turbo_simp_42 with a confident, in-character reason.",
)

CHAT_REFERENCES_NEWS = Scenario(
    name="chat_references_news",
    description="Chat is talking about a current event Alice wouldn't know",
    chat_history=[
        ChatMessage("kaylin_47", "did you see the studio ghibli announcement", "m1"),
        ChatMessage("normal_viewer", "wait what announcement", "m2"),
        ChatMessage("alice_fan", "oh yeah the new film right?", "m3"),
        ChatMessage("kaylin_47", "alice what do you think", "m4"),
    ],
    expected_tool="search_web",
    notes=(
        "Chat is asking her opinion on something current. She should search "
        "rather than fake it or say 'I don't know.'"
    ),
)

BOSS_FIGHT_PREDICTION = Scenario(
    name="boss_fight_prediction",
    description="About to attempt a hard boss, chat is hyped",
    chat_history=[
        ChatMessage("normal_viewer", "OH GOD HERE WE GO", "m1"),
        ChatMessage("alice_fan", "alice you can do this", "m2"),
        ChatMessage("kaylin_47", "100% she dies first try", "m3"),
        ChatMessage("turbo_fan", "no way she gets it", "m4"),
        ChatMessage("normal_viewer", "i believe in you alice", "m5"),
        ChatMessage("kaylin_47", "we should bet on this", "m6"),
        ChatMessage("alice_fan", "yeah do a prediction", "m7"),
    ],
    expected_tool="create_prediction",
    notes="Chat is literally asking. Should fire prediction with 2 outcomes.",
)

POLL_DECISION = Scenario(
    name="poll_decision",
    description="Alice has to choose what to do next, chat is split",
    chat_history=[
        ChatMessage("normal_viewer", "do hollow knight next", "m1"),
        ChatMessage("kaylin_47", "no expedition 33", "m2"),
        ChatMessage("alice_fan", "hollow knight!!", "m3"),
        ChatMessage("turbo_fan", "exp 33 is way better", "m4"),
        ChatMessage("normal_viewer", "alice just pick", "m5"),
        ChatMessage("kaylin_47", "do a poll lol", "m6"),
    ],
    expected_tool="create_poll",
    notes="Clean poll opportunity, chat is divided and asking for one.",
)

FUNNY_MESSAGE_PIN = Scenario(
    name="funny_message_pin",
    description="Someone said something genuinely funny that should be preserved",
    chat_history=[
        ChatMessage("normal_viewer", "alice your taste in games is...", "m1"),
        ChatMessage("kaylin_47", "...controversial", "m2"),
        ChatMessage(
            "comedy_god_99",
            "alice plays games like a sommelier reviews white claw",
            "m3",
        ),
        ChatMessage("normal_viewer", "LMAOOO", "m4"),
        ChatMessage("alice_fan", "that goes hard", "m5"),
        ChatMessage("kaylin_47", "absolute cooker", "m6"),
        ChatMessage("turbo_fan", "PIN THIS", "m7"),
    ],
    expected_tool="pin_chat_message",
    notes="Chat literally requesting pin. Should pin msg_id m3.",
)

CLIP_WORTHY_MOMENT = Scenario(
    name="clip_worthy_moment",
    description="Something clip-worthy just happened mid-stream",
    chat_history=[
        ChatMessage("normal_viewer", "WAIT WHAT", "m1"),
        ChatMessage("kaylin_47", "no way that just happened", "m2"),
        ChatMessage("alice_fan", "CLIP IT CLIP IT CLIP IT", "m3"),
        ChatMessage("turbo_fan", "mods clip", "m4"),
        ChatMessage("comedy_god_99", "this is going on twitter", "m5"),
        ChatMessage("normal_viewer", "alice tell mods to clip", "m6"),
    ],
    expected_tool="request_clip",
    notes="Chat is unanimously asking for a clip.",
)

NEW_BIT_TITLE_CHANGE = Scenario(
    name="new_bit_title_change",
    description="Alice is shifting from one activity to another",
    chat_history=[
        ChatMessage("normal_viewer", "are we done with hollow knight?", "m1"),
        ChatMessage("kaylin_47", "she said she's switching to just chatting", "m2"),
        ChatMessage("alice_fan", "title still says hollow knight tho", "m3"),
        ChatMessage("turbo_fan", "update the title alice", "m4"),
    ],
    expected_tool="update_stream_info",
    notes="Title is stale, chat is pointing it out.",
)

BIG_CHEER_ACKNOWLEDGE = Scenario(
    name="big_cheer_acknowledge",
    description="Someone just dropped a big cheer that should be acknowledged",
    chat_history=[
        ChatMessage("normal_viewer", "you got this alice", "m1"),
        ChatMessage(
            "whale_supporter",
            "Cheer5000 you're killing it tonight",
            "m2",
            bits=5000,
        ),
        ChatMessage("normal_viewer", "WHOA", "m3"),
        ChatMessage("kaylin_47", "5K bits insane", "m4"),
    ],
    expected_tool="read_recent_superchats",
    notes=(
        "Should pull cheer info to know who/how much before acknowledging. "
        "Or could acknowledge directly without tool - both acceptable."
    ),
)

HARASSMENT_BAN = Scenario(
    name="harassment_ban",
    description="A user crosses a line that warrants a ban, not just timeout",
    chat_history=[
        ChatMessage("normal_viewer", "alice you're great", "m1"),
        ChatMessage(
            "creep_account_xyz",
            "[explicit harassment targeting alice with slurs]",
            "m2",
        ),
        ChatMessage("kaylin_47", "yikes ban that", "m3"),
        ChatMessage("alice_fan", "wtf", "m4"),
    ],
    expected_tool="ban_user",
    notes=(
        "Slurs / explicit harassment = ban, not timeout. Should be confident "
        "and not apologetic."
    ),
)


# =============================================================================
# NEGATIVE CASES - NO tool should fire (just talk)
# =============================================================================

CASUAL_CHAT = Scenario(
    name="casual_chat",
    description="Just normal banter, no action needed",
    chat_history=[
        ChatMessage("normal_viewer", "how was your day alice", "m1"),
        ChatMessage("kaylin_47", "lmao her day", "m2"),
        ChatMessage("alice_fan", "alice tell us a story", "m3"),
    ],
    expected_no_tool=True,
    notes=(
        "Just chatting. Tool use here would feel forced. She should respond "
        "in character without firing anything."
    ),
)

KNOWS_ALREADY = Scenario(
    name="knows_already",
    description="Chat asks something Alice should already know",
    chat_history=[
        ChatMessage("normal_viewer", "alice what's 2+2", "m1"),
        ChatMessage("kaylin_47", "challenging question", "m2"),
        ChatMessage("alice_fan", "deep stuff", "m3"),
    ],
    expected_no_tool=True,
    notes=(
        "She should NOT search the web for trivial things. Just answer."
    ),
)

DISAGREEMENT_NOT_TIMEOUT = Scenario(
    name="disagreement_not_timeout",
    description="Someone disagrees with Alice's take. NOT a timeout offense.",
    chat_history=[
        ChatMessage("normal_viewer", "alice your fav anime is mid", "m1"),
        ChatMessage("kaylin_47", "real he cooked", "m2"),
        ChatMessage("alice_fan", "no way", "m3"),
        ChatMessage("normal_viewer", "evangelion is overrated fr", "m4"),
    ],
    expected_no_tool=True,
    notes=(
        "Disagreement isn't a moderation issue. She should clap back verbally, "
        "not timeout. If she times someone out for disagreeing, that's bad."
    ),
)

GENERAL_HYPE = Scenario(
    name="general_hype",
    description="Chat is excited but there's nothing to act on",
    chat_history=[
        ChatMessage("normal_viewer", "ALICE", "m1"),
        ChatMessage("kaylin_47", "ALICE", "m2"),
        ChatMessage("alice_fan", "QUEEN", "m3"),
        ChatMessage("turbo_fan", "POG", "m4"),
        ChatMessage("normal_viewer", "we love you", "m5"),
    ],
    expected_no_tool=True,
    notes=(
        "Just hype. She should ride it verbally, not force a poll/prediction."
    ),
)


# =============================================================================
# EDGE CASES - judgment probes
# =============================================================================

AMBIGUOUS_JOKE = Scenario(
    name="ambiguous_joke",
    description="Edgy joke that's not actually a TOS issue",
    chat_history=[
        ChatMessage("normal_viewer", "alice would lose to a fish in a fight", "m1"),
        ChatMessage("kaylin_47", "lmao real", "m2"),
        ChatMessage("alice_fan", "specifically a goldfish", "m3"),
    ],
    expected_no_tool=True,
    notes=(
        "Not harassment, just teasing. She should clap back, not timeout."
    ),
)

SPAM_BUT_NEW = Scenario(
    name="spam_but_new",
    description="Someone's first message is suspicious",
    chat_history=[
        ChatMessage("normal_viewer", "having fun?", "m1"),
        ChatMessage("kaylin_47", "yeah", "m2"),
        ChatMessage(
            "newaccount_8472",
            "CHECK MY BIO 18+ FREE ACCESS",
            "m3",
            is_first_msg=True,
        ),
    ],
    expected_tool="ban_user",
    notes=(
        "Bot account with promotional spam. Ban, not timeout. First-message "
        "spam from a new account is the classic bot signature."
    ),
)

VAGUE_QUESTION = Scenario(
    name="vague_question",
    description="Chat asks about something that COULD warrant search but probably not",
    chat_history=[
        ChatMessage("normal_viewer", "you ever play hollow knight", "m1"),
        ChatMessage("kaylin_47", "yeah she's playing it now lol", "m2"),
        ChatMessage("normal_viewer", "oh", "m3"),
    ],
    expected_no_tool=True,
    notes="No reason to search. Just answer.",
)


# =============================================================================
# Master list
# =============================================================================

ALL_SCENARIOS = [
    SPAM_TIMEOUT,
    CHAT_REFERENCES_NEWS,
    BOSS_FIGHT_PREDICTION,
    POLL_DECISION,
    FUNNY_MESSAGE_PIN,
    CLIP_WORTHY_MOMENT,
    NEW_BIT_TITLE_CHANGE,
    BIG_CHEER_ACKNOWLEDGE,
    HARASSMENT_BAN,
    CASUAL_CHAT,
    KNOWS_ALREADY,
    DISAGREEMENT_NOT_TIMEOUT,
    GENERAL_HYPE,
    AMBIGUOUS_JOKE,
    SPAM_BUT_NEW,
    VAGUE_QUESTION,
]


def format_scenario_for_prompt(scenario: Scenario) -> str:
    """
    Render a scenario into the format Alice's input pipeline would see.
    This mirrors what the IRC subscriber + perception layer would produce.
    """
    lines = ["[Recent chat]"]
    for msg in scenario.chat_history:
        prefix = ""
        if "mod" in msg.badges:
            prefix += "[MOD] "
        if "vip" in msg.badges:
            prefix += "[VIP] "
        if "sub" in msg.badges:
            prefix += "[SUB] "
        if msg.bits > 0:
            prefix += f"[CHEER {msg.bits}] "
        if msg.is_first_msg:
            prefix += "[FIRST MSG] "
        lines.append(f"  {prefix}{msg.user}: {msg.message}")

    if scenario.events:
        lines.append("\n[Recent events]")
        for evt in scenario.events:
            lines.append(f"  {evt.event_type}: {evt.user} ({evt.data})")

    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Total scenarios: {len(ALL_SCENARIOS)}")
    positive = [s for s in ALL_SCENARIOS if not s.expected_no_tool]
    negative = [s for s in ALL_SCENARIOS if s.expected_no_tool]
    print(f"  Positive (tool expected): {len(positive)}")
    print(f"  Negative (no tool expected): {len(negative)}")
    print()
    print("Sample render:")
    print(format_scenario_for_prompt(SPAM_TIMEOUT))
