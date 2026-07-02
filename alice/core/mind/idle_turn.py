"""
Idle turn — what Alice does when nobody's prompting her.

When the chat loop has been silent for `ALICE_IDLE_TURN_SECONDS` (default 60s),
chat.py asks this module for a self-prompt to drive the next turn. The result
runs through `get_response()` like a normal turn — Alice doesn't know it's
"autonomous", just sees an unusual user-input string.

Three modes (rotating to avoid repetition):
  - "proposal":   pick the freshest proposal from Mind's buffer, frame it
                  as Alice reacting to her own thought
  - "react_chat": grab the last few lines of Twitch chat, frame them as
                  scrollback she just glanced at
  - "yap":        no specific stimulus, just nudge her to fill the silence

Returns (mode, prompt_string) or None if no autonomous turn is appropriate
(e.g., everything is empty and no twitch).
"""
from __future__ import annotations

import logging
import os
import random
from typing import Optional, Tuple

logger = logging.getLogger("alice.mind.idle_turn")


def _last_used_path() -> str:
    """Per-process state to avoid same mode twice in a row. Lives in /tmp."""
    return os.path.join(os.environ.get("TMPDIR", "/tmp"), f"alice_idle_mode_{os.getpid()}")


def _read_last_mode() -> Optional[str]:
    try:
        with open(_last_used_path()) as f:
            return f.read().strip()
    except OSError:
        return None


def _write_last_mode(mode: str) -> None:
    try:
        with open(_last_used_path(), "w") as f:
            f.write(mode)
    except OSError:
        pass


def next_autonomous_input(
    mind=None,
    twitch_runtime=None,
) -> Optional[Tuple[str, str]]:
    """
    Pick the next autonomous turn input.

    Returns (mode, prompt) or None if no mode is available.

    The returned prompt is what gets passed to `get_response()` as if it
    were Rin's input. It uses bracketed framing the LLM understands
    as a stage direction ("[autonomous: …]") rather than a real user line.
    """
    last_mode = _read_last_mode()
    candidates: list[tuple[str, callable]] = []

    if mind is not None and getattr(mind, "proposals", None) is not None:
        recent = mind.proposals.get_recent(n=3)
        if recent:
            candidates.append(("proposal", lambda: _proposal_prompt(recent)))

    if twitch_runtime is not None and twitch_runtime.connected:
        try:
            recent_chat = twitch_runtime.client.recent_chat(count=5, since_seconds=300)
        except Exception:
            recent_chat = []
        if recent_chat:
            candidates.append(("react_chat", lambda: _react_chat_prompt(recent_chat)))

    # Yap is always available as a fallback.
    candidates.append(("yap", _yap_prompt))

    # Avoid the last mode if we have alternatives.
    if last_mode and len(candidates) > 1:
        filtered = [(m, fn) for m, fn in candidates if m != last_mode]
        if filtered:
            candidates = filtered

    mode, prompt_fn = random.choice(candidates)
    try:
        prompt = prompt_fn()
    except Exception as e:
        logger.warning(f"idle prompt builder failed for mode={mode}: {e}")
        return None
    _write_last_mode(mode)
    return mode, prompt


def _proposal_prompt(recent_proposals) -> str:
    """Frame Alice's freshest proposal as something she's about to say out loud."""
    proposal = recent_proposals[-1]
    content = getattr(proposal, "content", str(proposal))
    return f"[autonomous: voice your own thought]\n{content}"


def _react_chat_prompt(recent_chat) -> str:
    """Frame recent chat as scrollback Alice just noticed."""
    lines = []
    for cm in recent_chat[-5:]:
        text = (cm.text or "").strip()
        if not text:
            continue
        lines.append(f"  {cm.display_name or cm.username}: {text}")
    if not lines:
        return _yap_prompt()
    chat_block = "\n".join(lines)
    return (
        "[autonomous: react to the chat scrollback you just glanced at]\n"
        f"{chat_block}\n\n"
        "Pick whatever in there is worth a reaction. Don't address every line."
    )


def _yap_prompt() -> str:
    """No specific stimulus — Alice fills the silence."""
    nudges = [
        "It's been quiet. Say something.",
        "The stream's been silent for a bit. Fill the space.",
        "No one's talking. What's on your mind?",
        "Dead air. Riff on whatever you want.",
        "Nobody's prompting you. Be a streamer — just yap.",
    ]
    return f"[autonomous: fill the silence]\n{random.choice(nudges)}"


__all__ = ["next_autonomous_input"]
