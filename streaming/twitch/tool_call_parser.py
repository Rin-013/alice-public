"""
Streaming tool-call parser.

Alice's LLM streams plain text. When she wants to fire a tool, she emits:

    <tool_call>
    <function=timeout_user>
    <parameter=username>turbo_simp_42</parameter>
    <parameter=duration_seconds>300</parameter>
    <parameter=reason>spamming</parameter>
    </function>
    </tool_call>

This parser sits between the LLM streamer and the TTS/console pipeline:
  - Text outside <tool_call> tags is passed through as visible output.
  - Text inside the tags is suppressed (never spoken or printed) and parsed
    when the closing tag arrives.
  - Tags can split across token boundaries; the parser buffers partials.

The parser is stateful and not thread-safe — instantiate one per response stream.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


@dataclass
class ToolCall:
    """A successfully-extracted (or attempted) tool call."""
    tool: str
    args: dict
    raw_json: str
    parse_error: Optional[str] = None  # set if JSON was malformed


class StreamingToolCallParser:
    """
    Stateful streaming parser. Feed token chunks; receive (visible_text, completed_calls).

    Usage:
        parser = StreamingToolCallParser()
        for chunk in stream:
            visible, calls = parser.feed(chunk)
            if visible: emit_to_tts(visible)
            for call in calls: queue_dispatch(call)
        tail, _ = parser.flush()
        if tail: emit_to_tts(tail)
    """

    def __init__(self):
        self._buf = ""
        self._inside = False  # True between OPEN_TAG and CLOSE_TAG

    def feed(self, chunk: str) -> Tuple[str, List[ToolCall]]:
        self._buf += chunk
        visible_parts: list[str] = []
        completed: list[ToolCall] = []

        while True:
            if not self._inside:
                # Look for an opening tag in the buffer.
                idx = self._buf.find(OPEN_TAG)
                if idx >= 0:
                    visible_parts.append(self._buf[:idx])
                    self._buf = self._buf[idx + len(OPEN_TAG):]
                    self._inside = True
                    continue
                # No full open tag — but the tail might be a *partial* "<tool_"
                # that we shouldn't emit yet (in case the next token completes it).
                hold = _partial_tail_len(self._buf, OPEN_TAG)
                if hold:
                    visible_parts.append(self._buf[:-hold])
                    self._buf = self._buf[-hold:]
                else:
                    visible_parts.append(self._buf)
                    self._buf = ""
                break

            # Inside a tool call. Look for the closing tag.
            idx = self._buf.find(CLOSE_TAG)
            if idx >= 0:
                raw = self._buf[:idx]
                self._buf = self._buf[idx + len(CLOSE_TAG):]
                self._inside = False
                completed.append(_parse_tool_call(raw))
                continue
            # No full close tag yet — keep buffering. (Don't emit anything;
            # everything inside the tags is suppressed from output.)
            break

        return ("".join(visible_parts), completed)

    def flush(self) -> Tuple[str, List[ToolCall]]:
        """
        Call when the stream has ended. Returns any held visible tail and
        any incomplete tool call (as a parse_error record so Alice gets
        feedback that her tool call was cut off).
        """
        completed: list[ToolCall] = []
        if self._inside:
            completed.append(ToolCall(
                tool="",
                args={},
                raw_json=self._buf,
                parse_error="unclosed <tool_call> tag — stream ended before </tool_call>",
            ))
            self._buf = ""
            self._inside = False
            return ("", completed)

        # Outside a tag — flush any held partial as visible text. If it
        # was actually the start of a tag, we'd rather speak it than lose
        # it; an unclosed open tag mid-word is the LLM's bug, not ours.
        tail = self._buf
        self._buf = ""
        return (tail, completed)


def _partial_tail_len(buf: str, target: str) -> int:
    """
    Return how many trailing chars of `buf` could be the start of `target`.
    e.g. buf='hello <tool_', target='<tool_call>' -> 6 (the '<tool_' tail).
    Used to hold back text that might complete into a tag on the next chunk.
    """
    max_check = min(len(buf), len(target) - 1)
    for n in range(max_check, 0, -1):
        if buf.endswith(target[:n]):
            return n
    return 0


def _parse_tool_call(raw: str) -> ToolCall:
    """Parse <function=name><parameter=key>value</parameter></function> format."""
    body = raw.strip()

    func_match = re.match(r'<function=([^>]+)>', body)
    if not func_match:
        return ToolCall(tool="", args={}, raw_json=body, parse_error="missing <function=name> tag")

    tool_name = func_match.group(1).strip()
    if not tool_name:
        return ToolCall(tool="", args={}, raw_json=body, parse_error="empty function name")

    args: dict = {}
    for param_match in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.DOTALL):
        key = param_match.group(1).strip()
        value = param_match.group(2).strip()
        try:
            args[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            args[key] = value

    return ToolCall(tool=tool_name, args=args, raw_json=body)


__all__ = ["StreamingToolCallParser", "ToolCall"]
