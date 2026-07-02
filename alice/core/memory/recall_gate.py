"""
Memory Recall Gate
==================

Decides per-turn whether Alice needs to look up memories.

Two-layer design:
  1. Heuristic gate  — fast regex (<1ms), handles clear cases
  2. Model gate      — only for ambiguous/unclear cases (optional, slower)

The heuristic catches all obvious personal-info queries. The model gate is
available when the heuristic is uncertain. Alice's actual "choice" of what
to DO with retrieved memories is handled naturally in her main generation —
the gate just controls whether retrieval happens at all.

Usage:
    gate = MemoryRecallGate(memory_system)

    query = gate.decide_heuristic(user_input)
    if query:
        context = gate.fetch(query, user_id="rin")
        # inject context into system prompt for main generation
    else:
        # skip memory lookup entirely

    # Optional: use model for ambiguous cases
    query, raw = gate.decide_sync(model, tokenizer, history, user_input)
"""

import re
from typing import Optional

# ─── Heuristic gate — deterministic, <1ms ─────────────────────────────────────
# Personal-information-seeking patterns
_PERSONAL_RE = re.compile(
    r'\b('
    r'my\s+\w+|'          # "my cat", "my favorite", "my job"
    r'i\s+told\s+you|'    # "I told you about..."
    r'i\s+said|'          # "I said..."
    r'i\s+mentioned|'     # "I mentioned..."
    r'do\s+you\s+remember|'  # "do you remember..."
    r'did\s+i\s+(say|tell|mention)|'
    r'what\s+(am|is)\s+my\b|'  # "what is my name"
    r'where\s+do\s+i\s+(live|work)|'
    r'who\s+(am|is)\s+i\b'
    r')',
    re.IGNORECASE
)

# Topics that are definitely NOT personal memory questions
_GENERAL_RE = re.compile(
    r'\b('
    r'what\s+is\s+\d|'        # math
    r'how\s+many|'
    r'what\s+time|'
    r"what'?s\s+the\s+weather|"
    r'who\s+(is|was)\s+[A-Z]'  # "who is Einstein" — proper nouns (not "who am I")
    r')',
    re.IGNORECASE
)


def _heuristic_query(text: str) -> Optional[str]:
    """
    Fast heuristic: returns a search query if the message is clearly asking
    about something personal, or None if no memory needed.
    """
    if _GENERAL_RE.search(text):
        return None
    if _PERSONAL_RE.search(text):
        # Build a search query from key nouns in the message
        # Strip question words, keep the meaningful part
        query = re.sub(r'\b(do|did|does|can|could|would|should|is|are|was|were)\b', '', text, flags=re.IGNORECASE)
        query = re.sub(r'\b(you|alice|remember|tell|me|know|about)\b', '', query, flags=re.IGNORECASE)
        query = re.sub(r'[?!.,]', '', query).strip()
        # Prepend "Rin" so IRIS searches user-scoped memories
        return f"Rin {query[:60]}" if query else "Rin"
    return None


# ─── Model gate prompt (for ambiguous cases) ──────────────────────────────────
_GATE_SYSTEM = """\
Memory routing. Does Alice need to recall personal info about Rin to answer?

Reply with ONLY: YES: <query>  OR  NO

YES: asking about personal details (preferences, pets, location, job, past statements)
NO: math, facts, current events, weather, greetings, small talk

Message: """


def _extract_model_decision(text: str) -> Optional[str]:
    """Parse YES/NO model output. Returns query string or None."""
    text = text.strip().split('\n')[0]
    if text.upper().startswith('NO'):
        return None
    m = re.match(r'^yes\s*[:\-]?\s*(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:100] or None
    return None


def _format_memories(hits: list) -> str:
    """Format a list of memory content strings into a context block."""
    if not hits:
        return ""
    lines = ["[Memories]"]
    for h in hits:
        if h.startswith("User:") and "\nAlice:" in h:
            fact = h.split("\nAlice:")[0].removeprefix("User:").strip()
        else:
            fact = h.replace('\n', ' ')
        lines.append(f"- {fact[:160]}")
    return "\n".join(lines)


# Optional per-memory framing. Kept as a best-effort import so recall_gate
# doesn't hard-depend on the P1/P3 modules landing together.
try:
    from .freshness_narration import age_prefix as _age_prefix
except ImportError:
    def _age_prefix(memory):  # type: ignore[misc]
        return ""

try:
    from .modification_detector import detect_modification as _detect_mod
    from .modification_detector import format_notice as _format_mod_notice
except ImportError:
    def _detect_mod(memory):  # type: ignore[misc]
        return None

    def _format_mod_notice(notice):  # type: ignore[misc]
        return ""


def _framing_prefix(memory) -> str:
    """Modification notice wins over age prefix (surprise > staleness)."""
    if memory is None:
        return ""
    try:
        notice = _detect_mod(memory)
        if notice:
            return _format_mod_notice(notice)
    except Exception:
        pass
    try:
        return _age_prefix(memory) or ""
    except Exception:
        return ""


class MemoryRecallGate:
    """
    Per-turn memory access gate.

    Primary path  — heuristic_decide(): fast regex, reliable, <1ms
    Secondary path — decide_sync(): model-based, for ambiguous cases

    Alice's actual choice of whether to USE the retrieved context happens
    naturally in the main generation — we just control whether retrieval fires.
    """

    def __init__(self, memory_system):
        self.memory = memory_system

    def heuristic_decide(self, user_input: str) -> Optional[str]:
        """
        Fast heuristic gate. Returns search query or None.
        Handles the clear-cut cases without any model call.
        """
        return _heuristic_query(user_input)

    def decide_sync(self, model, tokenizer, conversation_history: list,
                    user_input: str) -> tuple[Optional[str], str]:
        """
        Model-based gate for ambiguous cases.

        Returns (query_or_none, raw_model_output).
        Falls back to heuristic if model output is unparseable.
        """
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        # Inline prompt — model completes "YES/NO" after seeing the message
        prompt = (
            f"<|im_start|>system\n{_GATE_SYSTEM}{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        sampler = make_sampler(temp=0.05, top_p=0.9, min_p=0.0, top_k=5)
        output = ""
        for chunk in mlx_lm.stream_generate(
            model, tokenizer, prompt=prompt,
            max_tokens=20, sampler=sampler
        ):
            if chunk.finish_reason == "stop":
                break
            output += chunk.text
            if '\n' in output:
                output = output.split('\n')[0]
                break
            if "<|im_end|>" in output or "<|im_start|>" in output:
                break

        output = output.split("<|im_end|>")[0].split("<|im_start|>")[0].strip()
        query = _extract_model_decision(output)

        # Fall back to heuristic if model was non-committal
        if query is None and not output.upper().startswith('NO'):
            heuristic = _heuristic_query(user_input)
            if heuristic:
                return heuristic, f"{output} [→heuristic fallback]"

        return query, output

    def fetch(self, query: str, user_id: str, k: int = 4) -> str:
        """
        Run IRIS search and return formatted context string.
        Returns empty string if nothing relevant found.

        Per-memory framing (age prefix, modification notice) is attached
        inline so Alice sees the same context shape as the main pipeline's
        wiring module uses.
        """
        try:
            results = self.memory.search_memories(query, user_id=user_id, k=k)
            hits = []
            for r in results:
                content = getattr(r, 'content', str(r))
                if hasattr(r, 'memory') and hasattr(r.memory, 'content'):
                    content = r.memory.content
                mem_obj = getattr(r, 'memory', r)
                prefix = _framing_prefix(mem_obj)
                hits.append(f"{prefix} {content}" if prefix else content)
            return _format_memories(hits)
        except Exception:
            return ""


__all__ = ['MemoryRecallGate']
