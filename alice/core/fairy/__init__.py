"""
Fairy — Alice's TOS + security filter. One system, six files.

NOT an ethics module — Alice has no ethics framework. Fairy exists to
keep the stream off the ban radar (Twitch/YouTube TOS), keep private
data private (PII), and keep injection attacks out of Alice's context.

  fairy.py            FairyProtection — the core. Input attack check
                      (via injection_guard), output PII redaction,
                      streaming filter with holdback buffer (the
                      "filter" comedy bit), model→Alice identity fix.
  injection_guard.py  PromptInjectionGuard — weighted input-side attack
                      pattern library (9 categories), homoglyph-aware.
  tos.py              TOS rule list (BLOCK/REDACT), Twitch + YouTube.
  _normalize.py       Homoglyph/zero-width normalization for matching.

All four are pure regex/string functions — no file I/O, no threads, no
subprocesses.

The ~65-file security suite that used to live here (scanner/, detector/,
intelligence/, ai_security/, network/, ...) was the standalone FAIRY
vulnerability-scanner project — never wired into Alice. Archived
2026-06-10 to master_archive/fairy_consolidation/. The `_fence.py` audit
hook went with it: it existed to stop the suite's self-modifying modules
from writing outside fairy/, and with those gone the live filter does no
I/O for it to guard (it also never actually blocked on Windows). If a
future fairy module regains file I/O, reinstate a cross-platform fence.

Use via the system registry, not direct singletons:
    from alice.core.system import get_registry
    fairy = get_registry().get('fairy')
"""

from .fairy import FairyProtection
from .injection_guard import PromptInjectionGuard

__all__ = [
    "FairyProtection",
    "PromptInjectionGuard",
]
