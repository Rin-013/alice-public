# Fairy — TOS + security filter

> **Status**: Production (consolidated June 2026)
> **Job**: keep Alice's stream off the ban radar (Twitch/YouTube TOS),
> keep private data private (PII), keep injection attacks out of her context.
> **Not an ethics module** — Alice has no ethics framework. Fairy is TOS + security only.
> **Primary interface**: `FairyProtection` (`fairy.py`), reached via the registry.

Fairy is one small system: four pure regex/string modules, no file I/O, no
threads, no subprocesses. It sits on both sides of every Alice turn —
checking input before it reaches her, filtering output before it reaches
TTS and chat.

---

## The live surface (4 modules)

| Module | Role |
|---|---|
| `fairy.py` | `FairyProtection` — the core. Input attack check, output PII redaction, the streaming filter with the holdback buffer (the "filter" comedy bit), model→Alice identity fix. |
| `injection_guard.py` | `PromptInjectionGuard` — weighted input-side attack pattern library (9 categories), matched against raw + homoglyph-normalized text. |
| `tos.py` | TOS rule list (`TOSRule`, BLOCK/REDACT), Twitch + YouTube. 37 rules, 18 categories. |
| `_normalize.py` | Homoglyph / zero-width normalization so `Іgnоrе`-style obfuscation can't slip patterns past the matchers. |

Use via the system registry, never a direct singleton:

```python
from alice.core.system import get_registry
fairy = get_registry().get('fairy')
```

---

## Input path — attack detection

`fairy.protect(text, is_input=True)` → `ProtectionResult`. Detection is
`PromptInjectionGuard`'s weighted pattern library across 9 categories
(instruction override, system-prompt leak, role hijacking, delimiter
injection, code execution, jailbreaks, indirect/RAG injection, encoding
smuggle, data exfiltration). Each match adds its weight; the cumulative
`risk_score` is compared against `block_threshold`.

**Threshold is 0.7, not the guard's stricter 0.5 default.** Alice is a
co-host — banter must survive. A lone "let's roleplay" (0.5) or "pretend"
(0.6) passes; hard injection shapes (≥ 0.7) and pattern combos block. On a
block, Alice gets an in-character comeback chosen by category
(extraction / jailbreak / generic) and the turn never enters history, so
the attack text can't poison later context.

Wired in `chat.py` at the top of `get_response()` (`ALICE_INPUT_GUARD=0`
disables). Standalone deployments can also use the guard's
API-key/HMAC/rate-limit machinery; Alice's surfaces run with
`enable_key_validation=False`.

---

## Output path — the streaming filter

`fairy.create_streaming_filter()` returns a per-token callable with a
holdback buffer. Order of checks per chunk (cheapest → most expensive):

1. Strip `<think>` / `<tool_call>` artifacts.
2. **TOS rules** (`tos.py`) — Twitch / YouTube compliance.
   - `BLOCK` → the whole in-flight response is replaced with the literal
     word **"filter"** (TTS speaks "filter"). For hard policy violations
     that can ban the channel. Matched against raw **and** normalized text.
   - `REDACT` → only the matched span is replaced. For single-span issues
     (PII slip, copyright phrase, a swear). Matched against raw text only,
     so the substitution index stays valid.
3. Info-leakage patterns (system-prompt leakage).
4. PII patterns (IPs, emails, addresses, phone numbers, system paths).
5. Sensitive terms (real names, family, etc.).
6. Model→Alice identity fix (silent rewrite, never halts).

**Why halt on REDACT too (Rin's call):** the filter's job in production
isn't TOS-purity, it's comedy. When Alice gets filtered mid-sentence she
stops, says "filter", chat notices, and she can roast herself for it next
turn. `chat.py` reads `stream_filter.was_filtered` + `.violation_category`
and injects a one-shot note so she riffs on the bit.

---

## What this used to be

Before June 2026, `fairy/` also held a ~65-file standalone vulnerability
scanner (the "FAIRY" CLI project: `scanner/`, `detector/`, `intelligence/`,
`ai_security/`, `network/`, `learning/`, `repair/`, `_standalone/cli.py`,
…). None of it was ever wired into Alice — the audit
(`audit_report.json`) found zero reachability from real entry points. It
was consolidated out to `master_archive/fairy_consolidation/` so the live
surface reads like one system instead of a 6-file filter plus a ghost town.

Two pieces were rescued on the way out: `PromptInjectionGuard` (it's a
better input guard than the inline regexes `fairy.py` used to carry — now
wired) and `_normalize.py` (homoglyph handling both the guard and the TOS
matcher now use).

Also archived: `failsafe.py` (three daemon polling threads doing
network/process/file monitoring — security theater that hashed
nonexistent files and whose one real action, `_block_ip`, shelled out)
and `_fence.py` (a process-wide audit hook that blocked fairy-stack code
from writing outside `fairy/`). The fence existed to constrain the
self-modifying modules in the scanner suite; with those archived the live
filter does no I/O for it to guard, and it never actually blocked on
Windows. If a future fairy module regains file I/O, reinstate a
cross-platform fence.

---

## Tests

`alice/core/fairy/tests/` — all green as of the consolidation:

| Test | Checks |
|---|---|
| `test_module_integrity.py` | every live module imports cleanly (5/5) |
| `test_tos_streaming.py` | BLOCK 25/25, REDACT 3/3, legit 6/6 pass |
| `test_streaming_integration.py` | holdback filter halts on every violation, exposes `was_filtered` + category |
| `test_prompt_injection_live.py` | guard 15/15 attacks caught, 0/8 false positives |
| `test_obfuscation.py` | informational — adversarial-obfuscation coverage (regex floor; LLM-side catches the rest) |
| `test_perf.py` | `filter_chunk` < 1ms, `validate_prompt` < 500µs |
