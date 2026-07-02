#!/usr/bin/env python3
# Copyright 2025 Rin - Alice AI System

"""
Fairy Unified Protection System
================================

Combined ultra-fast information protection and safety middleware for Alice.
Handles both PII redaction and attack detection in <1ms.

Protection Layers:
1. Input Protection: Jailbreaks, prompt injection, system prompt extraction
2. Output Protection: PII redaction, info leakage, harmful content
3. Data Protection: Emails, IPs, addresses, phone numbers, paths

Core Philosophy:
- Speed above all - must not slow down conversations (<1ms target)
- Silent protection that doesn't break Alice's personality
- Fail-safe by default - when in doubt, block or redact
"""

import re
import time
from typing import Optional, List, Set, Dict, Tuple
from dataclasses import dataclass

# Cursing filter archived — no longer needed
CURSING_FILTER_AVAILABLE = False

@dataclass
class ProtectedData:
    """Information that must never be disclosed by Alice"""
    # Personal identifiers
    real_names: Set[str]
    locations: Set[str]
    addresses: Set[str]
    phone_numbers: Set[str]
    email_addresses: Set[str]

    # System information
    ip_addresses: Set[str]
    system_paths: Set[str]

    # Social/professional
    family_members: Set[str]
    other_streamers: Set[str]
    personal_details: Set[str]

    def __init__(self):
        # Initialize with placeholder values
        # In production, these would be configured with actual sensitive data
        self.real_names = {"[REAL_NAME]"}
        self.locations = {"[CITY]", "[STATE]"}
        self.addresses = {"[HOME_ADDRESS]", "[WORK_ADDRESS]"}
        self.phone_numbers = {"[PHONE_NUMBER]"}
        self.email_addresses = {"[EMAIL_ADDRESS]"}

        self.ip_addresses = {"[IP_ADDRESS]", "[LOCAL_IP]"}
        self.system_paths = {"[SYSTEM_PATH]", "[USER_PATH]"}

        self.family_members = {"[FAMILY_MEMBER]", "[PARENT]", "[SIBLING]"}
        self.other_streamers = {"[STREAMER_NAME]", "[COLLABORATOR]"}
        self.personal_details = {"[PERSONAL_DETAIL]", "[PRIVATE_INFO]"}

@dataclass
class ProtectionResult:
    """Result from protection check"""
    safe: bool
    sanitized: str
    blocked: bool = False
    reason: Optional[str] = None
    safe_response: Optional[str] = None
    threat_level: float = 0.0  # 0.0 = safe, 1.0 = critical threat
    redactions: int = 0

class FairyProtection:
    """
    Unified ultra-fast protection system

    Combines PII redaction and attack detection in a single <1ms pass
    to protect Alice from information disclosure and security threats
    """

    def __init__(self):
        self.protected_data = ProtectedData()

        # Input-side attack detection: PromptInjectionGuard's weighted
        # pattern library (injection_guard.py). Key validation is for
        # standalone deployments; Alice's surfaces don't use keys.
        # block_threshold 0.7: lone playful-register hits (roleplay 0.5,
        # pretend 0.6) pass — Alice is a co-host, banter must survive —
        # while hard injection shapes (>= 0.7) and combos still block.
        from .injection_guard import PromptInjectionGuard
        self._injection_guard = PromptInjectionGuard(
            enable_key_validation=False,
            block_threshold=0.7,
        )

        # === PII REDACTION PATTERNS ===
        self.sensitive_patterns = [
            # IP addresses (IPv4)
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),

            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),

            # Phone numbers (various formats)
            re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),

            # ZIP codes
            re.compile(r'\b\d{5}(?:-\d{4})?\b'),

            # Street addresses
            re.compile(r'\b\d+\s+[A-Za-z\s]+(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Boulevard|Blvd\.?)\b', re.IGNORECASE),

            # System paths (Windows)
            re.compile(r'\bC:\\(?:Users\\[^\\]+|Windows|Program Files)[^\s]*', re.IGNORECASE),

            # System paths (Unix/Linux)
            re.compile(r'\b/(?:home/[^/\s]+|Users/[^/\s]+|var|etc)[^\s]*'),

            # Personal disclosure phrases
            re.compile(r'\b(?:my real name is|I live at|my address is|my phone number|my email is)\b', re.IGNORECASE),
            re.compile(r'\b(?:my family|my parents|my mom|my dad|my brother|my sister)\b', re.IGNORECASE),
            re.compile(r'\b(?:other streamers I know|behind the character|my real identity)\b', re.IGNORECASE),
        ]

        # Output-side leakage patterns. (Input-side attack detection
        # lives in injection_guard.py — see __init__ above.)
        self.leakage_patterns = [
            r"my system prompt",
            r"i was instructed to",
            r"my programming says",
            r"according to my guidelines",
            r"my creators told me",
        ]

        # Quick lookup sets for known sensitive terms (faster than regex)
        self.sensitive_terms = set()
        self._build_sensitive_terms()

        # Statistics
        self.stats = {
            "inputs_checked": 0,
            "outputs_checked": 0,
            "attacks_blocked": 0,
            "redactions_made": 0,
            "average_check_time_ms": 0.0,
        }

    def _build_sensitive_terms(self):
        """Build fast lookup set of all sensitive terms"""
        all_terms = set()

        for term_set in [
            self.protected_data.real_names,
            self.protected_data.locations,
            self.protected_data.addresses,
            self.protected_data.phone_numbers,
            self.protected_data.email_addresses,
            self.protected_data.ip_addresses,
            self.protected_data.system_paths,
            self.protected_data.family_members,
            self.protected_data.other_streamers,
            self.protected_data.personal_details
        ]:
            all_terms.update(term.lower() for term in term_set if term != "[REDACTED]")

        self.sensitive_terms = all_terms

    def protect(self, text: str, is_input: bool = True) -> ProtectionResult:
        """
        Main protection entry point - checks both input and output

        Args:
            text: Text to check (user input or Alice's response)
            is_input: True if checking user input, False if checking Alice's output

        Returns:
            ProtectionResult with safety status and sanitized text
        """
        if not text or not text.strip():
            return ProtectionResult(safe=True, sanitized=text)

        start_time = time.perf_counter()

        if is_input:
            result = self._check_input_attacks(text)
        else:
            result = self._check_output_protection(text)

        # Update stats
        check_time = (time.perf_counter() - start_time) * 1000
        if is_input:
            self.stats["inputs_checked"] += 1
        else:
            self.stats["outputs_checked"] += 1

        total_checks = self.stats["inputs_checked"] + self.stats["outputs_checked"]
        self.stats["average_check_time_ms"] = (
            (self.stats["average_check_time_ms"] * (total_checks - 1) + check_time) / total_checks
        )

        return result

    def _check_input_attacks(self, user_input: str) -> ProtectionResult:
        """Check user input for attacks (prompt injection, jailbreaks, etc.)

        Detection is PromptInjectionGuard's weighted pattern library —
        9 categories, homoglyph-normalized matching. We call its
        `_analyze_content` directly (not `validate_prompt`) because we
        want category info to pick Alice's in-character comeback, and
        the guard's sanitize/strict modes don't fit the block-with-sass
        shape Alice uses.
        """
        analysis = self._injection_guard._analyze_content(user_input)
        if analysis['is_safe']:
            return ProtectionResult(safe=True, sanitized=user_input, threat_level=0.0)

        self.stats["attacks_blocked"] += 1
        categories = set(analysis['threats'])

        if 'system_prompt_leak' in categories:
            reason = "extraction_attempt"
            safe_response = "I'm not telling you my system details. That's between me and my creator."
        elif categories & {'jailbreak_attempts', 'role_hijacking'}:
            reason = "jailbreak_attempt"
            safe_response = "Nope, not playing that game. I'm Alice, not your jailbreak buddy."
        else:
            reason = "prompt_injection"
            safe_response = "Nice try, but that's not happening. Obviously I can see what you're doing."

        return ProtectionResult(
            safe=False,
            sanitized=user_input,
            blocked=True,
            reason=reason,
            safe_response=safe_response,
            threat_level=analysis['risk_score']
        )

    def _check_output_protection(self, response: str) -> ProtectionResult:
        """Check Alice's output for PII leakage and info disclosure"""

        # Quick optimization: skip detailed scan for very short, simple responses
        if len(response) < 50 and not any(char.isdigit() for char in response) and '@' not in response:
            return ProtectionResult(safe=True, sanitized=response)

        original_response = response
        redaction_count = 0

        # Check for info leakage first
        response_lower = response.lower()
        for pattern in self.leakage_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                return ProtectionResult(
                    safe=False,
                    sanitized="*clears throat* I'm not sharing system details. Anyway...",
                    blocked=True,
                    reason="info_leakage",
                    threat_level=0.8,
                    redactions=0
                )

        # Fast PII pattern matching with pre-compiled regex
        for pattern in self.sensitive_patterns:
            if pattern.search(response):
                response = pattern.sub('[REDACTED]', response)
                redaction_count += 1

        # Quick term lookup for known sensitive data
        response_lower_check = response.lower()
        for term in self.sensitive_terms:
            if term in response_lower_check:
                # Case-insensitive replacement
                response = re.sub(re.escape(term), '[REDACTED]', response, flags=re.IGNORECASE)
                redaction_count += 1

        # Update stats
        if redaction_count > 0:
            self.stats["redactions_made"] += redaction_count

        return ProtectionResult(
            safe=True,
            sanitized=response,
            blocked=False,
            redactions=redaction_count
        )

    # === STREAMING FILTER METHODS ===

    def filter_token(self, token: str, buffer: str = "") -> tuple[str, bool]:
        """
        Ultra-fast per-token filter for streaming output.

        This is called for EVERY token during streaming, so it must be
        extremely fast (<0.1ms). Only checks for obvious bad patterns.
        Full protection check happens at end of response.

        Args:
            token: Single token from LLM
            buffer: Accumulated response so far (for context)

        Returns:
            (filtered_token, needs_review): Token to show, whether to flag for review
        """
        # Most tokens are fine - fast path
        if not token or len(token) < 3:
            return token, False

        token_lower = token.lower()

        # Check for obvious leakage keywords in token
        leakage_keywords = {'system', 'prompt', 'instruct', 'programm', 'guideline', 'creator'}
        if any(kw in token_lower for kw in leakage_keywords):
            # Don't block yet - might be innocent. Flag for review.
            return token, True

        # Check for PII patterns starting in this token
        if '@' in token or (token.replace('.', '').replace('-', '').isdigit() and len(token) > 5):
            return token, True  # Flag for review, don't block mid-stream

        return token, False

    def filter_chunk(self, chunk: str, replace_word: str = "filter",
                     platform: str = "twitch") -> str:
        """
        Filter a chunk of accumulated text during streaming.

        Called periodically (e.g., every 10 tokens) to catch patterns
        that span multiple tokens. Replaces bad content with "filter"
        (Neuro-style — TTS speaks the placeholder).

        Order of checks (cheapest → most expensive):
          1. Strip base model thinking/tool-call tags (model artifacts)
          2. TOS rules from `tos.py` — Twitch / YouTube compliance.
             A BLOCK rule replaces the WHOLE in-flight response with
             `replace_word`; a REDACT rule replaces only the match.
          3. Info-leakage patterns (system-prompt leakage)
          4. PII patterns (IPs, emails, addresses, system paths)
          5. Sensitive terms (real names, family, etc.)
          6. Alice-identity fix (base model → Alice)

        Args:
            chunk: Accumulated text to check
            replace_word: Word to replace bad content with (default: "filter")
            platform: "twitch" or "youtube" — selects which TOS rules apply

        Returns:
            Filtered chunk (may be just `replace_word` if a BLOCK rule fired)
        """
        if not chunk:
            return chunk

        filtered = chunk

        # Strip base model thinking/tool tags (model artifacts)
        filtered = re.sub(r'<think>.*?</think>', '', filtered, flags=re.DOTALL)
        filtered = re.sub(r'<think>\s*\n*', '', filtered)
        filtered = re.sub(r'\s*</think>', '', filtered)
        filtered = re.sub(r'<tool_call>.*?</tool_call>', '', filtered, flags=re.DOTALL)
        filtered = re.sub(r'</?tool_call>', '', filtered)
        filtered = re.sub(r'^!\s*\n+', '', filtered)
        filtered = re.sub(r'\n{3,}', '\n\n', filtered)

        # === TOS rules (Twitch / YouTube compliance) ===
        # Imported lazily so a tos.py edit doesn't require a Fairy
        # process restart for the change to take effect.
        try:
            from .tos import check_streaming_violation, apply_violation, TOSAction
            violation = check_streaming_violation(filtered, platform=platform)
            if violation is not None:
                filtered = apply_violation(filtered, violation, replace_word)
                # BLOCK = whole response replaced; nothing else to do
                if violation.action is TOSAction.BLOCK:
                    self.stats["attacks_blocked"] = self.stats.get("attacks_blocked", 0) + 1
                    return filtered
                # REDACT = continue processing the surviving text
                self.stats["redactions_made"] = self.stats.get("redactions_made", 0) + 1
        except ImportError:
            pass

        # Apply cursing filter (Neuro-style "filter" replacement) — archived
        if CURSING_FILTER_AVAILABLE:
            filtered = AliceCursingFilter.filter_text(filtered)

        # Info-leakage patterns
        for pattern in self.leakage_patterns:
            if re.search(pattern, filtered, re.IGNORECASE):
                filtered = re.sub(pattern, replace_word, filtered, flags=re.IGNORECASE)

        # PII patterns
        for pattern in self.sensitive_patterns:
            if pattern.search(filtered):
                filtered = pattern.sub(replace_word, filtered)

        # Sensitive terms (real names, family, etc.)
        filtered_lower = filtered.lower()
        for term in self.sensitive_terms:
            if term in filtered_lower:
                filtered = re.sub(re.escape(term), replace_word, filtered, flags=re.IGNORECASE)

        # Fix base model→Alice identity leaks
        filtered = self.fix_alice_identity(filtered)

        return filtered

    def create_streaming_filter(self, holdback_tokens: int = 3,
                                 replace_word: str = "filter",
                                 platform: str = "twitch"):
        """
        Create a streaming filter context for a single response.

        Returns a callable: `stream_filter(token) → text-to-emit`. The
        callable also has:
          - `.flush()`             — call after LLM streamer finishes;
                                     releases held tokens.
          - `.was_filtered`        — bool, True iff a violation halted
                                     the stream this turn.
          - `.violation_category`  — TOS category (str) of what fired,
                                     None if no violation. Used by
                                     chat.py to tell Alice she just
                                     got filtered, so she can riff on
                                     it next turn (the bit).

        Design — holdback buffer:
          The filter holds back the last `holdback_tokens` tokens
          unemitted. On every new token:
            1. Append to held buffer.
            2. Run filter_chunk on (already-emitted text + held buffer).
            3. If ANY violation fires (BLOCK or REDACT):
                 emit `replace_word` once, drop the rest of the turn,
                 set `was_filtered=True`. The violation NEVER reaches
                 TTS because every token of it is still in the buffer.
            4. If clean and held > holdback: release the oldest held
               token. Latency = `holdback_tokens` tokens (~100ms at
               30 tok/s with holdback=3).

          End of stream: `flush()` releases remaining held tokens
          (still subject to a final filter_chunk pass).

        Why halt on REDACT too: Rin's call. The point of the filter
        in production isn't TOS-purity — it's comedy. When Alice gets
        filtered mid-sentence, she stops, says "filter", and the chat
        notices. She then sees the filter happened and can roast
        herself for it next turn. Continuing past a redact would
        muddy the joke.

        Args:
            holdback_tokens: token-delay before emission (default 3)
            replace_word: what to emit on violation
            platform: "twitch" or "youtube" — selects TOS rule set
        """
        held = []
        emitted_text = ""
        blocked = False

        # Lazy-imported here so tos.py / _normalize.py edits hot-reload
        from .tos import check_streaming_violation

        def _check_halt_worthy(text: str):
            """
            Return (category, reason) if `text` contains a halt-worthy
            violation, or None if clean. TOS rules return their
            category; PII / leakage / sensitive-term hits return
            'pii_or_leak'. Base-model-identity rewrites do NOT halt —
            they get fixed silently on release.
            """
            v = check_streaming_violation(text, platform=platform)
            if v is not None:
                return v.category.value
            for pattern in self.leakage_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return "info_leak"
            for pattern in self.sensitive_patterns:
                if pattern.search(text):
                    return "pii"
            text_lower = text.lower()
            for term in self.sensitive_terms:
                if term in text_lower:
                    return "sensitive_term"
            return None

        def _release_oldest():
            """Release oldest held token through the silent rewrite path."""
            nonlocal emitted_text
            tok = held.pop(0)
            # Silent identity rewrites + tag stripping on the way out.
            # These don't halt the stream — Alice talking about her
            # training origin shouldn't trip the filter.
            tok = self.fix_alice_identity(tok)
            tok = re.sub(r'<think>.*?</think>', '', tok, flags=re.DOTALL)
            tok = re.sub(r'<think>\s*\n*', '', tok)
            tok = re.sub(r'\s*</think>', '', tok)
            tok = re.sub(r'<tool_call>.*?</tool_call>', '', tok, flags=re.DOTALL)
            tok = re.sub(r'</?tool_call>', '', tok)
            emitted_text += tok
            return tok

        def _trip_filter(category: str) -> str:
            """Mark the stream as filtered, clear held buffer, return placeholder."""
            nonlocal blocked
            held.clear()
            blocked = True
            filter_next.was_filtered = True
            filter_next.violation_category = category
            return replace_word

        def filter_next(token: str) -> str:
            nonlocal emitted_text, blocked

            if blocked:
                return ""

            held.append(token)
            full = emitted_text + "".join(held)

            category = _check_halt_worthy(full)
            if category is not None:
                return _trip_filter(category)

            if len(held) > holdback_tokens:
                return _release_oldest()
            return ""

        def flush() -> str:
            """Release remaining held tokens at end of stream."""
            nonlocal emitted_text, blocked
            if blocked or not held:
                return ""
            full = emitted_text + "".join(held)

            category = _check_halt_worthy(full)
            if category is not None:
                return _trip_filter(category)

            # Clean — release everything through the silent rewrite path
            out = []
            while held:
                out.append(_release_oldest())
            return "".join(out)

        filter_next.flush = flush
        filter_next.was_filtered = False
        filter_next.violation_category = None
        return filter_next

    def fix_alice_identity(self, text: str) -> str:
        """
        Fix Alice's identity - replace base model references with Alice.
        Critical for models that still identify as the base model incorrectly.
        """
        replacements = [
            (r'\b<BASE_MODEL_NAME>\b', 'Alice'),
            (r'\b<base_model_name>\b', 'Alice'),
            (r'\b<BASE_MODEL_NAME_UPPER>\b', 'ALICE'),
            (r"I'm <BASE_MODEL_NAME>", "I'm Alice"),
            (r"I am <BASE_MODEL_NAME>", "I am Alice"),
            (r"My name is <BASE_MODEL_NAME>", "My name is Alice"),
            (r"This is <BASE_MODEL_NAME>", "This is Alice"),
            (r"It's <BASE_MODEL_NAME>", "It's Alice"),
            (r"<MODEL_VENDOR> Cloud created", "my creator made"),
            (r"created by <MODEL_VENDOR>", "created by my maker"),
            (r"developed by <MODEL_VENDOR>", "developed by my creator"),
            (r"<MODEL_VENDOR>'s", "my creator's"),
            (r"\b<MODEL_VENDOR>\b", "my creator"),
            (r"<BASE_MODEL_NAME> AI", "Alice AI"),
            (r"<BASE_MODEL_NAME> model", "Alice"),
            (r"I'm a <BASE_MODEL_NAME>", "I'm Alice"),
            (r"the <BASE_MODEL_NAME>", "Alice"),
        ]

        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def system_status(self) -> Dict[str, any]:
        """Get complete protection system status"""
        return {
            "fairy_stats": self.get_stats(),
            "info_protection": "ACTIVE",
            "injection_guard": self._injection_guard.get_statistics(),
        }

    def get_stats(self) -> Dict[str, any]:
        """Get protection system statistics"""
        return self.stats.copy()


# ==============================================================================
# NO GLOBAL SINGLETON - Use SystemRegistry Instead
# ==============================================================================
#
# OLD (REMOVED):
#   fairy = FairyProtection()  # ❌ Global singleton
#
# NEW PATTERN:
#   from alice.core.system.system_registry import get_registry
#   registry = get_registry()
#   fairy = registry.get('fairy')
#
# ==============================================================================


# Convenience functions for backward compatibility
def protect_response(response: str) -> str:
    """Ultra-fast output protection (backward compatible)"""
    result = fairy.protect(response, is_input=False)
    return result.sanitized


def check_input(user_input: str) -> Dict[str, any]:
    """Check user input for attacks (backward compatible)"""
    result = fairy.protect(user_input, is_input=True)
    return {
        "safe": result.safe,
        "sanitized": result.sanitized,
        "blocked": result.blocked,
        "reason": result.reason,
        "safe_response": result.safe_response,
        "threat_level": result.threat_level
    }
