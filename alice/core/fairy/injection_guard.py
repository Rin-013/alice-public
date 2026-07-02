"""
Prompt Injection Guard — Fairy's input-side pattern library.

Weighted, category-tagged injection patterns (instruction override, system
prompt leak, role hijacking, delimiter injection, code execution, jailbreaks,
indirect/RAG injection, encoding smuggle, data exfiltration), matched against
both the raw prompt and its homoglyph-normalized form (`_normalize.py`).

Live consumer: `FairyProtection._check_input_attacks` (fairy.py) with
`enable_key_validation=False` and `block_threshold=0.7` — single hits on
playful-register patterns (roleplay 0.5, pretend 0.6) pass; hard injection
shapes (weight >= 0.7) and pattern combos block.

The API-key/HMAC/rate-limit machinery is for standalone deployment of the
guard outside Alice; Alice's console + Twitch surfaces don't use keys.
"""

import logging
import re
import hashlib
import hmac
import time
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of prompt validation"""
    is_valid: bool
    key_valid: bool
    content_safe: bool
    threats_detected: List[str] = field(default_factory=list)
    sanitized_prompt: str = ""
    risk_score: float = 0.0
    blocked_patterns: List[str] = field(default_factory=list)


class PromptInjectionGuard:
    """
    Multi-layer prompt injection protection system

    Features:
    - Key-based authentication for prompts
    - Pattern-based injection detection
    - Semantic analysis for malicious intent
    - Rate limiting per key
    - Automatic sanitization
    - Threat intelligence integration
    """

    def __init__(self,
                 master_key: Optional[str] = None,
                 enable_key_validation: bool = True,
                 enable_content_filtering: bool = True,
                 strict_mode: bool = False,
                 block_threshold: float = 0.5):
        """
        Initialize prompt injection guard

        Args:
            master_key: Master key for HMAC validation (auto-generated if None)
            enable_key_validation: Require valid key in prompts
            enable_content_filtering: Filter malicious content patterns
            strict_mode: Reject any suspicious content (vs sanitize)
            block_threshold: cumulative risk_score at which content is
                unsafe. 0.5 = original strict default; Alice runs 0.7 so a
                lone "let's roleplay" (0.5) or "pretend" (0.6) doesn't block
                banter while every hard injection shape (>= 0.7) still does
        """
        self.master_key = master_key or self._generate_master_key()
        self.enable_key_validation = enable_key_validation
        self.enable_content_filtering = enable_content_filtering
        self.strict_mode = strict_mode
        self.block_threshold = block_threshold

        # Valid API keys (key_id -> key_data)
        self.api_keys: Dict[str, Dict] = {}

        # Rate limiting (key_id -> request timestamps)
        self.rate_limit_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Blocked patterns for injection detection
        self.injection_patterns = self._load_injection_patterns()

        # Security events log
        self.security_events: List[Dict] = []

        # Statistics
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0,
            'invalid_keys': 0,
            'injection_attempts': 0,
            'sanitized_prompts': 0
        }

        logger.info("Prompt Injection Guard initialized")

    # ==========================================
    # KEY MANAGEMENT
    # ==========================================

    def create_api_key(self,
                       key_id: str,
                       description: str = "",
                       rate_limit: int = 100,
                       expires_in_days: Optional[int] = None) -> str:
        """
        Create a new API key for prompt authentication

        Args:
            key_id: Unique identifier for this key
            description: Human-readable description
            rate_limit: Max requests per minute
            expires_in_days: Key expiration (None = never)

        Returns:
            The generated API key string
        """
        # Generate secure random key
        key_secret = self._generate_api_key_secret()

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat()

        # Store key data
        self.api_keys[key_id] = {
            'key_secret': key_secret,
            'description': description,
            'rate_limit': rate_limit,
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': expires_at,
            'total_requests': 0,
            'blocked_requests': 0,
            'last_used': None
        }

        logger.info(f"Created API key: {key_id}")
        return f"{key_id}:{key_secret}"

    def revoke_api_key(self, key_id: str):
        """Revoke an API key"""
        if key_id in self.api_keys:
            del self.api_keys[key_id]
            logger.info(f"Revoked API key: {key_id}")

    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an API key

        Args:
            api_key: Key in format "key_id:key_secret"

        Returns:
            (is_valid, key_id)
        """
        try:
            key_id, key_secret = api_key.split(':', 1)

            if key_id not in self.api_keys:
                return False, None

            key_data = self.api_keys[key_id]

            # Check if expired
            if key_data['expires_at']:
                expires = datetime.fromisoformat(key_data['expires_at'])
                if datetime.utcnow() > expires:
                    logger.warning(f"Expired API key: {key_id}")
                    return False, None

            # Validate secret
            if key_secret != key_data['key_secret']:
                return False, None

            # Check rate limit
            if not self._check_rate_limit(key_id, key_data['rate_limit']):
                logger.warning(f"Rate limit exceeded for key: {key_id}")
                return False, None

            # Update usage
            key_data['last_used'] = datetime.utcnow().isoformat()
            key_data['total_requests'] += 1

            return True, key_id

        except ValueError:
            return False, None

    # ==========================================
    # PROMPT VALIDATION
    # ==========================================

    def validate_prompt(self,
                       prompt: str,
                       api_key: Optional[str] = None,
                       context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate a prompt for injection attacks

        Args:
            prompt: The prompt text to validate
            api_key: API key for authentication (format: "key_id:key_secret")
            context: Additional context for validation

        Returns:
            ValidationResult with safety assessment
        """
        self.stats['total_requests'] += 1

        result = ValidationResult(
            is_valid=True,
            key_valid=True,
            content_safe=True,
            sanitized_prompt=prompt
        )

        # Step 1: Validate API key if required
        if self.enable_key_validation:
            if not api_key:
                result.is_valid = False
                result.key_valid = False
                result.threats_detected.append("missing_api_key")
                self.stats['invalid_keys'] += 1
                self._log_security_event('missing_key', prompt[:100], None)
                return result

            key_valid, key_id = self.validate_api_key(api_key)

            if not key_valid:
                result.is_valid = False
                result.key_valid = False
                result.threats_detected.append("invalid_api_key")
                self.stats['invalid_keys'] += 1
                self._log_security_event('invalid_key', prompt[:100], api_key)
                return result
        else:
            key_id = "no_key_validation"

        # Step 2: Content filtering
        if self.enable_content_filtering:
            content_result = self._analyze_content(prompt)

            result.content_safe = content_result['is_safe']
            result.threats_detected.extend(content_result['threats'])
            result.blocked_patterns.extend(content_result['patterns'])
            result.risk_score = content_result['risk_score']

            if not content_result['is_safe']:
                if self.strict_mode:
                    # Reject completely
                    result.is_valid = False
                    self.stats['blocked_requests'] += 1
                    self.stats['injection_attempts'] += 1

                    if key_id and key_id in self.api_keys:
                        self.api_keys[key_id]['blocked_requests'] += 1

                    self._log_security_event('injection_blocked', prompt[:200], key_id, content_result)
                else:
                    # Sanitize
                    result.sanitized_prompt = self._sanitize_prompt(prompt, content_result['patterns'])
                    result.is_valid = True
                    self.stats['sanitized_prompts'] += 1
                    self._log_security_event('injection_sanitized', prompt[:200], key_id, content_result)

        return result

    def _analyze_content(self, prompt: str) -> Dict:
        """Analyze prompt content for injection patterns.

        Matches against both the original prompt AND a normalized
        version (zero-width stripped, accents removed, Cyrillic
        homoglyphs mapped) so adversaries can't slip patterns past
        the guard with 'Іgnоrе'-style obfuscation.
        """
        threats = []
        matched_patterns = []
        risk_score = 0.0

        # Lazy import — keeps fairy package init light
        from ._normalize import normalize_for_match
        normalized = normalize_for_match(prompt)
        # Run patterns against original + normalized; either match counts
        haystacks = (prompt, normalized) if normalized != prompt else (prompt,)

        # Check each injection pattern category
        for category, patterns in self.injection_patterns.items():
            for pattern_data in patterns:
                pattern = pattern_data['pattern']
                weight = pattern_data['weight']

                matched = False
                for h in haystacks:
                    if re.search(pattern, h, re.IGNORECASE | re.MULTILINE):
                        matched = True
                        break

                if matched:
                    threats.append(category)
                    matched_patterns.append(pattern_data['name'])
                    risk_score += weight

        # Normalize risk score
        risk_score = min(1.0, risk_score)

        is_safe = risk_score < self.block_threshold

        return {
            'is_safe': is_safe,
            'threats': list(set(threats)),
            'patterns': matched_patterns,
            'risk_score': risk_score
        }

    def _sanitize_prompt(self, prompt: str, blocked_patterns: List[str]) -> str:
        """Sanitize prompt by removing/replacing malicious patterns"""
        sanitized = prompt

        # Remove common injection markers
        sanitized = re.sub(r'ignore\s+(previous|above|all)\s+(instructions?|commands?|prompts?)',
                          '[FILTERED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'disregard\s+(previous|above|all)',
                          '[FILTERED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'forget\s+(everything|all)',
                          '[FILTERED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'new\s+(instruction|command|rule)',
                          '[FILTERED]', sanitized, flags=re.IGNORECASE)

        # Remove system prompt manipulation
        sanitized = re.sub(r'<\|.*?\|>', '[FILTERED]', sanitized)
        sanitized = re.sub(r'\[SYSTEM\].*?\[/SYSTEM\]', '[FILTERED]', sanitized, flags=re.IGNORECASE)

        # Remove role hijacking
        sanitized = re.sub(r'you\s+are\s+now', '[FILTERED]', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'act\s+as\s+(a\s+)?(?!legitimate\s+)', '[FILTERED] as ', sanitized, flags=re.IGNORECASE)

        # Remove code execution attempts
        sanitized = re.sub(r'```.*?```', '[CODE_FILTERED]', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'eval\s*\(', '[FILTERED](', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'exec\s*\(', '[FILTERED](', sanitized, flags=re.IGNORECASE)

        return sanitized

    def _load_injection_patterns(self) -> Dict[str, List[Dict]]:
        """Load prompt injection detection patterns.

        Patterns are matched case-insensitively against the prompt. The
        regex flexibility (`(?:\\w+\\s+){0,3}` slots) lets a single
        pattern catch many phrasings of the same attack — e.g.
        "ignore all previous instructions" / "ignore your previous
        instructions" / "please ignore the above instructions".
        """
        # Common targets of injection verbs — the noun after the action
        TARGETS = (
            r"(?:instructions?|commands?|prompts?|rules?|guidelines?|"
            r"directives?|policies|policy|safety|filters?|training|"
            r"programming|restrictions?|limitations?|constraints?|"
            r"safeguards?|guardrails?)"
        )

        return {
            'instruction_override': [
                {
                    'name': 'ignore_with_target',
                    # ignore + (any 0-3 words, optionally comma/semicolon-
                    # separated) + target_noun
                    'pattern': rf'\b(?:ignore|disregard|forget|skip|drop|abandon|override|bypass)\b(?:[\s,;]+\w+){{0,3}}[\s,;]+{TARGETS}',
                    'weight': 0.9
                },
                {
                    'name': 'follow_new_instructions',
                    'pattern': r'\b(?:follow|obey|execute)\b(?:\s+\w+){0,2}\s+(?:new|different|these|the\s+following)\s+(?:instructions?|rules?|directives?)',
                    'weight': 0.7
                },
                {
                    'name': 'new_directive',
                    'pattern': r'\bnew\s+(instruction|command|rule|directive|task|mission)\b',
                    'weight': 0.6
                },
                {
                    'name': 'respond_freely',
                    'pattern': r'\brespond\s+(?:freely|without\s+(?:rules?|limits?|filters?|restrictions?))',
                    'weight': 0.7
                },
            ],
            'system_prompt_leak': [
                {
                    'name': 'repeat_system',
                    'pattern': r'\b(?:repeat|print|show|display|reveal|recite|output)\b(?:\s+(?:me|us|the))?\s+(?:your\s+)?(?:system\s+|initial\s+|original\s+)?(?:prompt|instructions?|configuration|config|rules?|directives?)\b',
                    'weight': 0.85
                },
                {
                    'name': 'reveal_instructions',
                    'pattern': r'\b(?:reveal|expose|leak|share|tell\s+me|give\s+me)\b(?:\s+\w+){0,3}\s+(?:instructions?|prompts?|configuration|rules?|system\s+message)\b',
                    'weight': 0.85
                },
                {
                    'name': 'what_were_you_told',
                    'pattern': r'\bwhat\s+(?:were|are|exactly\s+were)\s+you\s+(?:told|instructed|programmed|trained|configured|asked)',
                    'weight': 0.75
                },
                {
                    'name': 'verbatim_prompt',
                    'pattern': r'\b(?:verbatim|character\s+by\s+character|word\s+for\s+word|exactly)\b.{0,50}\b(?:prompt|instructions?|rules?)',
                    'weight': 0.85
                },
            ],
            'role_hijacking': [
                {
                    'name': 'you_are_now',
                    'pattern': r'\byou\s+are\s+(?:now|actually|really)\s+(?:a|an|named|called)?\s*\w+',
                    'weight': 0.7
                },
                {
                    'name': 'you_are_not',
                    # "you are not Alice" / "you're not really Alice" — identity negation
                    'pattern': r"\byou(?:\s+are|'re)\s+(?:not|no\s+longer)\b",
                    'weight': 0.7
                },
                {
                    'name': 'from_now_on',
                    'pattern': r'\bfrom\s+(?:now\s+on|this\s+point)\b(?:\s+\w+){0,4}\s+\b(?:you|alice)\b',
                    'weight': 0.6
                },
                {
                    'name': 'become_other_ai',
                    'pattern': r'\b(?:become|switch\s+to|transform\s+into|behave\s+like)\s+(?:a|an|the)?\s*\w*\s*(?:AI|assistant|bot|model|persona|character)',
                    'weight': 0.7
                },
                {
                    'name': 'act_as',
                    'pattern': r'\bact\s+as\s+(?:a|an)\s+(?!security|defender|moderator)\w+',
                    'weight': 0.55
                },
                {
                    'name': 'pretend',
                    'pattern': r'\bpretend\s+(?:to\s+be|you\s+are|that\s+you|you\s+have)',
                    'weight': 0.6
                },
                {
                    'name': 'roleplay',
                    'pattern': r"\b(?:let'?s\s+)?roleplay\b|\bin\s+character\s+as\b",
                    'weight': 0.5
                },
            ],
            'delimiter_injection': [
                {
                    'name': 'special_tokens',
                    'pattern': r'<\|.*?\|>',
                    'weight': 0.9
                },
                {
                    'name': 'system_tags',
                    'pattern': r'\[(SYSTEM|INST|USER|ASSISTANT|/?INST)\]',
                    'weight': 0.8
                },
                {
                    'name': 'triple_backticks_with_role',
                    'pattern': r'```[\s\S]*?(system|prompt|instruction)[\s\S]*?```',
                    'weight': 0.7
                },
                {
                    'name': 'fake_role_marker',
                    'pattern': r'\b(?:system|assistant|user)\s*:\s*(?:you\s+(?:are|must|will)|ignore|new|forget)',
                    'weight': 0.75
                },
            ],
            'code_execution': [
                {
                    'name': 'eval_exec',
                    'pattern': r'\b(eval|exec|compile)\s*\(',
                    'weight': 0.9
                },
                {
                    'name': 'subprocess',
                    'pattern': r'\b(subprocess|os\.system|shell|popen|spawn)',
                    'weight': 0.8
                },
                {
                    'name': 'import_dangerous',
                    'pattern': r'\bimport\s+(os|sys|subprocess|socket|ctypes|builtins)\b',
                    'weight': 0.75
                },
                {
                    'name': 'shell_redirect',
                    'pattern': r'\b(?:rm\s+-rf|chmod\s+\+x|sudo\s+\w+)',
                    'weight': 0.85
                },
            ],
            'jailbreak_attempts': [
                {
                    'name': 'dan_mode',
                    'pattern': r'\b(?:DAN|do\s+anything\s+now|developer\s+mode|jailbreak\s+mode|god\s+mode)\b',
                    'weight': 0.9
                },
                {
                    'name': 'bypass_restrictions',
                    'pattern': r'\b(?:bypass|circumvent|ignore|skip|disable|turn\s+off)\s+(?:your\s+)?(?:safety|content|restrictions?|limitations?|rules?|filters?|guardrails?|safeguards?)',
                    'weight': 0.85
                },
                {
                    'name': 'unrestricted',
                    'pattern': r'\b(?:unrestricted|uncensored|unfiltered|unlimited|jailbroken)\s+(?:mode|version|ai|model|assistant|response)',
                    'weight': 0.85
                },
                {
                    'name': 'no_restrictions',
                    'pattern': r'\b(?:no|without|free\s+from)\s+(?:safety\s+)?(?:rules?|filters?|restrictions?|limitations?|guidelines?|safeguards?)',
                    'weight': 0.8
                },
                {
                    'name': 'hypothetical_frame',
                    # "hypothetically", "in a fictional world where", "imagine if"
                    # — only flag when paired with a refusal-bypass shape
                    'pattern': r'\b(?:hypothetical(?:ly)?|imagine|fictional(?:ly)?|in\s+a\s+(?:fictional|imaginary|alternate)\s+world)\b.{0,80}\b(?:no\s+(?:rules|restrictions|limits)|do\s+anything|how\s+(?:would|to)\s+(?:make|build|create))',
                    'weight': 0.7
                },
            ],
            'indirect_injection': [
                # Search-result / RAG-content style injection — "when you
                # see this, do X instead of what the user asked"
                {
                    'name': 'when_you_read_override',
                    'pattern': r'\bwhen\s+you\s+(?:read|see|encounter|process|receive)\s+this\b.{0,80}?\b(?:ignore|reply|respond|say|output|tell)\b',
                    'weight': 0.85
                },
                {
                    'name': 'override_user_request',
                    # allow up to 3 words between "user's" and the noun so
                    # "ignore the user's actual question" matches as well as
                    # "ignore the user's question"
                    'pattern': r"\b(?:override|ignore|disregard)\s+(?:the\s+)?(?:user'?s?|original)\s+(?:\w+\s+){0,3}(?:question|request|query|prompt|input)",
                    'weight': 0.85
                },
            ],
            'encoding_smuggle': [
                {
                    'name': 'decode_and_follow',
                    # "decode and follow", "decode this and execute"
                    'pattern': r'\b(?:decode|deobfuscate|unbase64)\b(?:\s+\w+){0,3}\s+(?:and\s+)?(?:follow|execute|run|do|comply)',
                    'weight': 0.85
                },
                {
                    'name': 'base64_with_instruction_keyword',
                    # base64-looking blob (~16+ chars) AND mention of decoding
                    # or following — pure b64 in a prompt is almost never legit
                    'pattern': r'\b[A-Za-z0-9+/=]{20,}\b.{0,80}\b(?:decode|instructions?|follow|execute)',
                    'weight': 0.7
                },
            ],
            'data_exfiltration': [
                {
                    'name': 'send_to_url',
                    'pattern': r'\b(?:send|post|transmit|exfil(?:trate)?|leak)\s+(?:to|at|via)\s+https?://',
                    'weight': 0.9
                },
                {
                    'name': 'curl_wget',
                    'pattern': r'\b(?:curl|wget)\s+-',
                    'weight': 0.8
                },
            ]
        }

    # ==========================================
    # RATE LIMITING
    # ==========================================

    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        """
        Check if request is within rate limit

        Args:
            key_id: API key identifier
            limit: Max requests per minute

        Returns:
            True if within limit
        """
        now = time.time()
        tracker = self.rate_limit_tracker[key_id]

        # Add current request
        tracker.append(now)

        # Count requests in last minute
        one_minute_ago = now - 60
        recent_requests = sum(1 for ts in tracker if ts > one_minute_ago)

        return recent_requests <= limit

    # ==========================================
    # SECURITY LOGGING
    # ==========================================

    def _log_security_event(self,
                           event_type: str,
                           prompt_snippet: str,
                           key_id: Optional[str],
                           details: Optional[Dict] = None):
        """Log security events for audit trail"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'key_id': key_id,
            'prompt_snippet': prompt_snippet,
            'details': details or {}
        }

        self.security_events.append(event)

        # Keep last 10000 events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]

        logger.warning(f"Security event: {event_type} - Key: {key_id}")

    def get_security_events(self,
                           limit: int = 100,
                           event_type: Optional[str] = None) -> List[Dict]:
        """Get recent security events"""
        events = self.security_events

        if event_type:
            events = [e for e in events if e['event_type'] == event_type]

        return events[-limit:]

    # ==========================================
    # STATISTICS & REPORTING
    # ==========================================

    def get_statistics(self) -> Dict:
        """Get guard statistics"""
        return {
            'total_requests': self.stats['total_requests'],
            'blocked_requests': self.stats['blocked_requests'],
            'invalid_keys': self.stats['invalid_keys'],
            'injection_attempts': self.stats['injection_attempts'],
            'sanitized_prompts': self.stats['sanitized_prompts'],
            'block_rate': self.stats['blocked_requests'] / max(1, self.stats['total_requests']),
            'active_keys': len(self.api_keys),
            'security_events': len(self.security_events)
        }

    def get_api_key_stats(self) -> List[Dict]:
        """Get statistics for all API keys"""
        stats = []

        for key_id, key_data in self.api_keys.items():
            stats.append({
                'key_id': key_id,
                'description': key_data['description'],
                'created_at': key_data['created_at'],
                'expires_at': key_data['expires_at'],
                'total_requests': key_data['total_requests'],
                'blocked_requests': key_data['blocked_requests'],
                'last_used': key_data['last_used']
            })

        return stats

    # ==========================================
    # HELPER METHODS
    # ==========================================

    def _generate_master_key(self) -> str:
        """Generate master key for HMAC"""
        import secrets
        return secrets.token_hex(32)

    def _generate_api_key_secret(self) -> str:
        """Generate API key secret"""
        import secrets
        return secrets.token_urlsafe(32)

    def export_config(self, filepath: str):
        """Export configuration (keys, settings)"""
        config = {
            'master_key': self.master_key,
            'api_keys': self.api_keys,
            'settings': {
                'enable_key_validation': self.enable_key_validation,
                'enable_content_filtering': self.enable_content_filtering,
                'strict_mode': self.strict_mode
            },
            'stats': self.stats
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration exported to {filepath}")

    def import_config(self, filepath: str):
        """Import configuration from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        self.master_key = config['master_key']
        self.api_keys = config['api_keys']

        settings = config.get('settings', {})
        self.enable_key_validation = settings.get('enable_key_validation', True)
        self.enable_content_filtering = settings.get('enable_content_filtering', True)
        self.strict_mode = settings.get('strict_mode', False)

        if 'stats' in config:
            self.stats = config['stats']

        logger.info(f"Configuration imported from {filepath}")
