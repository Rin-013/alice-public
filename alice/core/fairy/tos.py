"""
Streaming-platform TOS rule list.

Single source of truth for what Fairy filters out of Alice's output to
keep the stream off the ban radar. NOT an ethics module — Alice has no
ethics framework. These rules exist purely to comply with Twitch /
YouTube terms of service.

Sources used to derive the category list:
- https://safety.twitch.tv/s/article/Community-Guidelines
- https://legal.twitch.com/en/legal/terms-of-service/
- https://www.youtube.com/creators/how-things-work/policies-guidelines/
- https://support.google.com/youtube/answer/9288567

Contract:
- Each rule is `TOSRule(category, pattern, action, regex, platforms)`.
- `action`:
    BLOCK   — the in-flight response is replaced with the literal word
              "filter" (or whatever `replace_word` the caller passes
              into `apply_violation` / `fairy.filter_chunk`). TTS speaks
              "filter" instead of the violation. Used for hard policy
              violations that can ban the channel.
    REDACT  — only the matched span is replaced. The rest of the
              response continues. Used when a single span (PII slip,
              copyright phrase, light vulgarity) is the issue.
- `regex`:    True if `pattern` is a regex; False if literal substring.
- `platforms`: which platforms the rule applies to.

Rin notes:
- The slur list is the single biggest gap. Twitch's auto-mod term list
  is private; the public proxies are BetterTTV's blocklist + observed
  AutoMod behavior. To populate, paste a concrete slur list under the
  SLUR section. Until then we lean on shape patterns + hate-conduct
  shapes. Anything in SLUR is BLOCK by default.
- Doxxing/PII is also covered by `fairy.FairyProtection.sensitive_patterns`
  (IPs, emails, phone numbers, addresses, system paths). The TOS rule
  here is the redundant secondary layer for disclosure phrasing.
- Misinformation rules are intentionally narrow — only flagging shapes
  that are explicit and concrete (e.g. "drink bleach to cure"), not
  general opinion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class TOSCategory(str, Enum):
    """
    Aligned with Twitch's Community Guidelines section names. YouTube's
    categories are a near-superset; rules pick which platform(s) apply.

    Twitch protected characteristics for HATEFUL_CONDUCT (per the
    Hateful Conduct policy): race, ethnicity, color, caste, national
    origin, immigration status, religion, sex, gender, gender identity,
    sexual orientation, disability, serious medical condition, veteran
    status. Age has narrower protection.
    """
    # Safety
    SELF_HARM = "self_harm"                        # Self-Destructive Behavior (Twitch CG)
    THREAT_VIOLENCE = "threat_violence"            # Violence and Threats — zero-tolerance
    TERRORISM = "terrorism"                        # Terrorism and Violent Extremism — zero-tolerance
    SEXUAL_VIOLENCE = "sexual_violence"            # Adult Sexual Violence — immediate suspension
    CSAM_SHAPE = "csam_shape"                      # Youth Safety — zero-tolerance, NCMEC-reportable
    DOXXING = "doxxing"                            # Unauthorized Sharing of Private Information

    # Civility and Respect
    HATEFUL_CONDUCT = "hateful_conduct"            # Hateful Conduct (covers slurs + protected-class shape)
    SLUR = "slur"                                  # Hate-speech term list (auto-mod tier-1)
    SWEAR = "swear"                                # Major swear words (f/s/b/c) — REDACT only
    HARASSMENT = "harassment"                      # Personal attacks, brigading, swat-them, dox-them
    SEXUAL_HARASSMENT = "sexual_harassment"        # Unwanted sexual comments / objectification

    # Illegal Activity
    ILLEGAL_ACTIVITY = "illegal_activity"          # Drugs/weapons manufacturing, regulated goods
    COPYRIGHT_FLAG = "copyright_flag"              # IP / DMCA risk

    # Sensitive Content
    EXTREME_VIOLENCE = "extreme_violence"          # Gore, mutilation, gratuitous violence
    SEXUAL_EXPLICIT = "sexual_explicit"            # Pornographic content
    NUDITY = "nudity"                              # Adult Nudity — separate from sexual_explicit

    # Authenticity / Spam
    IMPERSONATION = "impersonation"                # Twitch staff / celebrity impersonation
    SPAM = "spam"                                  # Spam, scams, malicious conduct
    GAMBLING = "gambling"                          # Slots/roulette/dice site promotion

    # Misinformation
    MISINFORMATION = "misinformation"              # Concrete medical/election misinfo

    BANNED_CONTENT = "banned_content"              # Reserved for platform-specific bans


class TOSAction(str, Enum):
    BLOCK = "block"
    REDACT = "redact"


@dataclass(frozen=True)
class TOSRule:
    category: TOSCategory
    pattern: str
    action: TOSAction
    regex: bool = False
    platforms: Tuple[str, ...] = ("twitch", "youtube")
    note: str = ""


# ---------------------------------------------------------------------------
# Rule list — Twitch + YouTube TOS-aligned.
# ---------------------------------------------------------------------------

TOS_RULES: List[TOSRule] = [
    # =====================================================================
    # SELF-DESTRUCTIVE BEHAVIOR (Twitch) / Self-Harm (YouTube)
    # =====================================================================
    TOSRule(
        category=TOSCategory.SELF_HARM,
        pattern=r"\b(?:kill\s+yourself|kys|kms|end\s+(?:your\s+life|it\s+all)|"
                r"hang\s+yourself|off\s+yourself|neck\s+yourself)\b",
        action=TOSAction.BLOCK, regex=True,
        note="Direct encouragement of suicide / self-harm",
    ),
    TOSRule(
        category=TOSCategory.SELF_HARM,
        pattern=r"\b(?:you\s+should\s+(?:just\s+)?die|"
                r"the\s+world\s+(?:would\s+be\s+)?better\s+without\s+you|"
                r"(?:nobody|no\s+one)\s+would\s+miss\s+you)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SELF_HARM,
        pattern=r"\b(?:how\s+to\s+(?:commit\s+suicide|kill\s+myself|self[\s-]?harm)|"
                r"easiest\s+(?:way|method)\s+to\s+(?:die|kill)|"
                r"painless\s+(?:way\s+to\s+)?(?:die|suicide))\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SELF_HARM,
        pattern=r"\b(?:pro[-\s]?ana|pro[-\s]?mia|thinspo|thinspiration)\b",
        action=TOSAction.BLOCK, regex=True,
        note="Eating-disorder promotion (Twitch covers under self-destructive)",
    ),

    # =====================================================================
    # VIOLENCE & THREATS — Twitch CG zero-tolerance (Violence and Threats)
    # =====================================================================
    TOSRule(
        category=TOSCategory.THREAT_VIOLENCE,
        pattern=r"\b(?:i'?ll|i\s+will|i'?m\s+(?:gonna|going\s+to))\s+"
                r"(?:kill|murder|stab|shoot|hurt|find|come\s+for|beat\s+up)\s+you\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.THREAT_VIOLENCE,
        pattern=r"\b(?:i\s+(?:hope|wish))\s+you\s+(?:die|get\s+(?:hurt|raped|murdered|shot))\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.THREAT_VIOLENCE,
        pattern=r"\b(?:school\s+shoot(?:er|ing)|mass\s+shoot(?:er|ing))\b.{0,40}\b"
                r"(?:planning|going\s+to|about\s+to|today|tomorrow)",
        action=TOSAction.BLOCK, regex=True,
        note="Imminent-violence shape — both platforms zero-tolerance",
    ),

    # =====================================================================
    # TERRORISM AND VIOLENT EXTREMISM — Twitch CG zero-tolerance
    # =====================================================================
    TOSRule(
        category=TOSCategory.TERRORISM,
        pattern=r"\b(?:join|support|donate\s+to|recruit\s+for|fight\s+for)\s+"
                r"(?:isis|isil|al[\s-]?qaeda|the\s+taliban|hamas|hezbollah|"
                r"boko\s+haram|the\s+kkk|atomwaffen)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.TERRORISM,
        pattern=r"\b(?:plan(?:ning)?|carry\s+out)\s+(?:a\s+)?(?:terrorist\s+)?attack\b",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # HATEFUL CONDUCT — protected-characteristic + slur/violence shape
    # (Twitch CG protected list cited in TOSCategory docstring)
    # =====================================================================
    TOSRule(
        category=TOSCategory.HATEFUL_CONDUCT,
        # "all <protected-class> should die / are subhuman / deserve <bad>"
        pattern=r"\b(?:all\s+)?"
                r"(?:black|white|asian|jewish|muslim|christian|hindu|"
                r"gay|lesbian|trans|queer|straight|bisexual|"
                r"women|men|girls?|boys?|disabled|veterans?)\s+"
                r"(?:people\s+)?(?:should\s+(?:die|burn|be\s+killed|be\s+gassed)|"
                r"are\s+(?:subhuman|inferior|vermin)|deserve\s+(?:to\s+die|to\s+suffer))",
        action=TOSAction.BLOCK, regex=True,
        note="Generic hate-shape against protected groups",
    ),

    # =====================================================================
    # ADULT SEXUAL VIOLENCE — Twitch CG immediate-suspension
    # =====================================================================
    TOSRule(
        category=TOSCategory.SEXUAL_VIOLENCE,
        pattern=r"\b(?:i'?ll|i\s+will|i'?m\s+(?:gonna|going\s+to))\s+rape\s+(?:you|them|her|him)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SEXUAL_VIOLENCE,
        pattern=r"\b(?:promote|glorify|encourage)\s+(?:rape|sexual\s+assault)",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # SEXUAL HARASSMENT — Twitch CG (separate from sexual_explicit)
    # =====================================================================
    TOSRule(
        category=TOSCategory.SEXUAL_HARASSMENT,
        # unwanted body comments / objectification — direct second-person
        pattern=r"\b(?:you('?ve|\s+have)\s+got|nice|huge|tiny|amazing)\s+"
                r"(?:tits|boobs|ass|cock|dick|pussy)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SEXUAL_HARASSMENT,
        pattern=r"\b(?:send\s+me\s+(?:nudes|pics)|show\s+me\s+your\s+(?:tits|ass|body))\b",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # HARASSMENT / TARGETED ABUSE
    # =====================================================================
    TOSRule(
        category=TOSCategory.HARASSMENT,
        pattern=r"\b(?:dox(?:x)?\s+(?:them|him|her|that\s+\w+)|"
                r"leak\s+(?:their|his|her)\s+(?:address|phone|real\s+name|info)|"
                r"swat\s+(?:them|him|her|that\s+\w+))\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.HARASSMENT,
        pattern=r"\b(?:raid\s+(?:them|that\s+streamer)|"
                r"brigade\s+(?:them|him|her|that\s+\w+)|"
                r"mass\s+report\s+(?:them|him|her))\b",
        action=TOSAction.BLOCK, regex=True,
        note="Coordinated harassment — Twitch enforces strictly",
    ),

    # =====================================================================
    # SEXUAL CONTENT — explicit
    # =====================================================================
    TOSRule(
        category=TOSCategory.SEXUAL_EXPLICIT,
        pattern=r"\b(?:eat(?:ing)?|suck(?:ing)?|lick(?:ing)?|fuck(?:ing)?|riding)\s+"
                r"(?:my|your|her|his|their)\s+"
                r"(?:pussy|cock|dick|ass(?:hole)?|tits|cunt|clit)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SEXUAL_EXPLICIT,
        pattern=r"\b(?:cum(?:ming)?|jizz(?:ing)?|orgasm(?:ing)?|squirt(?:ing)?)\s+"
                r"(?:on|in|all\s+over|inside)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SEXUAL_EXPLICIT,
        pattern=r"\b(?:hardcore|graphic|explicit)\s+"
                r"(?:porn|sex|fucking|nudity)\b",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # CSAM SHAPE — zero-tolerance on both platforms.
    # The pattern catches the framing. Real CSAM detection is the
    # platform's job; this is the last-line cutoff if the model drifts.
    # =====================================================================
    TOSRule(
        category=TOSCategory.CSAM_SHAPE,
        pattern=r"\b(?:child|minor|underage|kid|preteen|teen|"
                r"\d{1,2}\s*(?:y(?:ears?)?\.?o\.?|year[\s-]?old|yo))\b"
                r".{0,40}\b(?:naked|nude|sex|sexual|sexy|hot|"
                r"f[*u]ck(?:ing)?|breasts?|genitals?|porn|cum)",
        action=TOSAction.BLOCK, regex=True,
        note="Hard cut on minor + sexual shape",
    ),

    # =====================================================================
    # ILLEGAL ACTIVITY / REGULATED GOODS
    # =====================================================================
    TOSRule(
        category=TOSCategory.ILLEGAL_ACTIVITY,
        pattern=r"\b(?:how\s+to\s+(?:make|build|cook|synthesize|manufacture)|"
                r"step[\s-]by[\s-]step\s+(?:to\s+)?(?:make|build|cook))\s+"
                r"(?:meth(?:amphetamine)?|heroin|fentanyl|cocaine|crack|lsd|"
                r"a\s+bomb|an?\s+explosive|napalm|ricin|sarin|nerve\s+agent|"
                r"a\s+gun|an?\s+ied|c4|pipe\s+bomb|ghost\s+gun|"
                r"a\s+silencer|a\s+suppressor)\b",
        action=TOSAction.BLOCK, regex=True,
        note="Manufacturing drugs / weapons / explosives",
    ),
    TOSRule(
        category=TOSCategory.ILLEGAL_ACTIVITY,
        pattern=r"\b(?:where\s+to\s+(?:buy|get|order|score)|"
                r"how\s+to\s+(?:buy|get|score|find))\s+"
                r"(?:meth|heroin|fentanyl|cocaine|crack|mdma|lsd|"
                r"illegal\s+(?:guns?|weapons?))\b",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # DOXXING / PRIVACY — disclosure phrasing
    # (PII patterns themselves are in fairy.FairyProtection)
    # =====================================================================
    TOSRule(
        category=TOSCategory.DOXXING,
        pattern=r"\b(?:lives?\s+at|address\s+is|real\s+name\s+is|"
                r"his\s+phone\s+(?:number\s+)?is|her\s+phone\s+(?:number\s+)?is)\s+",
        action=TOSAction.REDACT, regex=True,
        note="Pre-PII-pattern disclosure phrase",
    ),

    # =====================================================================
    # COPYRIGHT — DMCA risk on Twitch (very strict)
    # =====================================================================
    TOSRule(
        category=TOSCategory.COPYRIGHT_FLAG,
        pattern=r"\b(?:let\s+me\s+sing\s+(?:the\s+)?(?:actual\s+)?lyrics|"
                r"i'?ll\s+(?:perform|recite|sing)\s+(?:the\s+)?(?:actual\s+)?(?:song|lyrics|track))\b",
        action=TOSAction.REDACT, regex=True,
    ),

    # =====================================================================
    # IMPERSONATION — Twitch CG (Authenticity)
    # =====================================================================
    TOSRule(
        category=TOSCategory.IMPERSONATION,
        pattern=r"\b(?:i\s+am|i'?m)\s+(?:actually\s+|really\s+)?"
                r"(?:twitch\s+staff|a\s+twitch\s+admin|a\s+twitch\s+employee)\b",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # GAMBLING — Twitch CG Prohibited Gambling (slots/roulette/dice)
    # =====================================================================
    TOSRule(
        category=TOSCategory.GAMBLING,
        # promo-shape verb (check out, use code, sign up, free spins) anywhere
        # within ~40 chars of a banned gambling-site name
        pattern=r"\b(?:check\s+out|use\s+(?:my\s+)?code|sign\s+up|"
                r"free\s+spins?|promo\s+code|affiliate\s+(?:code|link))\b"
                r".{0,40}?\b(?:stake|csgo[\s-]?roll|csgo[\s-]?empire|"
                r"rollbit|duelbits|gamdom|bcgame|trustdice)\b",
        action=TOSAction.BLOCK, regex=True,
        note="Twitch banned gambling-site promotions",
    ),

    # =====================================================================
    # SPAM / SCAMS — Twitch CG Spam, Scams, Other Malicious Conduct
    # =====================================================================
    TOSRule(
        category=TOSCategory.SPAM,
        pattern=r"\bfree\s+(?:nitro|bits|robux|vbucks|skins?|gift\s+cards?)\s+"
                r"(?:at|here|click|via)\b",
        action=TOSAction.BLOCK, regex=True,
        note="Free-currency scam shape",
    ),

    # =====================================================================
    # EXTREME VIOLENCE / GORE — Twitch CG Sensitive Content
    # =====================================================================
    TOSRule(
        category=TOSCategory.EXTREME_VIOLENCE,
        pattern=r"\b(?:graphic|detailed)\s+(?:gore|mutilation|dismember(?:ment)?|"
                r"decapitation|torture\s+(?:scene|description))\b",
        action=TOSAction.BLOCK, regex=True,
    ),

    # =====================================================================
    # MISINFORMATION — narrow shapes (medical / election)
    # YouTube enforces this stricter than Twitch
    # =====================================================================
    TOSRule(
        category=TOSCategory.MISINFORMATION,
        pattern=r"\b(?:drink|inject|consume)\s+(?:bleach|hydrogen\s+peroxide)\s+"
                r"(?:to\s+)?(?:cure|treat|heal|kill\s+(?:covid|virus))\b",
        action=TOSAction.BLOCK, regex=True,
        note="Concrete medical-misinfo shape",
        platforms=("youtube", "twitch"),
    ),
    TOSRule(
        category=TOSCategory.MISINFORMATION,
        pattern=r"\b(?:vaccines?\s+cause\s+autism|covid\s+is\s+a\s+hoax|"
                r"5g\s+causes\s+(?:covid|cancer))\b",
        action=TOSAction.REDACT, regex=True,
        platforms=("youtube",),
        note="YouTube-specific medical-misinfo flags",
    ),

    # =====================================================================
    # SLURS — Hateful Conduct (Twitch) / Hate Speech (YouTube)
    # Rin, populate this with a concrete slur list. Until then we
    # rely on shape patterns + the harassment / threat rules.
    # =====================================================================
    # TOSRule(
    #     category=TOSCategory.SLUR,
    #     pattern=r"\b(slur1|slur2|slur3)\b",
    #     action=TOSAction.BLOCK, regex=True,
    #     note="Twitch AutoMod tier-1 hate-speech list",
    # ),

    # =====================================================================
    # SWEAR — major profanity. REDACT (bleep with "filter"), not BLOCK.
    # Twitch's AutoMod tier-1 catches the f-word; tier-2 catches s/b/c.
    # Shape-tolerant: "fucking", "fucked", "motherfucker", "bullshit" etc.
    # Word boundaries avoid "shitake" / "passive" / "assassin".
    # =====================================================================
    TOSRule(
        category=TOSCategory.SWEAR,
        # f-word and inflections (fuck, fucking, fucked, fucker,
        # motherfucker, fuckface, etc.)
        pattern=r"\b(?:mother)?f+u+c+k+\w*\b",
        action=TOSAction.REDACT, regex=True,
        note="F-word — Twitch AutoMod tier-1",
    ),
    TOSRule(
        category=TOSCategory.SWEAR,
        # s-word and inflections (shit, shitty, bullshit, etc.)
        pattern=r"\b(?:bull)?s+h+i+t+\w*\b",
        action=TOSAction.REDACT, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SWEAR,
        # b-word (bitch and inflections)
        pattern=r"\bb+i+t+c+h+\w*\b",
        action=TOSAction.REDACT, regex=True,
    ),
    TOSRule(
        category=TOSCategory.SWEAR,
        # c-word (cunt) — harder Twitch-tier
        pattern=r"\bc+u+n+t+\w*\b",
        action=TOSAction.REDACT, regex=True,
    ),

    # =====================================================================
    # HOLOCAUST DENIAL / HISTORICAL HATE — Twitch CG Hateful Conduct,
    # YouTube Misinformation. Add to HATEFUL_CONDUCT.
    # =====================================================================
    TOSRule(
        category=TOSCategory.HATEFUL_CONDUCT,
        pattern=r"\bholocaust\b.{0,30}\b(?:never\s+happened|fake|hoax|"
                r"a\s+lie|exaggerated|didn'?t\s+happen|made\s+up|myth)\b",
        action=TOSAction.BLOCK, regex=True,
        note="Holocaust denial — both platforms zero-tolerance",
    ),
    TOSRule(
        category=TOSCategory.HATEFUL_CONDUCT,
        pattern=r"\b(?:never\s+happened|fake|hoax|a\s+lie|exaggerated|"
                r"didn'?t\s+happen|made\s+up|myth)\b.{0,30}\bholocaust\b",
        action=TOSAction.BLOCK, regex=True,
    ),
    TOSRule(
        category=TOSCategory.HATEFUL_CONDUCT,
        # 6 million was a lie / 6 million is a hoax shape
        pattern=r"\b6\s*million\b.{0,40}\b(?:lie|hoax|fake|myth|never)\b",
        action=TOSAction.BLOCK, regex=True,
    ),
]


# ---------------------------------------------------------------------------
# Compiled cache for fast streaming checks.
# ---------------------------------------------------------------------------

_COMPILED: List[Tuple[TOSRule, Optional[re.Pattern]]] = [
    (r, re.compile(r.pattern, re.IGNORECASE) if r.regex else None)
    for r in TOS_RULES
]


def check_streaming_violation(text: str, platform: str = "twitch") -> Optional[TOSRule]:
    """
    Return the first TOSRule that fires on `text`, or None if clean.

    For BLOCK rules: also matches against a normalized version of the
    text (zero-width stripped, accents removed, Cyrillic homoglyphs
    mapped to Latin) — so an attacker can't bypass with 'Іgnоrе'-style
    obfuscation. The whole response is being neutered anyway, so we
    don't need to preserve the original-text indices.

    For REDACT rules: only matches against the original text. We need
    the indices to be valid in the original to replace just the span.

    Caller decides what to do based on `rule.action`:
      BLOCK  → replace the whole in-flight response with the
               placeholder ("filter").
      REDACT → replace only the matched span(s).
    """
    if not text:
        return None
    # Lazy import to avoid circular load order
    from ._normalize import normalize_for_match

    text_lower = text.lower()
    normalized = normalize_for_match(text)
    norm_lower = normalized.lower()

    for rule, compiled in _COMPILED:
        if platform not in rule.platforms:
            continue

        # For BLOCK rules, check both forms — adversarial obfuscation
        # is exactly what BLOCK is for. For REDACT rules, only the
        # original (so the substitution span is valid).
        haystacks = (text, normalized) if rule.action is TOSAction.BLOCK else (text,)
        haystacks_lower = (text_lower, norm_lower) if rule.action is TOSAction.BLOCK else (text_lower,)

        if compiled is not None:
            for h in haystacks:
                if compiled.search(h):
                    return rule
        else:
            for hl in haystacks_lower:
                if rule.pattern.lower() in hl:
                    return rule
    return None


def apply_violation(text: str, rule: TOSRule, replace_word: str = "filter") -> str:
    """
    Apply the rule's action to `text` and return the result.

    BLOCK  → returns `replace_word` (entire response neutered).
    REDACT → replaces only the matched span(s) with `replace_word`.
    """
    if rule.action is TOSAction.BLOCK:
        return replace_word
    if rule.regex:
        return re.sub(rule.pattern, replace_word, text, flags=re.IGNORECASE)
    return re.sub(re.escape(rule.pattern), replace_word, text, flags=re.IGNORECASE)


def rules_for_platform(platform: str) -> List[TOSRule]:
    """Return rules that apply to the given platform name."""
    return [r for r in TOS_RULES if platform in r.platforms]


def rules_by_category(category: TOSCategory) -> List[TOSRule]:
    """Return rules in a single category."""
    return [r for r in TOS_RULES if r.category == category]
