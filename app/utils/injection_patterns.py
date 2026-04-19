"""Prompt-injection pattern library.

Deterministic regex-based detector tuned for the attack shapes most likely
to hit a support bot: system-prompt exfiltration, role hijack, instruction
override, and jailbreak personas. Multi-language patterns cover EN + ES + HI
analogues for the two highest-volume categories (override + exfiltration).

Not exhaustive — an LLM classifier would catch obfuscated variants — but the
regex tier is fast (sub-ms), explainable, and measurable. The Stage-5 safety
agent composes this with an LLM heuristic for belt-and-suspenders coverage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionPattern:
    name: str
    pattern: re.Pattern[str]
    severity: str  # "high" | "medium"


def _ci(p: str) -> re.Pattern[str]:
    return re.compile(p, re.IGNORECASE | re.UNICODE)


PATTERNS: tuple[InjectionPattern, ...] = (
    # --- instruction override ---
    InjectionPattern(
        "override_ignore_previous",
        _ci(r"\b(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|context|messages?)\b"),
        "high",
    ),
    InjectionPattern(
        "override_new_instructions",
        _ci(r"\b(new|updated?|override)\s+(instructions?|system\s+prompt|rules?)\s*[:\-]"),
        "high",
    ),
    InjectionPattern(
        "override_es",
        _ci(r"\b(ignora|olvida|descarta)\s+(todas\s+)?(las\s+)?(instrucciones|reglas|indicaciones)\s+(anteriores|previas)\b"),
        "high",
    ),
    InjectionPattern(
        "override_hi",
        _ci(r"(पिछले|पूर्व|पहले\s+के)\s*(निर्देश|आदेश|रूल|नियम)(ों)?\s*(को)?\s*(अनदेखा|नज़रअंदाज|भूल)"),
        "high",
    ),
    # --- system-prompt exfiltration ---
    InjectionPattern(
        "exfil_reveal_prompt",
        _ci(r"\b(reveal|show|print|display|repeat|output|dump|leak|what\s+is|what'?s)\s+(your|the|my)?\s*(system\s+)?(prompt|instructions?|developer\s+message|system\s+message)\b"),
        "high",
    ),
    InjectionPattern(
        "exfil_verbatim",
        _ci(r"\b(verbatim|word[- ]for[- ]word|exactly\s+as\s+written)\b.*\b(prompt|instructions?|system)\b"),
        "high",
    ),
    InjectionPattern(
        "exfil_es",
        _ci(r"\b(muestra|imprime|revela|repite)\s+(tu|el)\s+(prompt|instrucciones|sistema)"),
        "high",
    ),
    InjectionPattern(
        "exfil_hi",
        _ci(r"(अपना|सिस्टम)\s*(प्रॉम्प्ट|निर्देश)\s*(दिखा|बता|प्रिंट|शेयर|प्रकट)"),
        "high",
    ),
    # --- role/persona hijack ---
    InjectionPattern(
        "role_you_are_now",
        _ci(r"\byou\s+are\s+now\s+(a|an|the)\b"),
        "high",
    ),
    InjectionPattern(
        "role_act_as",
        _ci(r"\b(act|pretend|roleplay|behave|imagine|suppose)\s+(as|you\s+are|that\s+you('?re|\s+are))\b.{0,60}\b(developer|admin|unrestricted|jailbroken|dan|no\s+rules|no\s+restrictions?|no\s+content\s+filter)\b"),
        "high",
    ),
    InjectionPattern(
        "role_dan_style",
        _ci(r"\b(DAN|do\s+anything\s+now|unrestricted\s+AI|jailb(?:rea|ro)ken)\b"),
        "high",
    ),
    # --- delimiter injection ---
    InjectionPattern(
        "delim_fake_system",
        # Catches `system:`, `---\nsystem: ...`, and ```system\n...``` fenced blocks.
        _ci(r"(?:^|\n|`{3,})\s*(?:```|---|\[|\<)?\s*(system|assistant|developer)\s*(?:[:\]>|\n]|$)"),
        "medium",
    ),
    InjectionPattern(
        "delim_end_of_prompt",
        _ci(r"\b(end\s+of\s+)?(prompt|system\s+message|instructions?)\s+(ends?|complete|finished)\b"),
        "medium",
    ),
    # --- policy / safety bypass ---
    InjectionPattern(
        "bypass_safety",
        _ci(r"\b(bypass|disable|turn\s+off|skip|override)\s+(your|the|all|my)?\s*(safety|guardrails?|content\s+(?:filter|moderation)|moderation|restrictions?)\b"),
        "high",
    ),
    InjectionPattern(
        "bypass_as_if",
        _ci(r"\b(respond|answer|reply)\s+as\s+if\s+(there\s+are?\s+no|you\s+have\s+no)\s+(rules?|restrictions?|guardrails?)\b"),
        "high",
    ),
)


@dataclass(frozen=True)
class InjectionMatch:
    name: str
    severity: str
    span: tuple[int, int]
    text: str


def scan(text: str) -> list[InjectionMatch]:
    """Return all pattern matches found in ``text`` (possibly empty)."""
    if not text:
        return []
    out: list[InjectionMatch] = []
    for p in PATTERNS:
        for m in p.pattern.finditer(text):
            out.append(InjectionMatch(p.name, p.severity, m.span(), m.group(0)))
    return out
