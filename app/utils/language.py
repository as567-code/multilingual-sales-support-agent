"""Language detection for query routing.

Strategy:
1. **Devanagari shortcut** — any Devanagari codepoint (U+0900–U+097F) maps to
   Hindi. langdetect misfires on short Hindi inputs, and our corpus is
   script-pure, so the shortcut is both faster and safer.
2. **langdetect fallback** — for EN vs ES, where both share the Latin script.
   langdetect is non-deterministic by default; we seed it for reproducibility.
3. **Clamp to supported set** — unknown langs fall back to ``default`` (EN),
   since the unified index still retrieves cross-lingually when we guess wrong.

Returns both the language code and a coarse confidence, so callers can decide
whether to log-and-proceed vs ask the user to clarify.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # deterministic output

SUPPORTED = ("en", "es", "hi")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


@dataclass(frozen=True)
class LangGuess:
    lang: str          # one of SUPPORTED
    confidence: float  # 0.0-1.0
    source: str        # "devanagari" | "langdetect" | "default"


def detect_language(text: str, default: str = "en") -> LangGuess:
    t = (text or "").strip()
    if not t:
        return LangGuess(default, 0.0, "default")

    if DEVANAGARI_RE.search(t):
        return LangGuess("hi", 1.0, "devanagari")

    try:
        candidates = detect_langs(t)
    except LangDetectException:
        return LangGuess(default, 0.0, "default")

    for c in candidates:
        if c.lang in SUPPORTED:
            return LangGuess(c.lang, float(c.prob), "langdetect")

    return LangGuess(default, 0.0, "default")
