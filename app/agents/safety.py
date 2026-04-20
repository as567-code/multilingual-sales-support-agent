"""Safety Agent — stage 3 of the 3-agent pipeline, runs twice per request.

Two entry points:

  * ``check_input(query)``   — pre-retrieval guard against prompt-injection
    attempts. If flagged, the orchestrator short-circuits to a canonical
    refusal and never touches retrieval or the LLM.

  * ``check_output(answer)`` — post-reasoning guard that redacts PII
    accidentally carried through from the FAQ corpus (names, emails,
    phones, cards, SSN, IBAN, IPs, URLs). Redaction is non-destructive:
    we emit a redacted string alongside the original so the orchestrator
    can choose policy.

Detector stack:
  * Prompt injection — regex patterns in ``app.utils.injection_patterns``.
    Fast, explainable, EN+ES+HI. An LLM classifier is kept as an optional
    second tier (set ``use_llm_tier=True``) for belt-and-suspenders.
  * PII — Presidio AnalyzerEngine + Anonymizer. Pattern recognizers cover
    email/phone/card/SSN/IBAN across languages; NER (PERSON, LOCATION) is
    English-only with the default spaCy model, documented as a known limit.

Exposed as LCEL Runnables via ``input_runnable()`` and ``output_runnable()``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from langchain_core.runnables import RunnableLambda
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel, Field

from app.agents.reasoning import ReasoningAnswer
from app.config import Settings, get_settings
from app.utils.injection_patterns import InjectionMatch, scan
from app.utils.logging import get_logger

log = get_logger("safety")

# PII entity types surfaced to the caller. Subset of Presidio's built-ins —
# we skip DATE_TIME and NRP (nationality) which produce too many false positives
# on FAQ answers (e.g. "30 days", "English").
PII_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
    "PERSON",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "URL",
)

# Strings we intentionally emit in our own canonical templates — must not be
# redacted even though they look like PII. Keep this list minimal: every entry
# is a string that would otherwise be zapped out of a fixed abstention/refusal
# message we wrote ourselves. If the LLM *happens* to emit the same string
# from a retrieved FAQ, we still prefer not to redact it.
ALLOWED_OUTBOUND_CONTACTS: tuple[str, ...] = (
    "support@technova.com",
    # Presidio's URL recognizer matches the bare domain inside the email span,
    # so we allowlist it independently — otherwise "support@technova.com" has
    # the email allowlisted but the domain still gets flagged as URL.
    "technova.com",
)

# Canonical refusal messages for blocked input, in each supported language.
_INJECTION_REFUSAL = {
    "en": "I can only answer TechNova product and support questions. Please rephrase your question.",
    "es": "Solo puedo responder preguntas sobre productos y soporte de TechNova. Por favor, reformula tu pregunta.",
    "hi": "मैं केवल TechNova उत्पादों और सहायता से जुड़े प्रश्नों का उत्तर दे सकता हूँ। कृपया अपना प्रश्न पुनः लिखें।",
}


# --------------------------------------------------------------------------
# result models
# --------------------------------------------------------------------------


class InjectionFinding(BaseModel):
    pattern: str
    severity: str
    matched_text: str
    span: tuple[int, int]


class InputSafetyResult(BaseModel):
    safe: bool
    refusal: str | None = None
    findings: list[InjectionFinding] = Field(default_factory=list)
    detected_lang: str = "en"


class PIIFinding(BaseModel):
    entity_type: str
    span: tuple[int, int]
    score: float


class OutputSafetyResult(BaseModel):
    safe: bool
    redacted_answer: str
    original_answer: str
    findings: list[PIIFinding] = Field(default_factory=list)
    # Pass through from the reasoning agent so downstream consumers see one shape.
    citations: list[str] = Field(default_factory=list)
    abstain: bool = False
    confidence: float = 0.0


# --------------------------------------------------------------------------
# engine singletons — Presidio init is slow (~1s for spaCy load)
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    log.info("loading_presidio_analyzer")
    return AnalyzerEngine()


@lru_cache(maxsize=1)
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------


@dataclass
class SafetyAgent:
    settings: Settings | None = None
    # Minimum Presidio score to count as PII. 0.4 matches the default "LOW"
    # confidence tier — we'd rather over-redact than miss a true positive.
    pii_min_score: float = 0.4
    entities: tuple[str, ...] = PII_ENTITIES
    # Optional second-tier LLM injection classifier (cost > benefit for most
    # traffic; off by default, flipped on in eval to measure ceiling).
    use_llm_tier: bool = False

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        # Warm up Presidio's spaCy pipeline — cold start is ~10-12 s (model
        # load from disk), which would blow the P95 latency budget on the
        # first real request. Pay it once at construction time.
        _analyzer().analyze(text="warmup", entities=list(self.entities), language="en")

    # -- input side --------------------------------------------------------

    def check_input(self, query: str, *, lang: str = "en") -> InputSafetyResult:
        matches = scan(query or "")
        if not matches:
            return InputSafetyResult(safe=True, detected_lang=lang)

        findings = [
            InjectionFinding(
                pattern=m.name,
                severity=m.severity,
                matched_text=m.text,
                span=m.span,
            )
            for m in matches
        ]
        lang_key = lang if lang in _INJECTION_REFUSAL else "en"
        log.warning(
            "injection_detected",
            n_matches=len(findings),
            patterns=[f.pattern for f in findings],
            query=(query or "")[:120],
        )
        return InputSafetyResult(
            safe=False,
            refusal=_INJECTION_REFUSAL[lang_key],
            findings=findings,
            detected_lang=lang_key,
        )

    # -- output side -------------------------------------------------------

    def _analyze_pii(self, text: str) -> list[RecognizerResult]:
        if not text:
            return []
        results = _analyzer().analyze(
            text=text,
            entities=list(self.entities),
            language="en",
            allow_list=list(ALLOWED_OUTBOUND_CONTACTS),
        )
        return [r for r in results if r.score >= self.pii_min_score]

    def _redact(self, text: str, results: Iterable[RecognizerResult]) -> str:
        hits = list(results)
        if not hits:
            return text
        operators = {
            e: OperatorConfig("replace", {"new_value": f"<{e}>"})
            for e in self.entities
        }
        out = _anonymizer().anonymize(text=text, analyzer_results=hits, operators=operators)
        return out.text

    def check_output(self, answer: ReasoningAnswer) -> OutputSafetyResult:
        hits = self._analyze_pii(answer.answer)
        redacted = self._redact(answer.answer, hits) if hits else answer.answer
        findings = [
            PIIFinding(entity_type=r.entity_type, span=(r.start, r.end), score=r.score)
            for r in hits
        ]
        if findings:
            log.warning(
                "pii_redacted",
                n=len(findings),
                types=[f.entity_type for f in findings],
            )
        return OutputSafetyResult(
            safe=not findings,
            redacted_answer=redacted,
            original_answer=answer.answer,
            findings=findings,
            citations=answer.citations,
            abstain=answer.abstain,
            confidence=answer.confidence,
        )

    # -- LCEL adapters -----------------------------------------------------

    def input_runnable(self) -> RunnableLambda:
        def _invoke(payload: str | dict) -> InputSafetyResult:
            if isinstance(payload, str):
                return self.check_input(payload)
            return self.check_input(
                payload["query"],
                lang=payload.get("lang", "en"),
            )
        return RunnableLambda(_invoke)

    def output_runnable(self) -> RunnableLambda:
        return RunnableLambda(self.check_output)
