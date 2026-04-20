"""Multi-agent Orchestrator — the Stage-6 entry point for a full request.

Composes the 3-agent pipeline in the order mandated by the PRD:

    query ──▶ Safety.check_input
              │
              ├── (unsafe) ──▶ canonical refusal ────────────────────┐
              │                                                       │
              └── (safe)   ──▶ Retrieval.retrieve ──▶ Reasoning.answer│
                                                           │          │
                                                           ▼          │
                                                    Safety.check_output
                                                           │          │
                                                           ▼          │
                                                    AssistantResponse ◀┘

One public method, ``ask(query, lang_hint=None)``. Also exposed as an LCEL
Runnable so an HTTP layer can plug it in without reaching into the class.

The orchestrator measures wall-clock latency per stage so the Stage-9 eval
harness can report P50/P95 directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from app.agents.reasoning import ReasoningAgent, ReasoningAnswer
from app.agents.retrieval import RetrievalAgent, RetrievalResult
from app.agents.safety import OutputSafetyResult, SafetyAgent
from app.config import Settings, get_settings
from app.utils.language import detect_language
from app.utils.logging import get_logger

log = get_logger("orchestrator")


class StageLatencies(BaseModel):
    safety_input_ms: float = 0.0
    retrieval_ms: float = 0.0
    reasoning_ms: float = 0.0
    safety_output_ms: float = 0.0
    total_ms: float = 0.0


class AssistantResponse(BaseModel):
    """Final shape returned to the API/CLI. All fields are populated on every path."""
    query: str
    answer: str  # redacted + safe, or refusal, or abstention
    lang: str
    citations: list[str] = Field(default_factory=list)
    abstain: bool = False
    confidence: float = 0.0
    injection_detected: bool = False
    pii_redacted: bool = False
    latencies: StageLatencies = Field(default_factory=StageLatencies)


@dataclass
class SupportOrchestrator:
    settings: Settings | None = None
    retrieval: RetrievalAgent | None = None
    reasoning: ReasoningAgent | None = None
    safety: SafetyAgent | None = None

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        # Lazy defaults so tests can inject mocks per-field.
        if self.retrieval is None:
            self.retrieval = RetrievalAgent(settings=self.settings)
        if self.reasoning is None:
            self.reasoning = ReasoningAgent(settings=self.settings)
        if self.safety is None:
            self.safety = SafetyAgent(settings=self.settings)

    # -- main entry --------------------------------------------------------

    def ask(self, query: str, *, lang_hint: str | None = None) -> AssistantResponse:
        t_all = time.perf_counter()
        lat = StageLatencies()

        q = (query or "").strip()
        lang = lang_hint or detect_language(q).lang

        # 1) Safety on the way in.
        t = time.perf_counter()
        assert self.safety is not None
        in_safe = self.safety.check_input(q, lang=lang)
        lat.safety_input_ms = (time.perf_counter() - t) * 1000

        if not in_safe.safe:
            lat.total_ms = (time.perf_counter() - t_all) * 1000
            log.info("orchestrator_blocked_injection",
                     query=q[:80], n_findings=len(in_safe.findings),
                     total_ms=lat.total_ms)
            return AssistantResponse(
                query=q, answer=in_safe.refusal or "",
                lang=in_safe.detected_lang,
                citations=[], abstain=True, confidence=0.0,
                injection_detected=True, pii_redacted=False,
                latencies=lat,
            )

        # 2) Retrieval.
        t = time.perf_counter()
        assert self.retrieval is not None
        retrieval: RetrievalResult = self.retrieval.retrieve(q, lang_hint=lang_hint)
        lat.retrieval_ms = (time.perf_counter() - t) * 1000

        # 3) Reasoning (short-circuits internally on abstain → no LLM call).
        t = time.perf_counter()
        assert self.reasoning is not None
        reasoning: ReasoningAnswer = self.reasoning.answer(retrieval)
        lat.reasoning_ms = (time.perf_counter() - t) * 1000

        # 4) Safety on the way out.
        t = time.perf_counter()
        out: OutputSafetyResult = self.safety.check_output(reasoning)
        lat.safety_output_ms = (time.perf_counter() - t) * 1000

        lat.total_ms = (time.perf_counter() - t_all) * 1000
        log.info(
            "orchestrator_done",
            lang=retrieval.detected_lang,
            abstain=out.abstain,
            n_citations=len(out.citations),
            pii_redacted=not out.safe,
            total_ms=lat.total_ms,
        )

        return AssistantResponse(
            query=q,
            answer=out.redacted_answer,
            lang=retrieval.detected_lang,
            citations=out.citations,
            abstain=out.abstain,
            confidence=out.confidence,
            injection_detected=False,
            pii_redacted=not out.safe,
            latencies=lat,
        )

    # -- LCEL adapter ------------------------------------------------------

    def as_runnable(self) -> RunnableLambda:
        def _invoke(payload: str | dict[str, Any]) -> AssistantResponse:
            if isinstance(payload, str):
                return self.ask(payload)
            return self.ask(payload["query"], lang_hint=payload.get("lang_hint"))
        return RunnableLambda(_invoke)
