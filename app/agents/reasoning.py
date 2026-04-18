"""Reasoning Agent — stage 2 of the 3-agent pipeline.

Consumes a ``RetrievalResult`` from Stage 3 and produces a grounded,
language-matched answer with explicit citations via Mistral Small in JSON
mode.

Guarantees (enforced by prompt + pydantic validation + abstention short-circuit):
  * Every answer cites at least one retrieved FAQ OR has ``abstain=True``.
  * Answer language matches the query language (tested in the prompt; not
    enforced mechanically — the eval harness in Stage 9 measures drift).
  * Zero LLM calls when retrieval already abstained (saves tokens + latency).
  * Pure-JSON output: an SDKError bubbles up rather than a best-effort parse.

Prompt variant is pluggable (zero-shot by default, few-shot for the Stage-10
ablation) via the ``prompt_variant`` constructor arg.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

import yaml
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field, field_validator

from app.agents.retrieval import RetrievalHit, RetrievalResult
from app.config import Settings, get_settings
from app.utils.logging import get_logger
from app.utils.mistral_client import chat_json

log = get_logger("reasoning")

PROMPTS_DIR = Path("app/prompts")
PromptVariant = Literal["zero_shot", "few_shot"]

# Canonical abstention strings per language, used when retrieval itself abstains.
# Matches the strings in data/eval (so eval accuracy is well-defined).
_ABSTAIN_ANSWER = {
    "en": "I don't have enough information to answer that. Please contact support@technova.com for help.",
    "es": "No tengo suficiente información para responder. Por favor, contacta con support@technova.com para ayuda.",
    "hi": "क्षमा करें, इस विषय पर मेरे पास पर्याप्त जानकारी नहीं है। कृपया सहायता के लिए support@technova.com से संपर्क करें।",
}


class ReasoningAnswer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    abstain: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("citations")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        return [c.strip() for c in v if c and c.strip()]


@lru_cache(maxsize=4)
def _load_prompt(variant: PromptVariant) -> tuple[str, str]:
    path = PROMPTS_DIR / f"reasoning_{variant}.yaml"
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc["system"], doc["user"]


def _format_context(hits: list[RetrievalHit]) -> str:
    if not hits:
        return "(no FAQs retrieved)"
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(
            f"[{i}] {h.faq_id} ({h.category}): Q: {h.question}\nA: {h.answer}"
        )
    return "\n\n".join(lines)


def _filter_citations(cites: list[str], hits: list[RetrievalHit]) -> list[str]:
    """Drop citations that don't match any retrieved FAQ ID — anti-fabrication guard."""
    allowed = {h.faq_id for h in hits}
    kept, dropped = [], []
    for c in cites:
        (kept if c in allowed else dropped).append(c)
    if dropped:
        log.warning("dropped_hallucinated_citations", dropped=dropped, allowed=list(allowed))
    return kept


# Test/DI seam: the agent calls through this function. Swap it in tests.
LLMFn = Callable[..., dict]


@dataclass
class ReasoningAgent:
    settings: Settings | None = None
    prompt_variant: PromptVariant = "zero_shot"
    temperature: float = 0.2
    max_tokens: int = 1024
    llm: LLMFn = field(default=chat_json)

    def __post_init__(self) -> None:
        self.settings = self.settings or get_settings()
        self._system_tpl, self._user_tpl = _load_prompt(self.prompt_variant)

    # -- core entry point --------------------------------------------------

    def answer(self, retrieval: RetrievalResult) -> ReasoningAnswer:
        # Short-circuit: retrieval already abstained → canonical apology, no LLM call.
        if retrieval.abstain or not retrieval.hits:
            lang = retrieval.detected_lang if retrieval.detected_lang in _ABSTAIN_ANSWER else "en"
            log.info("reasoning_abstain_short_circuit", lang=lang, query=retrieval.query[:80])
            return ReasoningAnswer(
                answer=_ABSTAIN_ANSWER[lang],
                citations=[],
                abstain=True,
                confidence=0.0,
            )

        system = self._system_tpl.format(lang=retrieval.detected_lang)
        user = self._user_tpl.format(
            lang=retrieval.detected_lang,
            query=retrieval.query,
            context=_format_context(retrieval.hits),
        )

        raw = self.llm(
            system=system,
            user=user,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        parsed = ReasoningAnswer.model_validate(raw)
        parsed.citations = _filter_citations(parsed.citations, retrieval.hits)

        # Defensive: if model claims non-abstain but returned no valid citations,
        # treat as abstain — we'd rather say "I don't know" than speculate.
        if not parsed.abstain and not parsed.citations:
            log.warning("promoting_to_abstain_no_citations", query=retrieval.query[:80])
            lang = retrieval.detected_lang if retrieval.detected_lang in _ABSTAIN_ANSWER else "en"
            return ReasoningAnswer(
                answer=_ABSTAIN_ANSWER[lang],
                citations=[],
                abstain=True,
                confidence=0.0,
            )

        log.info(
            "reasoning_done",
            query=retrieval.query[:80],
            n_citations=len(parsed.citations),
            abstain=parsed.abstain,
            confidence=parsed.confidence,
        )
        return parsed

    # -- LCEL adapter ------------------------------------------------------

    def as_runnable(self) -> RunnableLambda:
        return RunnableLambda(self.answer)
