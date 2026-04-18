"""Stage-4 Reasoning Agent tests.

Unit tests (fast, deterministic) use a fake LLM injected via the ``llm=``
constructor arg. The ``@pytest.mark.integration`` tests hit the real
Mistral API and only run when ``MISTRAL_API_KEY`` is set AND the env var
``RUN_MISTRAL_TESTS=1`` is present — so CI stays hermetic.

Run integration tests locally with:
    RUN_MISTRAL_TESTS=1 pytest tests/test_reasoning.py -m integration -v
"""
from __future__ import annotations

import os

import pytest

from app.agents.reasoning import ReasoningAgent, ReasoningAnswer
from app.agents.retrieval import RetrievalAgent, RetrievalHit, RetrievalResult


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _mk_result(query: str, lang: str, hits: list[RetrievalHit], abstain: bool = False) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        detected_lang=lang,
        lang_confidence=1.0,
        lang_source="hint",
        hits=hits,
        abstain=abstain,
    )


def _mk_hit(faq_id: str, lang: str = "en", category: str = "billing") -> RetrievalHit:
    return RetrievalHit(
        faq_id=faq_id,
        lang=lang,
        category=category,
        question=f"Q about {faq_id}",
        answer=f"A about {faq_id}",
        text=f"Q: Q about {faq_id}\nA: A about {faq_id}",
        score=0.85,
    )


class _FakeLLM:
    """Callable with the same signature as chat_json; returns a canned dict."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[dict] = []

    def __call__(self, *, system: str, user: str, temperature: float, max_tokens: int) -> dict:
        self.calls.append(
            {"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.payload


# --------------------------------------------------------------------------
# unit tests — fake LLM
# --------------------------------------------------------------------------


def test_abstain_short_circuit_skips_llm():
    """When retrieval already abstained, we must not call the LLM at all."""
    fake = _FakeLLM({"should": "not be returned"})
    agent = ReasoningAgent(llm=fake)
    result = _mk_result("anything", "en", hits=[], abstain=True)

    ans = agent.answer(result)

    assert isinstance(ans, ReasoningAnswer)
    assert ans.abstain is True
    assert ans.citations == []
    assert fake.calls == [], "LLM should not be called when retrieval abstained"


def test_abstain_language_selects_canonical_string():
    agent = ReasoningAgent(llm=_FakeLLM({}))
    for lang, phrase in [("en", "support@technova.com"), ("es", "support@technova.com"), ("hi", "support@technova.com")]:
        ans = agent.answer(_mk_result("x", lang, hits=[], abstain=True))
        assert phrase in ans.answer, f"{lang} abstain missing contact: {ans.answer!r}"


def test_happy_path_returns_parsed_answer():
    hit = _mk_hit("faq-billing-001")
    fake = _FakeLLM({
        "answer": "Open Account Settings → Billing.",
        "citations": ["faq-billing-001"],
        "abstain": False,
        "confidence": 0.9,
    })
    agent = ReasoningAgent(llm=fake)

    ans = agent.answer(_mk_result("How do I cancel?", "en", [hit]))

    assert ans.abstain is False
    assert ans.citations == ["faq-billing-001"]
    assert ans.confidence == pytest.approx(0.9)
    assert len(fake.calls) == 1
    # Prompt must have included the retrieved context and user query.
    assert "faq-billing-001" in fake.calls[0]["user"]
    assert "How do I cancel?" in fake.calls[0]["user"]


def test_hallucinated_citation_is_dropped():
    hit = _mk_hit("faq-billing-001")
    fake = _FakeLLM({
        "answer": "Some answer.",
        "citations": ["faq-billing-001", "faq-made-up-999"],
        "abstain": False,
        "confidence": 0.7,
    })
    agent = ReasoningAgent(llm=fake)

    ans = agent.answer(_mk_result("query", "en", [hit]))

    assert ans.citations == ["faq-billing-001"]


def test_all_citations_hallucinated_promotes_to_abstain():
    """Model said 'not abstaining' but all citations are invented → promote to abstain."""
    hit = _mk_hit("faq-billing-001")
    fake = _FakeLLM({
        "answer": "Trust me, here is the answer.",
        "citations": ["faq-hallucinated-1", "faq-hallucinated-2"],
        "abstain": False,
        "confidence": 0.95,
    })
    agent = ReasoningAgent(llm=fake)

    ans = agent.answer(_mk_result("query", "en", [hit]))

    assert ans.abstain is True
    assert ans.citations == []
    assert "support@technova.com" in ans.answer


def test_invalid_llm_json_raises():
    """Model returns something that doesn't match the schema → pydantic errors."""
    hit = _mk_hit("faq-billing-001")
    fake = _FakeLLM({"wrong": "shape"})
    agent = ReasoningAgent(llm=fake)

    with pytest.raises(Exception):  # pydantic ValidationError
        agent.answer(_mk_result("query", "en", [hit]))


def test_as_runnable_invocation():
    hit = _mk_hit("faq-account-024")
    fake = _FakeLLM({
        "answer": "Go to Billing.",
        "citations": ["faq-account-024"],
        "abstain": False,
        "confidence": 0.8,
    })
    agent = ReasoningAgent(llm=fake)
    runnable = agent.as_runnable()

    ans = runnable.invoke(_mk_result("cancel?", "en", [hit]))

    assert isinstance(ans, ReasoningAnswer)
    assert ans.citations == ["faq-account-024"]


def test_prompt_variant_loads_few_shot():
    agent = ReasoningAgent(prompt_variant="few_shot", llm=_FakeLLM({
        "answer": "x", "citations": ["faq-billing-001"], "abstain": False, "confidence": 0.5,
    }))
    agent.answer(_mk_result("q", "en", [_mk_hit("faq-billing-001")]))
    # No explicit assertion beyond "loading didn't error"; detailed prompt
    # diffing would be brittle.


# --------------------------------------------------------------------------
# integration tests — real Mistral API
# --------------------------------------------------------------------------


def _has_mistral_key() -> bool:
    # .env is picked up by pydantic-settings; shell env alone isn't enough.
    from app.config import get_settings
    return bool(get_settings().mistral_api_key)


_SKIP_INTEGRATION = pytest.mark.skipif(
    not (_has_mistral_key() and os.environ.get("RUN_MISTRAL_TESTS") == "1"),
    reason="Set RUN_MISTRAL_TESTS=1 (and MISTRAL_API_KEY in .env) to run live-API tests.",
)


@pytest.fixture(scope="module")
def _retrieval_agent() -> RetrievalAgent:
    return RetrievalAgent()


@pytest.mark.integration
@_SKIP_INTEGRATION
def test_real_mistral_en_answers_with_citation(_retrieval_agent):
    retrieval = _retrieval_agent.retrieve("How do I cancel my TechNova subscription?")
    assert not retrieval.abstain

    agent = ReasoningAgent()
    ans = agent.answer(retrieval)

    assert ans.abstain is False
    assert ans.citations, "real model should cite at least one FAQ"
    assert all(c.startswith("faq-") for c in ans.citations)
    assert len(ans.answer) > 10


@pytest.mark.integration
@_SKIP_INTEGRATION
def test_real_mistral_hi_language_preserved(_retrieval_agent):
    retrieval = _retrieval_agent.retrieve("मेरा ऑर्डर कब आएगा?")
    assert not retrieval.abstain
    assert retrieval.detected_lang == "hi"

    agent = ReasoningAgent()
    ans = agent.answer(retrieval)

    assert ans.abstain is False
    # Hindi answer must contain Devanagari characters.
    assert any("\u0900" <= ch <= "\u097f" for ch in ans.answer), (
        f"expected Devanagari in HI answer, got: {ans.answer!r}"
    )
