"""Stage-3 Retrieval Agent tests.

Complements ``test_retrieval.py`` (Stage-2 index-level recall gate). These
tests cover the agent-level behaviors: language detection, score thresholding,
abstention, typed output, and the LCEL Runnable wrapper.
"""
from __future__ import annotations

import pytest

from app.agents.retrieval import RetrievalAgent, RetrievalHit, RetrievalResult
from app.utils.language import detect_language


@pytest.fixture(scope="module")
def agent() -> RetrievalAgent:
    # Module-scoped so the model + index load once.
    return RetrievalAgent()


# ---- language detection ---------------------------------------------------


@pytest.mark.parametrize(
    "text,expected,source",
    [
        ("How do I cancel my subscription?", "en", "langdetect"),
        ("¿Cómo actualizo mi método de pago?", "es", "langdetect"),
        ("मेरा ऑर्डर कब आएगा?", "hi", "devanagari"),
        # Latin-script mention of TechNova still Hindi by script
        ("TechNova खाता बंद क्यों हो गया है?", "hi", "devanagari"),
        ("", "en", "default"),
    ],
)
def test_detect_language(text, expected, source):
    g = detect_language(text)
    assert g.lang == expected
    assert g.source == source


# ---- retrieval, per language ----------------------------------------------


@pytest.mark.parametrize(
    "query,expected_lang,expected_faq",
    [
        ("How can I cancel my subscription?", "en", "faq-account-024"),
        ("¿Cómo actualizo mi método de pago?", "es", "faq-account-019"),
        ("मेरा ऑर्डर कब आएगा?", "hi", "faq-shipping-004"),
    ],
)
def test_retrieve_hits(agent, query, expected_lang, expected_faq):
    res = agent.retrieve(query)
    assert isinstance(res, RetrievalResult)
    assert res.detected_lang == expected_lang
    assert not res.abstain, "should not abstain on an in-domain query"
    assert 0 < len(res.hits) <= agent.top_k
    assert all(isinstance(h, RetrievalHit) for h in res.hits)
    assert all(h.score >= agent.min_score for h in res.hits)
    top_ids = [h.faq_id for h in res.hits]
    assert expected_faq in top_ids, (
        f"expected {expected_faq} in top-{agent.top_k}, got {top_ids}"
    )


def test_scores_monotonic(agent):
    res = agent.retrieve("How do I request a refund?")
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


# ---- thresholding + abstention --------------------------------------------


def test_high_threshold_forces_abstain(agent):
    # All real hits sit in 0.8x; 0.99 is unreachable → agent must abstain.
    a = RetrievalAgent(min_score=0.99)
    res = a.retrieve("How do I cancel my subscription?")
    assert res.abstain is True
    assert res.hits == []


def test_empty_query_abstains(agent):
    res = agent.retrieve("   ")
    assert res.abstain is True
    assert res.hits == []


def test_ood_query_behaviour(agent):
    # Clearly out-of-domain. We don't require abstention at the default 0.70
    # threshold (borderline OOD like laptop-brand questions can still match
    # loosely), but we do require the top score to be meaningfully lower than
    # an in-domain query.
    ood = agent.retrieve("What's the weather in Tokyo tomorrow?")
    in_domain = agent.retrieve("How do I cancel my subscription?")
    top_ood = ood.hits[0].score if ood.hits else 0.0
    top_in = in_domain.hits[0].score
    assert top_in - top_ood > 0.05, (
        f"expected in-domain top score to dominate OOD, got in={top_in:.3f} ood={top_ood:.3f}"
    )


# ---- lang hint override ---------------------------------------------------


def test_lang_hint_overrides_detection(agent):
    # langdetect would call this EN; the hint wins and is recorded.
    res = agent.retrieve("TechNova subscription?", lang_hint="es")
    assert res.detected_lang == "es"
    assert res.lang_source == "hint"


# ---- LCEL runnable --------------------------------------------------------


def test_as_runnable_with_string(agent):
    runnable = agent.as_runnable()
    res = runnable.invoke("How do I cancel my subscription?")
    assert isinstance(res, RetrievalResult)
    assert not res.abstain


def test_as_runnable_with_dict(agent):
    runnable = agent.as_runnable()
    res = runnable.invoke({"query": "TechNova plan?", "lang_hint": "en"})
    assert isinstance(res, RetrievalResult)
    assert res.lang_source == "hint"
