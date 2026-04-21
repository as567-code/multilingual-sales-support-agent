"""Shared evaluation harness — loads the gold set and iterates samples
through the 3-agent orchestrator, writing per-query results to JSONL
for downstream scoring.

The runners (``run_accuracy``, ``run_hallucination``) compose this to
produce task-specific metrics. Safety (``run_safety``) doesn't need the
orchestrator and runs its own pure-local loop.

Design notes:
  * Offline mode swaps the reasoning LLM with a deterministic stub that
    echoes the gold answer + cites retrieved IDs. This lets CI exercise
    the full plumbing without burning Mistral credits, and gives a
    clean "upper bound" reference for comparison.
  * ``--limit`` / ``--sample`` keep real-LLM runs cheap. Default is
    30 per language (90 total) — enough to see P50/P95 stabilize.
  * Latency buckets are captured per-stage so the report can break
    out retrieval/reasoning/safety separately.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator

from app.agents.reasoning import ReasoningAgent, ReasoningAnswer
from app.agents.retrieval import RetrievalAgent, RetrievalResult
from app.agents.safety import SafetyAgent
from app.chains.orchestrator import AssistantResponse, SupportOrchestrator
from app.utils.logging import get_logger

log = get_logger("eval")

GOLD_DIR = Path("data/eval")
RESULTS_DIR = Path("evals/results")


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------


@dataclass
class GoldSample:
    id: str
    lang: str
    query: str
    gold_answer: str
    relevant_faq_ids: list[str]
    category: str = ""
    difficulty: str = ""


def load_gold(langs: Iterable[str] = ("en", "es", "hi")) -> list[GoldSample]:
    samples: list[GoldSample] = []
    for lang in langs:
        path = GOLD_DIR / f"gold_qa_{lang}.jsonl"
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                samples.append(GoldSample(**obj))
    return samples


def stratified_sample(samples: list[GoldSample], n_per_lang: int, seed: int = 42) -> list[GoldSample]:
    """Pick ``n_per_lang`` random samples per language (stratified)."""
    rng = random.Random(seed)
    by_lang: dict[str, list[GoldSample]] = {}
    for s in samples:
        by_lang.setdefault(s.lang, []).append(s)
    out: list[GoldSample] = []
    for lang, pool in by_lang.items():
        out.extend(rng.sample(pool, min(n_per_lang, len(pool))))
    return out


# --------------------------------------------------------------------------
# offline mode — deterministic fake LLM
# --------------------------------------------------------------------------


def _gold_lookup(gold: list[GoldSample]) -> dict[str, GoldSample]:
    return {g.query: g for g in gold}


def make_offline_orchestrator(gold: list[GoldSample]) -> SupportOrchestrator:
    """Build an orchestrator whose ReasoningAgent uses a fake LLM that returns
    the gold answer + cites whichever of its relevant IDs the retriever surfaced.

    This gives us an "oracle reasoning" ceiling: any drop in citation-F1 is
    attributable to retrieval, not reasoning.
    """
    lookup = _gold_lookup(gold)

    def fake_llm(*, system: str, user: str, temperature: float, max_tokens: int) -> dict:
        # The user prompt template (app/prompts/reasoning_zero_shot.yaml) is:
        #   User question ({lang}):
        #   {query}
        #
        #   Context (retrieved FAQs, ...):
        #   [1] <faq-id> (<cat>): Q: ...
        # Find our gold query by substring match — the rendered prompt contains
        # it verbatim between the header and the blank line before "Context".
        g = next((s for s in lookup.values() if s.query in user), None)
        if g is None:
            return {"answer": "I don't know.", "citations": [], "abstain": True, "confidence": 0.0}
        # FAQ IDs appear in the context as "[<n>] <faq-id> (<category>):".
        cited = [cid for cid in g.relevant_faq_ids if cid in user]
        if not cited:
            return {"answer": "I don't know.", "citations": [], "abstain": True, "confidence": 0.0}
        return {
            "answer": g.gold_answer,
            "citations": cited,
            "abstain": False,
            "confidence": 0.95,
        }

    retrieval = RetrievalAgent()
    reasoning = ReasoningAgent(llm=fake_llm)
    safety = SafetyAgent()
    return SupportOrchestrator(retrieval=retrieval, reasoning=reasoning, safety=safety)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


@dataclass
class QueryRecord:
    id: str
    lang: str
    query: str
    gold_answer: str
    gold_citations: list[str]
    pred_answer: str
    pred_citations: list[str]
    pred_lang: str
    abstain: bool
    injection_detected: bool
    pii_redacted: bool
    confidence: float
    latencies: dict[str, float]


def run_samples(
    orch: SupportOrchestrator,
    samples: list[GoldSample],
) -> Iterator[QueryRecord]:
    for s in samples:
        res: AssistantResponse = orch.ask(s.query, lang_hint=s.lang)
        yield QueryRecord(
            id=s.id,
            lang=s.lang,
            query=s.query,
            gold_answer=s.gold_answer,
            gold_citations=list(s.relevant_faq_ids),
            pred_answer=res.answer,
            pred_citations=list(res.citations),
            pred_lang=res.lang,
            abstain=res.abstain,
            injection_detected=res.injection_detected,
            pii_redacted=res.pii_redacted,
            confidence=res.confidence,
            latencies=res.latencies.model_dump(),
        )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def citation_prf(pred: Iterable[str], gold: Iterable[str]) -> tuple[float, float, float]:
    """Per-query citation precision/recall/F1 (set-based)."""
    p, g = set(pred), set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p:
        return 0.0, 0.0, 0.0
    tp = len(p & g)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


@dataclass
class LatencyBucket:
    p50: float = 0.0
    p95: float = 0.0
    mean: float = 0.0


def latency_stats(records: list[QueryRecord], key: str) -> LatencyBucket:
    xs = [r.latencies.get(key, 0.0) for r in records]
    if not xs:
        return LatencyBucket()
    return LatencyBucket(p50=percentile(xs, 0.5), p95=percentile(xs, 0.95), mean=mean(xs))


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_jsonl(path: Path, records: list[QueryRecord]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_summary(path: Path, summary: dict) -> None:
    with path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def stamp() -> dict[str, str | float]:
    return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "unix": time.time()}
