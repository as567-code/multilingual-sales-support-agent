"""Stage-2 retrieval correctness gate.

Two gates:

* ``test_smoke`` — 9 hand-picked gold queries (3 per language) must hit their
  gold FAQ inside top-5 on the unified multilingual index. A regression here
  usually means the e5 ``passage:``/``query:`` prefixing or the L2 normalization
  was dropped.
* ``test_recall_at_5`` — random 30-query sample (10 per language) from the gold
  set must have recall@5 ≥ 0.85. This is the PRD's minimum bar before Stage 3.

The Retrieval *agent* itself (filters, thresholds, LangChain wrapping) is
implemented and tested in Stage 3.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import faiss  # type: ignore[import-untyped]
import pytest
from sentence_transformers import SentenceTransformer

from ingest.build_faiss import MODEL_NAME, encode_query

PROCESSED = Path("data/processed")
GOLD = Path("data/eval")

# 3 queries per language, sourced from data/eval/gold_qa_*.jsonl.
# Each (lang, query, expected_faq_id) triple must hit top-5 on faiss_all.
SMOKE_CASES: list[tuple[str, str, str]] = [
    # EN — direct paraphrase, direct paraphrase, direct paraphrase
    ("en", "Is the TechNova SmartDesk Pro compatible with wireless charging?", "faq-product-002"),
    ("en", "How can I permanently delete my TechNova account?", "faq-account-013"),
    ("en", "What steps do I take to reactivate a canceled TechNova subscription?", "faq-account-025"),
    # ES
    ("es", "¿Cómo actualizo el firmware de mi TechNova SmartPen?", "faq-product-015"),
    ("es", "¿Puedo devolver un producto TechNova si ya no lo quiero?", "faq-returns-010"),
    ("es", "¿Cómo pido un reembolso por un hardware de TechNova que compré?", "faq-billing-015"),
    # HI
    ("hi", "मेरा TechNova खाता बंद क्यों हो गया है?", "faq-account-016"),
    ("hi", "TechNova खरीद के दौरान प्रमोशन कोड कैसे डालें?", "faq-sales-002"),
    ("hi", "TechNova वापसी शिपमेंट को कैसे ट्रैक करें?", "faq-returns-015"),
]


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@pytest.fixture(scope="module")
def index_and_meta() -> tuple[faiss.IndexFlatIP, list[dict]]:
    idx = faiss.read_index(str(PROCESSED / "faiss_all.index"))
    meta = list(_iter_jsonl(PROCESSED / "metadata.jsonl"))
    assert idx.ntotal == len(meta), "index ↔ metadata row count mismatch"
    return idx, meta


@pytest.fixture(scope="module")
def model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def _topk_ids(
    model: SentenceTransformer,
    idx: faiss.IndexFlatIP,
    meta: list[dict],
    query: str,
    k: int,
) -> list[str]:
    qv = encode_query(model, query)
    _scores, ids = idx.search(qv, k)
    return [meta[int(i)]["id"] for i in ids[0]]


@pytest.mark.parametrize("lang,query,expected", SMOKE_CASES, ids=[f"{c[0]}-{c[2]}" for c in SMOKE_CASES])
def test_smoke(index_and_meta, model, lang, query, expected):
    idx, meta = index_and_meta
    top5 = _topk_ids(model, idx, meta, query, k=5)
    assert expected in top5, (
        f"[{lang}] expected {expected} in top-5 for query {query!r}, got {top5}"
    )


def test_recall_at_5(index_and_meta, model):
    idx, meta = index_and_meta
    rng = random.Random(42)

    per_lang = 10
    sample: list[dict] = []
    for lang in ("en", "es", "hi"):
        rows = [
            r for r in _iter_jsonl(GOLD / f"gold_qa_{lang}.jsonl")
            if r["relevant_faq_ids"]  # drop OOD — no gold FAQ to retrieve
        ]
        sample.extend(rng.sample(rows, per_lang))

    hits = 0
    misses: list[tuple[str, str, list[str], list[str]]] = []
    for r in sample:
        top5 = _topk_ids(model, idx, meta, r["query"], k=5)
        if any(fid in top5 for fid in r["relevant_faq_ids"]):
            hits += 1
        else:
            misses.append((r["lang"], r["query"], r["relevant_faq_ids"], top5))

    recall = hits / len(sample)
    print(f"\nrecall@5 = {recall:.3f} ({hits}/{len(sample)})")
    for lang, q, gold, got in misses:
        print(f"  MISS [{lang}] {q!r} gold={gold} top5={got}")

    assert recall >= 0.85, f"recall@5 {recall:.3f} below 0.85 threshold"
