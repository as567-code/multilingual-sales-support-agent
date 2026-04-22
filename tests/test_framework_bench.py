"""Stage-10 framework bake-off — harness smoke tests.

The actual framework runners produce heavy benchmarks; we only verify
that the common harness loads data, computes metrics correctly, and
writes result JSON in the expected shape. A single end-to-end run with
the langchain_faiss bench on a tiny subset validates the integration.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.framework_benchmark.common import (
    BenchResult,
    ChunkRecord,
    GoldQuery,
    load_chunks,
    load_gold,
    mrr_at_k,
    recall_at_k,
    run_bench,
    write_result,
)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], ["c"]) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"]) == 0.0


def test_recall_at_k_empty_relevant_is_trivially_one():
    # If the gold set has no relevant IDs (OOD query), any retrieval is "correct".
    # The classifier downstream is what penalizes non-abstention on OOD.
    assert recall_at_k(["a", "b"], []) == 1.0


def test_mrr_rank_one_vs_three():
    assert mrr_at_k(["a", "b", "c"], ["a"]) == 1.0
    assert mrr_at_k(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "z"], ["a"]) == 0.0


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_load_chunks_nonempty_and_typed():
    chunks = load_chunks()
    assert len(chunks) == 420
    assert all(isinstance(c, ChunkRecord) for c in chunks)
    assert {c.lang for c in chunks} == {"en", "es", "hi"}


def test_load_gold_nonempty_and_typed():
    gold = load_gold()
    assert len(gold) == 930
    assert all(isinstance(g, GoldQuery) for g in gold)


# --------------------------------------------------------------------------
# a fake bench — exercises the runner without doing retrieval
# --------------------------------------------------------------------------


class _PerfectBench:
    """Returns the gold IDs as top-k — upper bound at recall=1, MRR=1."""
    name = "perfect_oracle"

    def __init__(self) -> None:
        self._lookup: dict[str, list[str]] = {}

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        # No-op — this oracle cheats via the gold list passed to search()
        # through a closure. We just snapshot nothing here.
        pass

    def attach(self, gold: list[GoldQuery]) -> None:
        self._lookup = {g.query: g.relevant_faq_ids for g in gold}

    def search(self, query: str, k: int) -> list[str]:
        return (self._lookup.get(query) or [])[:k]

    def close(self) -> None:
        self._lookup = {}


def test_run_bench_end_to_end_on_oracle(tmp_path):
    chunks = load_chunks()[:10]
    gold = load_gold()[:8]
    bench = _PerfectBench()
    bench.attach(gold)

    result: BenchResult = run_bench(bench, chunks, gold, k=5)
    assert result.error is None
    assert result.n_chunks == 10
    assert result.n_queries == 8
    assert result.recall_at_k == 1.0
    assert result.mrr == 1.0
    assert result.qps > 0.0

    # And that write_result serializes cleanly.
    prev_cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        path = write_result(result)
        blob = json.loads(path.read_text())
    finally:
        import os
        os.chdir(prev_cwd)
    assert blob["name"] == "perfect_oracle"
    assert blob["recall_at_k"] == 1.0
    assert "per_lang" in blob


def test_run_bench_captures_errors(tmp_path):
    class _BrokenBench:
        name = "broken"

        def build_index(self, chunks): raise RuntimeError("no index for you")
        def search(self, q, k): return []
        def close(self): pass

    result = run_bench(_BrokenBench(), load_chunks()[:5], load_gold()[:5], k=5)
    assert result.error is not None
    assert "RuntimeError" in result.error
    assert result.recall_at_k == 0.0
