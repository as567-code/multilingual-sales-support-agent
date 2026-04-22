"""LangChain + FAISS retrieval benchmark — our production baseline.

This one doesn't technically need LangChain (we use FAISS + sentence-
transformers directly in the RetrievalAgent), but the naming reflects the
orchestration choice. The bench rebuilds an index in-memory from the same
chunks rather than mmap'ing the persisted one, so index_ms is comparable
against the other frameworks.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from evals.framework_benchmark.common import (
    Bench,
    ChunkRecord,
    load_chunks,
    load_gold,
    run_bench,
    write_result,
)

MODEL_NAME = "intfloat/multilingual-e5-base"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


class LangChainFAISSBench:
    name = "langchain_faiss"

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.ids: list[str] = []

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        texts = [f"passage: {c.text}" for c in chunks]
        self.ids = [c.id for c in chunks]
        emb = _model().encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb.astype(np.float32))

    def search(self, query: str, k: int) -> list[str]:
        assert self.index is not None
        q = _model().encode([f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        _, idx = self.index.search(q.astype(np.float32), k)
        return [self.ids[i] for i in idx[0] if i >= 0]

    def close(self) -> None:
        self.index = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks()
    gold = load_gold()
    bench: Bench = LangChainFAISSBench()
    result = run_bench(bench, chunks, gold, k=args.k)
    path = write_result(result)
    print(f"{bench.name}: recall@{args.k}={result.recall_at_k:.3f} mrr={result.mrr:.3f} qps={result.qps:.1f} → {path}")


if __name__ == "__main__":
    main()
