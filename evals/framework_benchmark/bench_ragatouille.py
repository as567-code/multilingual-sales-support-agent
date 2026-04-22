"""RAGatouille (ColBERTv2) retrieval benchmark.

Requires: ``pip install '.[ragatouille]'``

ColBERT uses late-interaction — each token gets its own embedding and
relevance is max-sim over pairs. This gives stronger retrieval in
English but has well-known drops on non-Latin scripts (Hindi in our
case) unless a multilingual checkpoint is used. We use the public
``colbert-ir/colbertv2.0`` model as a portfolio-time baseline; the
Hindi drop shows up in per_lang breakdowns.
"""
from __future__ import annotations

import argparse

from evals.framework_benchmark.common import (
    Bench,
    ChunkRecord,
    load_chunks,
    load_gold,
    run_bench,
    write_result,
)


class RAGatouilleBench:
    name = "ragatouille"

    def __init__(self) -> None:
        self.rag = None
        self.id_map: list[str] = []

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        from ragatouille import RAGPretrainedModel  # type: ignore

        self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        # Index name must be filesystem-safe; ColBERT persists to disk.
        self.id_map = [c.id for c in chunks]
        self.rag.index(
            index_name="techfnova_bench",
            collection=[c.text for c in chunks],
            document_ids=[c.id for c in chunks],
            max_document_length=256,
            split_documents=False,
        )

    def search(self, query: str, k: int) -> list[str]:
        assert self.rag is not None
        hits = self.rag.search(query=query, k=k)
        return [h.get("document_id") for h in hits if h.get("document_id")]

    def close(self) -> None:
        self.rag = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks()
    gold = load_gold()
    bench: Bench = RAGatouilleBench()
    result = run_bench(bench, chunks, gold, k=args.k)
    path = write_result(result)
    msg = f"recall@{args.k}={result.recall_at_k:.3f} mrr={result.mrr:.3f} qps={result.qps:.1f}"
    if result.error:
        msg = f"ERROR: {result.error}"
    print(f"{bench.name}: {msg} → {path}")


if __name__ == "__main__":
    main()
