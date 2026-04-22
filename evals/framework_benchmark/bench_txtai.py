"""txtai retrieval benchmark.

Requires: ``pip install '.[txtai]'``

txtai wraps transformers + FAISS behind a single Embeddings class. We
point it at the same e5 multilingual model and its default Faiss backend
for direct comparison.
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


class TxtaiBench:
    name = "txtai"

    def __init__(self) -> None:
        self.emb = None
        self.id_map: list[str] = []

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        from txtai import Embeddings  # type: ignore

        # method="sentence-transformers" gives us the same e5 backbone.
        self.emb = Embeddings({
            "path": "intfloat/multilingual-e5-base",
            "method": "sentence-transformers",
            "content": False,
            "normalize": True,
        })
        self.id_map = [c.id for c in chunks]
        # txtai's index() takes (uid, text, tags) tuples; we key by list index
        # and resolve to faq_id via id_map on lookup.
        self.emb.index(
            (i, f"passage: {c.text}", None) for i, c in enumerate(chunks)
        )

    def search(self, query: str, k: int) -> list[str]:
        assert self.emb is not None
        hits = self.emb.search(f"query: {query}", k)
        # hits is list of (uid, score)
        return [self.id_map[int(uid)] for uid, _ in hits]

    def close(self) -> None:
        self.emb = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks()
    gold = load_gold()
    bench: Bench = TxtaiBench()
    result = run_bench(bench, chunks, gold, k=args.k)
    path = write_result(result)
    msg = f"recall@{args.k}={result.recall_at_k:.3f} mrr={result.mrr:.3f} qps={result.qps:.1f}"
    if result.error:
        msg = f"ERROR: {result.error}"
    print(f"{bench.name}: {msg} → {path}")


if __name__ == "__main__":
    main()
