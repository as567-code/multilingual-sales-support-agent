"""Haystack 2.x retrieval benchmark.

Requires: ``pip install '.[haystack]'``

Uses Haystack's InMemoryDocumentStore + SentenceTransformersTextEmbedder
(same e5 multilingual model) + InMemoryEmbeddingRetriever. Single-threaded,
CPU-only, matches the baseline configuration.
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


class HaystackBench:
    name = "haystack"

    def __init__(self) -> None:
        self.store = None
        self.text_embedder = None
        self.retriever = None

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        from haystack import Document  # type: ignore
        from haystack.components.embedders import (  # type: ignore
            SentenceTransformersDocumentEmbedder,
            SentenceTransformersTextEmbedder,
        )
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever  # type: ignore
        from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore

        self.store = InMemoryDocumentStore()
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model="intfloat/multilingual-e5-base",
            prefix="passage: ",
        )
        doc_embedder.warm_up()

        docs = [Document(content=c.text, meta={"faq_id": c.id}) for c in chunks]
        embedded = doc_embedder.run(documents=docs)["documents"]
        self.store.write_documents(embedded)

        self.text_embedder = SentenceTransformersTextEmbedder(
            model="intfloat/multilingual-e5-base",
            prefix="query: ",
        )
        self.text_embedder.warm_up()
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.store, top_k=5)

    def search(self, query: str, k: int) -> list[str]:
        assert self.text_embedder is not None and self.retriever is not None
        q_emb = self.text_embedder.run(text=query)["embedding"]
        self.retriever.top_k = k
        docs = self.retriever.run(query_embedding=q_emb)["documents"]
        return [d.meta.get("faq_id") for d in docs if d.meta.get("faq_id")]

    def close(self) -> None:
        self.store = None
        self.text_embedder = None
        self.retriever = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks()
    gold = load_gold()
    bench: Bench = HaystackBench()
    result = run_bench(bench, chunks, gold, k=args.k)
    path = write_result(result)
    msg = f"recall@{args.k}={result.recall_at_k:.3f} mrr={result.mrr:.3f} qps={result.qps:.1f}"
    if result.error:
        msg = f"ERROR: {result.error}"
    print(f"{bench.name}: {msg} → {path}")


if __name__ == "__main__":
    main()
