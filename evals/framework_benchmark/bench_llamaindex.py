"""LlamaIndex retrieval benchmark.

Requires: ``pip install '.[llamaindex]'``

Uses LlamaIndex's VectorStoreIndex with a HuggingFace embedding wrapper
pointed at the same e5 multilingual model for apples-to-apples. Storage
is the built-in SimpleVectorStore (in-memory), which mirrors the FAISS-
flat-IP baseline closely.
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


class LlamaIndexBench:
    name = "llamaindex"

    def __init__(self) -> None:
        self.index = None
        self.retriever = None

    def build_index(self, chunks: list[ChunkRecord]) -> None:
        from llama_index.core import Document, VectorStoreIndex, Settings  # type: ignore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore

        Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-base")
        # Disable the default OpenAI LLM; we're only benching retrieval.
        Settings.llm = None

        docs = [
            Document(text=f"passage: {c.text}", metadata={"faq_id": c.id})
            for c in chunks
        ]
        self.index = VectorStoreIndex.from_documents(docs)
        self.retriever = self.index.as_retriever(similarity_top_k=5)

    def search(self, query: str, k: int) -> list[str]:
        assert self.retriever is not None
        self.retriever.similarity_top_k = k
        nodes = self.retriever.retrieve(f"query: {query}")
        return [n.node.metadata.get("faq_id") for n in nodes if n.node.metadata.get("faq_id")]

    def close(self) -> None:
        self.index = None
        self.retriever = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks()
    gold = load_gold()
    bench: Bench = LlamaIndexBench()
    result = run_bench(bench, chunks, gold, k=args.k)
    path = write_result(result)
    msg = f"recall@{args.k}={result.recall_at_k:.3f} mrr={result.mrr:.3f} qps={result.qps:.1f}"
    if result.error:
        msg = f"ERROR: {result.error}"
    print(f"{bench.name}: {msg} → {path}")


if __name__ == "__main__":
    main()
