# RAG Framework Comparison — Stage 10

Live numbers live at
[`evals/results/framework_bakeoff.md`](../evals/results/framework_bakeoff.md);
this document explains methodology, interpretation, and why we keep
LangChain + raw FAISS as the production path.

## What we measured

Retrieval-only quality on the shared 420-chunk corpus (EN + ES + HI)
against the shared 930-query gold set. Metrics:

- **recall@k** (k=5) — fraction of queries for which the gold FAQ ID
  appears anywhere in the top-k results.
- **MRR@k** — mean reciprocal rank of the first gold hit. Differentiates
  frameworks that return the right doc at rank 1 vs. rank 4.
- **QPS** — single-threaded queries per second on CPU, measured after
  the index is warm.
- **index_ms** — wall-clock build time from `chunks.jsonl` to a
  queryable index (includes embedding compute for everything except
  RAGatouille, which does a compressed index build on top of that).

We deliberately **did not** run end-to-end generation on each
framework; that would conflate retriever quality with prompt templates
and model calls. Retrieval quality is the bottleneck in RAG, and
holding the generator fixed while swapping retrievers is the cleanest
ablation.

## Why the ceiling is ~95%, not 100%

The gold set includes queries with legitimate paraphrase / synonym
gaps between the user phrasing and the FAQ wording — most of these are
Hindi where e5's training skew shows. None of the bi-encoder
frameworks clear 92% on Hindi alone; the remaining ~8% would need
either a stronger multilingual encoder or a rerank step. RAGatouille
(ColBERT late-interaction) was the hypothesis for closing that gap
but we didn't finish running it (see below).

## Contestants

| Framework | Retriever | Why we tried it |
|---|---|---|
| `langchain_faiss` | Raw FAISS `IndexFlatIP` + e5 | Baseline, matches production |
| `llamaindex` | `VectorStoreIndex` with HF embeddings | Popular RAG framework alternative |
| `haystack` | `InMemoryDocumentStore` + e5 | Deepset's production-oriented framework |
| `txtai` | `Embeddings` (SQLite + faiss) | Lightweight, self-contained option |
| `ragatouille` | ColBERT v2 late-interaction (opt-in) | Stronger retrieval at cost of index size |

Each framework implements a tiny protocol
(`build_index(chunks)` + `search(query, k) → [faq_id]`), so the
harness can rank them identically.

## Headline result

```
langchain_faiss   95.1%  MRR 0.787   57.9 q/s    8.9 s index
llamaindex        95.2%  MRR 0.791   40.7 q/s   57.2 s index
haystack          95.1%  MRR 0.787   20.4 q/s   12.9 s index
txtai             95.1%  MRR 0.787   58.5 q/s   14.6 s index
ragatouille         —      —           —          —      (not run)
```

Reading it:

- **Recall is essentially tied** across the four bi-encoder frameworks.
  Once they're all using the same e5 model on an L2-normalized flat
  index, there's nothing left to vary except per-framework overhead.
- **LlamaIndex wins by a hair on MRR (0.791 vs. 0.787)** — likely
  tie-breaking noise on a handful of queries. Not a reason to switch.
- **LangChain + raw FAISS and txtai lead on QPS** (~58 q/s). Haystack's
  `InMemoryDocumentStore` is ~3× slower despite using the same encoder
  — the abstraction overhead shows at query time.
- **LlamaIndex indexes 6× slower** (~57 s vs. ~9 s). Irrelevant for a
  one-shot build; would matter if we re-indexed on every deploy.

## Decision

Stay on LangChain + raw FAISS. Rationale:

1. **Tied on quality** — nothing to gain on recall by switching.
2. **Lowest overhead** — both at index build and query time.
3. **Transparent seams** — the rest of the pipeline (`RetrievalAgent`,
   `ReasoningAgent`, `SafetyAgent`) is already plain Python dataclasses
   with LCEL `RunnableLambda` adapters; bringing in LlamaIndex or
   Haystack's abstractions would obscure what's happening without
   improving any gate.
4. **Debuggability** — we own the index file and the query path.
   Frameworks with their own doc store / query engine layers make it
   harder to trace a bad retrieval to a specific chunk.

The frameworks are pinned as **optional extras** (`pip install
'.[llamaindex]' '.[haystack]' '.[txtai]'`) so the bake-off stays
reproducible without bloating the production image.

## RAGatouille (ColBERT)

Pinned as an optional extra (`pip install '.[ragatouille]'`) but not
run in the current report. Two reasons:

1. **Index size blows up** — ColBERT stores per-token embeddings, so
   420 chunks × ~100 tokens each = ~42k vectors vs. 420 in the flat
   index. The compressed PLAID index is ~20× larger on disk than the
   flat FAISS index.
2. **QPS hit on CPU** — late-interaction scoring is ~5-10× slower on
   CPU than bi-encoder cosine. With live Mistral dominating the budget
   at ~2.1 s, a 100 ms retrieval would still fit, but the ops cost
   (index rebuilds, cache invalidation) isn't worth a speculative
   quality bump.

The bench runner is written and the `compare.py` row shows
`_not run_`; dropping a one-line change into CI would execute it. We'd
only turn it on if per-language recall on Hindi became a hard blocker.

## Reproducing

```bash
# Optional deps
pip install '.[llamaindex]' '.[haystack]' '.[txtai]'
# pip install '.[ragatouille]'   # adds ~1GB torch/colbert weights

# Individual benches (each writes evals/results/bench_<name>.json)
python -m evals.framework_benchmark.bench_langchain
python -m evals.framework_benchmark.bench_llamaindex
python -m evals.framework_benchmark.bench_haystack
python -m evals.framework_benchmark.bench_txtai
# python -m evals.framework_benchmark.bench_ragatouille

# Aggregate
python -m evals.framework_benchmark.compare
# → evals/results/framework_bakeoff.md
```

Each bench uses the shared `common.run_bench` runner, which times
index build, iterates the full 930-query gold set, computes
recall@k/MRR overall and per-language, and serializes a
`BenchResult` JSON.

## What this doesn't measure

- **Reranker stacks.** Cross-encoder rerank on top of any of the
  above would likely push recall into the 97%+ range but adds ~30-80
  ms per query and a second model to operate. Outside Stage 10 scope.
- **Hybrid (dense + BM25).** Haystack and LlamaIndex both support
  this; could help on exact-token queries that e5 underweights. Not
  measured here.
- **End-to-end answer quality.** Citation F1 and hallucination rate
  (see [eval_report.md](eval_report.md)) depend on both retrieval and
  generation. Swapping retrievers while holding Mistral Small fixed
  would be the way to check whether the ~5% retrieval gap we can't
  close is actually bottlenecking the final answer — it isn't, based
  on the offline oracle numbers.
