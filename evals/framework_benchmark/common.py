"""Shared harness for Stage-10 framework bake-off.

Each framework implements a tiny ``Bench`` protocol:
  * ``build_index(chunks)`` — one-shot index build, captures wall time
    + on-disk size when relevant.
  * ``search(query, k)`` — returns an ordered list of FAQ IDs.
  * ``close()`` — release handles (GPU tensors, open files, …).

The harness runs the bench over the gold QA corpus and computes:
  * recall@k — is any relevant ID in the top-k?
  * MRR@k    — 1/rank of first relevant ID, else 0.
  * QPS      — queries per wall-second on single-threaded CPU.
  * index_ms — time to build the index from chunks.
  * index_bytes — on-disk (or in-mem estimate) of the index.

The apples-to-apples is imperfect — RAGatouille uses ColBERTv2, everyone
else uses the same e5 multilingual model — but the point of the
bake-off is to show we picked LangChain+FAISS for defensible reasons,
not that we tuned every framework to its peak.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


CHUNKS_FILE = Path("data/processed/chunks.jsonl")
GOLD_FILES = [
    Path("data/eval/gold_qa_en.jsonl"),
    Path("data/eval/gold_qa_es.jsonl"),
    Path("data/eval/gold_qa_hi.jsonl"),
]


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    lang: str
    category: str
    text: str
    question: str
    answer: str


@dataclass(frozen=True)
class GoldQuery:
    id: str
    lang: str
    query: str
    relevant_faq_ids: list[str]


def load_chunks() -> list[ChunkRecord]:
    out: list[ChunkRecord] = []
    with CHUNKS_FILE.open() as f:
        for line in f:
            o = json.loads(line)
            out.append(
                ChunkRecord(
                    id=o["id"],
                    lang=o["lang"],
                    category=o.get("category", ""),
                    text=o["text"],
                    question=o.get("question", ""),
                    answer=o.get("answer", ""),
                )
            )
    return out


def load_gold(langs: Iterable[str] = ("en", "es", "hi")) -> list[GoldQuery]:
    out: list[GoldQuery] = []
    for p in GOLD_FILES:
        lang = p.stem.split("_")[-1]
        if lang not in langs:
            continue
        with p.open() as f:
            for line in f:
                o = json.loads(line)
                out.append(
                    GoldQuery(
                        id=o["id"],
                        lang=o["lang"],
                        query=o["query"],
                        relevant_faq_ids=list(o["relevant_faq_ids"]),
                    )
                )
    return out


class Bench(Protocol):
    name: str

    def build_index(self, chunks: list[ChunkRecord]) -> None: ...
    def search(self, query: str, k: int) -> list[str]: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def recall_at_k(predicted: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 1.0
    rel = set(relevant)
    return 1.0 if any(p in rel for p in predicted) else 0.0


def mrr_at_k(predicted: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for rank, pid in enumerate(predicted, 1):
        if pid in rel:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


@dataclass
class BenchResult:
    name: str
    k: int
    n_chunks: int
    n_queries: int
    index_ms: float = 0.0
    index_bytes: int = 0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    qps: float = 0.0
    per_lang: dict[str, dict[str, float]] = field(default_factory=dict)
    error: str | None = None


def run_bench(
    bench: Bench,
    chunks: list[ChunkRecord],
    gold: list[GoldQuery],
    *,
    k: int = 5,
) -> BenchResult:
    result = BenchResult(name=bench.name, k=k, n_chunks=len(chunks), n_queries=len(gold))
    try:
        t0 = time.perf_counter()
        bench.build_index(chunks)
        result.index_ms = (time.perf_counter() - t0) * 1000

        by_lang: dict[str, list[tuple[float, float]]] = {}
        recs, mrrs = [], []
        t0 = time.perf_counter()
        for q in gold:
            preds = bench.search(q.query, k=k)
            r = recall_at_k(preds, q.relevant_faq_ids)
            m = mrr_at_k(preds, q.relevant_faq_ids)
            recs.append(r)
            mrrs.append(m)
            by_lang.setdefault(q.lang, []).append((r, m))
        dt = time.perf_counter() - t0
        result.qps = len(gold) / dt if dt else 0.0
        result.recall_at_k = sum(recs) / len(recs)
        result.mrr = sum(mrrs) / len(mrrs)
        for lang, pairs in by_lang.items():
            result.per_lang[lang] = {
                "n": float(len(pairs)),
                "recall_at_k": sum(p[0] for p in pairs) / len(pairs),
                "mrr": sum(p[1] for p in pairs) / len(pairs),
            }
    except Exception as e:  # pragma: no cover
        result.error = f"{type(e).__name__}: {e}"
    finally:
        try:
            bench.close()
        except Exception:  # pragma: no cover
            pass
    return result


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------


RESULTS_DIR = Path("evals/results")


def write_result(result: BenchResult) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"bench_{result.name}.json"
    data = {
        "name": result.name,
        "k": result.k,
        "n_chunks": result.n_chunks,
        "n_queries": result.n_queries,
        "index_ms": result.index_ms,
        "index_bytes": result.index_bytes,
        "recall_at_k": result.recall_at_k,
        "mrr": result.mrr,
        "qps": result.qps,
        "per_lang": result.per_lang,
        "error": result.error,
    }
    path.write_text(json.dumps(data, indent=2))
    return path
