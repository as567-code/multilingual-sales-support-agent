"""Retrieval Agent — stage 1 of the 3-agent pipeline.

Wraps the Stage-2 FAISS unified index behind a clean, typed interface:
  agent = RetrievalAgent()
  hits = agent.retrieve("How do I cancel my plan?")

Responsibilities:
  * Detect the query language (EN/ES/HI) for logging + downstream prompting.
  * Encode with the e5 ``query:`` prefix (symmetric with passage encoding).
  * FAISS top-k search against the unified cross-lingual index.
  * Apply a cosine-similarity floor → abstain signal when nothing clears it.
  * Return typed ``RetrievalHit`` records with the full FAQ metadata.

Exposed as an LCEL Runnable via ``as_runnable()`` so the Stage-6 orchestrator
can chain it with the Reasoning + Safety agents.

Not included (by design):
  * Reranking — the e5-base retrieval quality on this corpus (recall@5=0.93)
    makes a reranker premature. Stage 10 revisits this in the framework bake-off.
  * Query rewriting — would help on multi-hop queries, but those are noted
    future work in the eval card; single-hop dominates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import faiss  # type: ignore[import-untyped]
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings
from app.utils.language import LangGuess, detect_language
from app.utils.logging import get_logger
from ingest.build_faiss import encode_query

log = get_logger("retrieval")


class RetrievalHit(BaseModel):
    """One FAQ match. Score is cosine similarity (inner product on L2-norm vectors)."""
    faq_id: str
    lang: str
    category: str
    question: str
    answer: str
    text: str
    score: float = Field(description="cosine similarity in [-1, 1]")


class RetrievalResult(BaseModel):
    query: str
    detected_lang: str
    lang_confidence: float
    lang_source: str
    hits: list[RetrievalHit]
    abstain: bool = Field(
        description="True when no hit cleared retrieval_min_score."
    )


@lru_cache(maxsize=1)
def _load_model(name: str) -> SentenceTransformer:
    log.info("loading_embed_model", model=name)
    return SentenceTransformer(name)


@lru_cache(maxsize=4)
def _load_index_and_meta(index_path: str, meta_path: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    idx = faiss.read_index(index_path)
    meta: list[dict] = []
    with Path(meta_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    if idx.ntotal != len(meta):
        raise RuntimeError(
            f"index ntotal={idx.ntotal} vs metadata rows={len(meta)} mismatch"
        )
    log.info("loaded_index", path=index_path, ntotal=idx.ntotal)
    return idx, meta


@dataclass
class RetrievalAgent:
    """Encapsulates the model + index + retrieval policy.

    Defaults come from ``app.config.Settings``; override for tests or ablations.
    """
    settings: Settings | None = None
    top_k: int | None = None
    min_score: float | None = None

    def __post_init__(self) -> None:
        s = self.settings or get_settings()
        self.settings = s
        if self.top_k is None:
            self.top_k = s.retrieval_top_k
        if self.min_score is None:
            self.min_score = s.retrieval_min_score
        self._model = _load_model(s.embed_model)
        self._idx, self._meta = _load_index_and_meta(
            str(s.index_dir / s.index_file),
            str(s.index_dir / s.meta_file),
        )

    def retrieve(
        self,
        query: str,
        *,
        lang_hint: str | None = None,
        k: int | None = None,
    ) -> RetrievalResult:
        q = (query or "").strip()
        k = k or self.top_k
        assert k is not None

        if lang_hint:
            guess = LangGuess(lang_hint, 1.0, "hint")
        else:
            guess = detect_language(q)

        if not q:
            log.warning("empty_query")
            return RetrievalResult(
                query=q, detected_lang=guess.lang,
                lang_confidence=guess.confidence, lang_source=guess.source,
                hits=[], abstain=True,
            )

        qv = encode_query(self._model, q)
        scores, ids = self._idx.search(qv, k)

        min_score = self.min_score
        assert min_score is not None
        hits: list[RetrievalHit] = []
        for i, s in zip(ids[0], scores[0]):
            if int(i) < 0:  # FAISS pad for n<k
                continue
            if float(s) < min_score:
                continue
            row = self._meta[int(i)]
            hits.append(RetrievalHit(
                faq_id=row["id"],
                lang=row["lang"],
                category=row["category"],
                question=row["question"],
                answer=row["answer"],
                text=row["text"],
                score=float(s),
            ))

        abstain = not hits
        log.info(
            "retrieval_done",
            query=q[:80],
            lang=guess.lang,
            lang_source=guess.source,
            n_hits=len(hits),
            top_score=hits[0].score if hits else None,
            abstain=abstain,
        )
        return RetrievalResult(
            query=q,
            detected_lang=guess.lang,
            lang_confidence=guess.confidence,
            lang_source=guess.source,
            hits=hits,
            abstain=abstain,
        )

    def as_runnable(self) -> RunnableLambda:
        """Wrap ``retrieve`` as an LCEL Runnable.

        Input: ``{"query": str, "lang_hint": str | None}`` or a bare string.
        Output: ``RetrievalResult``.
        """
        def _invoke(payload: str | dict) -> RetrievalResult:
            if isinstance(payload, str):
                return self.retrieve(payload)
            return self.retrieve(
                payload["query"],
                lang_hint=payload.get("lang_hint"),
            )
        return RunnableLambda(_invoke)
