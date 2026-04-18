"""Build FAISS indices over the multilingual FAQ corpus.

Produces:
  data/processed/chunks.jsonl          — all chunks (EN+ES+HI), one per line
  data/processed/metadata.jsonl        — row-aligned metadata for the unified index
  data/processed/faiss_all.index       — unified cross-lingual index
  data/processed/faiss_{en,es,hi}.index — per-language indices (for ablation)
  data/processed/meta_{en,es,hi}.jsonl — row-aligned metadata per per-language index

Uses intfloat/multilingual-e5-base with the e5 passage/query prefixing
convention. Vectors are L2-normalized and stored in an IndexFlatIP so inner
product == cosine similarity.

Usage:
  python -m ingest.build_faiss --input data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest.chunking import Chunk, chunks_to_jsonl, load_faq_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_faiss")

MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

LANGS = ("en", "es", "hi")


def _encode_passages(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode passages with the e5 prefix, return L2-normalized float32 array."""
    prefixed = [PASSAGE_PREFIX + t for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embs.astype(np.float32, copy=False)


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode a single query with the e5 query prefix (L2-normalized, 2-D)."""
    emb = model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32, copy=False)


def _build_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx


def _save_meta(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def _load_all_chunks(raw_dir: Path) -> dict[str, list[Chunk]]:
    by_lang: dict[str, list[Chunk]] = {}
    for lang in LANGS:
        src = raw_dir / f"faqs_{lang}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        by_lang[lang] = list(load_faq_chunks(src, lang=lang))
        log.info("Loaded %d chunks from %s", len(by_lang[lang]), src.name)
    return by_lang


def build(raw_dir: Path, out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_lang = _load_all_chunks(raw_dir)

    all_chunks: list[Chunk] = [c for lang in LANGS for c in by_lang[lang]]
    chunks_to_jsonl(all_chunks, out_dir / "chunks.jsonl")
    log.info("Wrote %d chunks to %s", len(all_chunks), out_dir / "chunks.jsonl")

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    # Per-language indices
    for lang in LANGS:
        chunks = by_lang[lang]
        log.info("[%s] encoding %d passages", lang, len(chunks))
        embs = _encode_passages(model, [c.text for c in chunks])
        idx = _build_index(embs)
        out_idx = out_dir / f"faiss_{lang}.index"
        faiss.write_index(idx, str(out_idx))
        _save_meta(chunks, out_dir / f"meta_{lang}.jsonl")
        log.info("[%s] wrote %s  (n=%d, dim=%d)", lang, out_idx.name, idx.ntotal, embs.shape[1])

    # Unified cross-lingual index
    log.info("[all] encoding unified %d passages", len(all_chunks))
    embs = _encode_passages(model, [c.text for c in all_chunks])
    idx = _build_index(embs)
    faiss.write_index(idx, str(out_dir / "faiss_all.index"))
    _save_meta(all_chunks, out_dir / "metadata.jsonl")
    log.info("[all] wrote faiss_all.index  (n=%d, dim=%d)", idx.ntotal, embs.shape[1])

    _smoke_test(model, idx, all_chunks)


def _smoke_test(
    model: SentenceTransformer, idx: faiss.IndexFlatIP, meta: list[Chunk]
) -> None:
    probes = {
        "en": "How can I cancel my subscription?",
        "es": "¿Cómo actualizo mi método de pago?",
        "hi": "मेरा ऑर्डर कब आएगा?",
    }
    log.info("--- Smoke test on unified index (top-3 per query) ---")
    for lang, q in probes.items():
        qv = encode_query(model, q)
        scores, ids = idx.search(qv, 3)
        log.info("[%s] %r", lang, q)
        for rank, (i, s) in enumerate(zip(ids[0], scores[0]), 1):
            c = meta[int(i)]
            log.info("    %d. [%.3f] %s  %s: %s",
                     rank, float(s), c.lang, c.id, c.question[:80])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args(argv)
    build(args.input, args.out, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
