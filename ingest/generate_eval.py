"""Build the multilingual eval set (gold_qa_{lang}.jsonl).

Per language, produces:
  - Paraphrase/indirect queries: 1-2 per source FAQ, varying directness
  - Distractor queries: out-of-scope, gold = canonical abstention

Schema written to data/eval/gold_qa_{en,es,hi}.jsonl:
  {
    "id": "qa-<lang>-NNNN",
    "lang": "en|es|hi",
    "query": "...",
    "gold_answer": "...",
    "relevant_faq_ids": ["faq-..."]  # [] for distractors
    "category": "billing|...|ood",
    "difficulty": "easy|medium|hard"
  }

Target: ≥ 165 per lang, ≥ 500 total.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ingest._client import chat_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_eval")

LANG_NAME = {"en": "English", "es": "Spanish", "hi": "Hindi (Devanagari)"}

ABSTENTION = {
    "en": "I don't have that information in the TechNova knowledge base. Please contact support for help.",
    "es": "No tengo esa información en la base de conocimientos de TechNova. Por favor, contacta con soporte.",
    "hi": "मेरे पास TechNova जानकारी आधार में यह जानकारी नहीं है। कृपया सहायता के लिए समर्थन टीम से संपर्क करें।",
}

VALID_DIFF = {"easy", "medium", "hard"}

# ---- Paraphrase round ---------------------------------------------------------

PARA_SYSTEM = """You produce realistic customer support queries for an eval dataset.
Queries must feel like what a real customer would type into a help search bar or
chat widget. Vary phrasing, directness, and length. Some queries are direct
rewrites of the FAQ question; some are oblique (describing the problem without
using FAQ keywords). You write ONLY in the requested target language."""

PARA_USER_TMPL = """Target language: {lang_name}. Source FAQs follow as JSON.
For EACH FAQ, produce EXACTLY TWO evaluation queries:
  - variant "direct": a natural rephrase / near-paraphrase of the question
  - variant "indirect": an oblique phrasing where a customer describes their
    situation without repeating the FAQ's keywords (still clearly about the same topic)

For each query, assign a difficulty:
  - "easy"   for direct rewrites with obvious keyword overlap
  - "medium" for natural paraphrases with some keyword drift
  - "hard"   for oblique phrasings that require real semantic understanding

Return JSON (no prose, no markdown):
{{
  "queries": [
    {{
      "faq_id": "<source id>",
      "variant": "direct" | "indirect",
      "query": "<customer query in {lang_name}>",
      "difficulty": "easy" | "medium" | "hard"
    }},
    ...
  ]
}}

Exactly {expected} objects — two per FAQ. Every faq_id must match one from the input.

Source FAQs:
{payload}
"""


class ParaQuery(BaseModel):
    faq_id: str
    variant: str
    query: str = Field(min_length=5, max_length=400)
    difficulty: str


class ParaBatch(BaseModel):
    queries: list[ParaQuery]


# ---- Distractor round ---------------------------------------------------------

DIST_SYSTEM = """You generate OUT-OF-DOMAIN customer queries for a sales/support
eval dataset. The product, TechNova, sells productivity software and consumer
hardware (laptops, smart devices). Generate queries that sound like real user
input but which TechNova CANNOT answer — e.g. questions about unrelated
products, personal advice, weather, legal matters, other companies. Write
ONLY in the requested target language. Vary topics and phrasings."""

DIST_USER_TMPL = """Produce EXACTLY {n} out-of-domain customer queries in {lang_name}.
Each query MUST be something a helpful support chatbot for TechNova should
*decline* to answer because it's outside the product scope.

Return JSON (no prose):
{{
  "queries": [
    {{"query": "<in {lang_name}>", "topic": "<brief english tag>"}},
    ...
  ]
}}
"""


class DistItem(BaseModel):
    query: str = Field(min_length=5, max_length=400)
    topic: str


class DistBatch(BaseModel):
    queries: list[DistItem]


# ---- IO -----------------------------------------------------------------------


def _load_faqs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---- Generation ---------------------------------------------------------------


def _gen_paraphrases(
    faqs: list[dict], lang: str, batch_size: int, rng: random.Random
) -> list[dict]:
    """Two queries per FAQ (direct + indirect)."""
    out: list[dict] = []
    by_id = {f["id"]: f for f in faqs}
    order = list(faqs)
    rng.shuffle(order)

    for i in range(0, len(order), batch_size):
        batch = order[i : i + batch_size]
        payload_items = [
            {"id": f["id"], "question": f["question"], "answer": f["answer"], "category": f["category"]}
            for f in batch
        ]
        user = PARA_USER_TMPL.format(
            lang_name=LANG_NAME[lang],
            expected=len(batch) * 2,
            payload=json.dumps(payload_items, ensure_ascii=False),
        )
        raw = chat_json(system=PARA_SYSTEM, user=user, max_tokens=6000, temperature=0.5)
        try:
            parsed = ParaBatch.model_validate(raw)
        except ValidationError as e:
            log.error("[%s] paraphrase batch %d schema error: %s", lang, i // batch_size, e)
            raise

        seen_q: set[str] = set()
        for pq in parsed.queries:
            src = by_id.get(pq.faq_id)
            if src is None:
                log.debug("[%s] unknown faq_id %s, skipping", lang, pq.faq_id)
                continue
            qkey = pq.query.strip().lower()
            if qkey in seen_q:
                continue
            seen_q.add(qkey)
            diff = pq.difficulty.lower() if pq.difficulty.lower() in VALID_DIFF else "medium"
            out.append({
                "id": None,  # assigned later
                "lang": lang,
                "query": pq.query.strip(),
                "gold_answer": src["answer"],
                "relevant_faq_ids": [src["id"]],
                "category": src["category"],
                "difficulty": diff,
            })
        log.info("[%s] paraphrase batch %d: +%d (total %d)",
                 lang, i // batch_size, len(parsed.queries), len(out))
    return out


def _gen_distractors(lang: str, count: int, per_call: int = 30) -> list[dict]:
    out: list[dict] = []
    while len(out) < count:
        ask = min(per_call, count - len(out))
        user = DIST_USER_TMPL.format(n=ask, lang_name=LANG_NAME[lang])
        raw = chat_json(system=DIST_SYSTEM, user=user, max_tokens=3000, temperature=0.7)
        try:
            parsed = DistBatch.model_validate(raw)
        except ValidationError as e:
            log.error("[%s] distractor batch schema error: %s", lang, e)
            raise
        seen: set[str] = {r["query"].lower() for r in out}
        for it in parsed.queries:
            qkey = it.query.strip().lower()
            if qkey in seen:
                continue
            seen.add(qkey)
            out.append({
                "id": None,
                "lang": lang,
                "query": it.query.strip(),
                "gold_answer": ABSTENTION[lang],
                "relevant_faq_ids": [],
                "category": "ood",
                "difficulty": "hard",
            })
        log.info("[%s] distractor total: %d / %d", lang, len(out), count)
    return out


def build_lang(
    lang: str, faqs_dir: Path, out_path: Path, batch_size: int, distractors: int,
    seed: int,
) -> int:
    rng = random.Random(seed + hash(lang))
    faqs = _load_faqs(faqs_dir / f"faqs_{lang}.jsonl")
    log.info("[%s] loaded %d FAQs", lang, len(faqs))

    rows = _gen_paraphrases(faqs, lang, batch_size, rng)
    rows.extend(_gen_distractors(lang, distractors))

    # Assign stable IDs
    for idx, r in enumerate(rows, 1):
        r["id"] = f"qa-{lang}-{idx:04d}"

    _write_jsonl(out_path, rows)
    log.info("[%s] wrote %d queries to %s", lang, len(rows), out_path)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faqs-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument("--langs", nargs="+", default=["en", "es", "hi"])
    parser.add_argument("--batch-size", type=int, default=15,
                        help="# FAQs per paraphrase batch (×2 queries per FAQ)")
    parser.add_argument("--distractors", type=int, default=30,
                        help="# OOD distractors per language")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    totals: dict[str, int] = {}
    for lang in args.langs:
        out = args.out_dir / f"gold_qa_{lang}.jsonl"
        totals[lang] = build_lang(
            lang=lang, faqs_dir=args.faqs_dir, out_path=out,
            batch_size=args.batch_size, distractors=args.distractors, seed=args.seed,
        )

    total = sum(totals.values())
    log.info("TOTALS: %s → %d queries", totals, total)
    if total < 500 or any(v < 165 for v in totals.values()):
        log.error("Below PRD targets (≥500 total, ≥165/lang)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
