"""Generate the English FAQ corpus for fictional SaaS/e-commerce 'TechNova'.

Produces data/raw/faqs_en.jsonl with per-FAQ records:
    {"id": "faq-<cat>-NNN", "question": str, "answer": str, "category": str}

Batches per category so the LLM keeps context consistent within a category.
Uses Mistral JSON mode for structured output; idempotent across reruns if
`--overwrite` is not passed (skips already-written IDs).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field, ValidationError

from ingest._client import chat_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate_faqs")

CATEGORIES: dict[str, tuple[int, str]] = {
    "billing":  (25, "pricing tiers, invoices, refunds, payment methods, tax, currency"),
    "account":  (25, "sign-up, login, password reset, 2FA, profile, account deletion, email change"),
    "shipping": (25, "delivery windows, tracking, carriers, international shipping, address changes"),
    "returns":  (20, "RMA process, return windows, refund timelines, damaged items, exchanges"),
    "product":  (25, "product features, compatibility, setup, troubleshooting, warranty, specs"),
    "sales":    (20, "discounts, promo codes, volume/B2B pricing, free trials, contact sales, comparisons"),
}

SYSTEM = """You are a technical writer building the FAQ knowledge base for TechNova,
a fictional SaaS + e-commerce company that sells productivity software and
accompanying hardware (laptops, accessories, smart devices). Answers are
concise, professional, customer-service tone, factual-sounding but entirely
fabricated (TechNova is fictional — do not name real brands or real URLs).
Keep each answer 2-4 sentences. Questions must be naturally phrased, the way
a real customer would type into a help search bar."""

USER_TMPL = """Write {n} distinct English FAQ entries for the "{category}" category.
Topic coverage hints: {hints}.
Return JSON with exactly this shape (no prose, no markdown fences):

{{
  "faqs": [
    {{"question": "<customer question>", "answer": "<2-4 sentence answer>"}},
    ...
  ]
}}

Requirements:
- Exactly {n} items in the "faqs" array.
- Each question must be unique and phrased naturally (no "FAQ #1:" prefixes).
- Do not use placeholder text like [YOUR COMPANY] — write "TechNova".
- Do not invent phone numbers, emails, or URLs; use phrases like "contact support" instead.
"""


class FAQItem(BaseModel):
    question: str = Field(min_length=8, max_length=300)
    answer: str = Field(min_length=20, max_length=1200)


class FAQBatch(BaseModel):
    faqs: list[FAQItem]


def _load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ids.add(json.loads(line)["id"])
    return ids


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def generate(output_path: Path, overwrite: bool) -> None:
    if overwrite and output_path.exists():
        output_path.unlink()
        log.info("Removed existing %s (--overwrite)", output_path)

    existing = _load_existing(output_path)
    total_target = sum(n for n, _ in CATEGORIES.values())
    log.info("Existing IDs: %d / target %d", len(existing), total_target)

    for category, (n_target, hints) in CATEGORIES.items():
        have = sum(1 for i in existing if i.startswith(f"faq-{category}-"))
        need = n_target - have
        if need <= 0:
            log.info("[%s] already has %d entries, skipping", category, have)
            continue
        log.info("[%s] need %d more entries", category, need)

        user = USER_TMPL.format(n=need, category=category, hints=hints)
        raw = chat_json(system=SYSTEM, user=user, max_tokens=6000)

        try:
            batch = FAQBatch.model_validate(raw)
        except ValidationError as e:
            log.error("[%s] schema validation failed: %s", category, e)
            raise

        # Dedup within category by normalized question
        seen_q: set[str] = set()
        rows: list[dict] = []
        start_idx = have + 1
        for item in batch.faqs:
            qnorm = item.question.lower().strip()
            if qnorm in seen_q:
                continue
            seen_q.add(qnorm)
            fid = f"faq-{category}-{start_idx + len(rows):03d}"
            rows.append({
                "id": fid,
                "question": item.question.strip(),
                "answer": item.answer.strip(),
                "category": category,
            })
            if len(rows) == need:
                break

        wrote = _append_jsonl(output_path, rows)
        log.info("[%s] wrote %d rows (now %d total)", category, wrote, have + wrote)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/faqs_en.jsonl"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    generate(args.output, args.overwrite)

    # Final count
    ids = _load_existing(args.output)
    log.info("Done. Total FAQs: %d", len(ids))
    target = sum(n for n, _ in CATEGORIES.values())
    if len(ids) < target:
        log.warning("Below target (%d < %d)", len(ids), target)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
