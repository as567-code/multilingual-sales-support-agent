"""Validate Stage-1 artefacts: schemas, counts, duplicates, ID consistency.

Run: python -m ingest.validate_stage1
Exit code 0 on pass, non-zero with a list of failures on fail.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

RAW = Path("data/raw")
EVAL = Path("data/eval")
LANGS = ("en", "es", "hi")
VALID_DIFF = {"easy", "medium", "hard"}
VALID_CAT = {"billing", "account", "shipping", "returns", "product", "sales", "ood"}


class FAQ(BaseModel):
    id: str
    question: str = Field(min_length=5)
    answer: str = Field(min_length=10)
    category: str


class QA(BaseModel):
    id: str
    lang: str
    query: str = Field(min_length=3)
    gold_answer: str = Field(min_length=10)
    relevant_faq_ids: list[str]
    category: str
    difficulty: str


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln_no} invalid JSON: {e}")
    return rows


def _check(label: str, ok: bool, msg: str, failures: list[str]) -> None:
    status = "✓" if ok else "✗"
    print(f"  {status} {label}: {msg}")
    if not ok:
        failures.append(f"{label}: {msg}")


def main() -> int:
    failures: list[str] = []

    # --- FAQ per-language checks
    faq_ids_by_lang: dict[str, list[str]] = {}
    for lang in LANGS:
        p = RAW / f"faqs_{lang}.jsonl"
        rows = _load(p)
        print(f"\n[{p}] {len(rows)} rows")
        _check("count", len(rows) >= 120, f"{len(rows)} (>= 120)", failures)
        ids = []
        categories = collections.Counter()
        for i, r in enumerate(rows):
            try:
                FAQ.model_validate(r)
            except ValidationError as e:
                failures.append(f"{p} row {i}: {e}")
                continue
            ids.append(r["id"])
            categories[r["category"]] += 1
        faq_ids_by_lang[lang] = ids
        _check("unique ids", len(set(ids)) == len(ids), f"{len(set(ids))}/{len(ids)}", failures)
        _check("categories", set(categories) <= VALID_CAT,
               f"{dict(categories)}", failures)

    # --- Cross-language ID parity
    en_ids = set(faq_ids_by_lang["en"])
    for lang in ("es", "hi"):
        other = set(faq_ids_by_lang[lang])
        _check(f"id parity en↔{lang}",
               en_ids == other,
               f"diff = {len(en_ids ^ other)}", failures)

    # --- Eval per-language checks
    totals = 0
    for lang in LANGS:
        p = EVAL / f"gold_qa_{lang}.jsonl"
        rows = _load(p)
        print(f"\n[{p}] {len(rows)} rows")
        totals += len(rows)
        _check("count", len(rows) >= 165, f"{len(rows)} (>= 165)", failures)
        seen_q: set[str] = set()
        seen_id: set[str] = set()
        diff_counter: collections.Counter = collections.Counter()
        cat_counter: collections.Counter = collections.Counter()
        lang_mismatches = 0
        bad_faq_refs = 0
        for i, r in enumerate(rows):
            try:
                QA.model_validate(r)
            except ValidationError as e:
                failures.append(f"{p} row {i}: {e}")
                continue
            if r["lang"] != lang:
                lang_mismatches += 1
            seen_id.add(r["id"])
            seen_q.add((r["query"] or "").strip().lower())
            diff_counter[r["difficulty"]] += 1
            cat_counter[r["category"]] += 1
            for fid in r["relevant_faq_ids"]:
                if fid not in en_ids:
                    bad_faq_refs += 1
        _check("unique ids", len(seen_id) == len(rows), f"{len(seen_id)}/{len(rows)}", failures)
        _check("unique queries", len(seen_q) == len(rows), f"{len(seen_q)}/{len(rows)} (dedup)", failures)
        _check("lang field correct", lang_mismatches == 0, f"{lang_mismatches} wrong-lang rows", failures)
        _check("faq refs resolve", bad_faq_refs == 0, f"{bad_faq_refs} dangling FAQ IDs", failures)
        _check("difficulty set", set(diff_counter) <= VALID_DIFF,
               f"{dict(diff_counter)}", failures)
        _check("category set", set(cat_counter) <= VALID_CAT,
               f"{dict(cat_counter)}", failures)
        print(f"    difficulty: {dict(diff_counter)}")
        print(f"    category:   {dict(cat_counter)}")

    _check("total eval queries", totals >= 500, f"{totals} (>= 500)", failures)

    print("\n" + ("=" * 60))
    if failures:
        print(f"FAIL: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: Stage 1 artefacts valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
