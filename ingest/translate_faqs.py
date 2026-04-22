"""Translate the EN FAQ corpus to ES and HI, preserving IDs.

Reads data/raw/faqs_en.jsonl and writes data/raw/faqs_{es,hi}.jsonl
with identical id/category and translated question+answer. Uses Mistral
JSON mode with explicit script validation for Hindi (Devanagari).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from ingest._client import chat_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translate_faqs")

LANG_NAMES = {"es": "Spanish (Spain / neutral LATAM)", "hi": "Hindi (Devanagari script)"}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

BATCH = 10  # Translating more than this per call starts losing fidelity.

SYSTEM_TMPL = """You are a professional translator producing natural, customer-service
quality translations into {lang_name}. You translate faithfully — same meaning,
tone, and level of detail — and preserve product name 'TechNova' verbatim.
Do NOT transliterate 'TechNova'; keep it in Latin characters."""

USER_TMPL = """Translate the following {n} English FAQ entries to {lang_name}.
Return JSON with this exact shape (no prose, no markdown):

{{
  "items": [
    {{"id": "<same id>", "question": "<translated>", "answer": "<translated>"}},
    ...
  ]
}}

Rules:
- Preserve each "id" exactly.
- Keep "TechNova" in Latin characters (do not transliterate).
- Keep a professional support-doc tone; do not add content.
- Answers: 2-4 sentences, same length as source.

Input JSON:
{payload}
"""


class TranslatedItem(BaseModel):
    id: str
    question: str = Field(min_length=5, max_length=400)
    answer: str = Field(min_length=10, max_length=2000)


class TranslatedBatch(BaseModel):
    items: list[TranslatedItem]


def _load_en(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_existing(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _looks_like_hindi(text: str) -> bool:
    """Devanagari should dominate Hindi output; reject mostly-Latin responses."""
    deva = len(DEVANAGARI_RE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    # 'TechNova' contributes ~8 Latin chars per mention; allow some but not dominance.
    return deva >= 5 and deva > latin / 2


def translate_lang(en_rows: list[dict], out_path: Path, lang: str, overwrite: bool) -> None:
    if overwrite and out_path.exists():
        out_path.unlink()
        log.info("Removed existing %s", out_path)

    done_ids = _load_existing(out_path)
    todo = [r for r in en_rows if r["id"] not in done_ids]
    log.info("[%s] %d already translated, %d to go", lang, len(done_ids), len(todo))

    lang_name = LANG_NAMES[lang]
    system = SYSTEM_TMPL.format(lang_name=lang_name)

    for i in range(0, len(todo), BATCH):
        batch_in = todo[i : i + BATCH]
        payload = json.dumps(
            [{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in batch_in],
            ensure_ascii=False,
        )
        user = USER_TMPL.format(n=len(batch_in), lang_name=lang_name, payload=payload)
        raw = chat_json(system=system, user=user, max_tokens=5000, temperature=0.2)

        try:
            parsed = TranslatedBatch.model_validate(raw)
        except ValidationError as e:
            log.error("[%s] batch %d schema error: %s", lang, i // BATCH, e)
            raise

        by_id = {it.id: it for it in parsed.items}
        rows_out: list[dict] = []
        for src in batch_in:
            tr = by_id.get(src["id"])
            if tr is None:
                log.warning("[%s] missing translation for %s", lang, src["id"])
                continue
            if lang == "hi" and not _looks_like_hindi(tr.question + " " + tr.answer):
                log.warning("[%s] id=%s looks non-Devanagari; skipping for review", lang, src["id"])
                continue
            rows_out.append({
                "id": src["id"],
                "question": tr.question.strip(),
                "answer": tr.answer.strip(),
                "category": src["category"],
            })
        _write_rows(out_path, rows_out)
        log.info("[%s] batch %d: wrote %d / %d", lang, i // BATCH, len(rows_out), len(batch_in))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/faqs_en.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--langs", nargs="+", default=["es", "hi"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    en = _load_en(args.input)
    log.info("Loaded %d English FAQs from %s", len(en), args.input)

    for lang in args.langs:
        out = args.out_dir / f"faqs_{lang}.jsonl"
        translate_lang(en, out, lang, args.overwrite)
        final = _load_existing(out)
        log.info("[%s] final count: %d", lang, len(final))
        if len(final) < len(en) * 0.95:
            log.error("[%s] coverage %d/%d too low", lang, len(final), len(en))
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
