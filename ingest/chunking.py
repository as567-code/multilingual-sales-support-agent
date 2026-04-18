"""FAQ chunker: one chunk per FAQ (question + answer), plus metadata.

The FAQ corpus is already tight — each entry is 2-4 sentences — so splitting
isn't useful. Each chunk carries the original question, the answer, and all
metadata needed for lang-aware retrieval.

Chunk text format embeds the question verbatim because users rarely type the
exact answer — matching on the question + surrounding answer context is what
gives the e5 encoder good signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    id: str
    lang: str
    category: str
    text: str           # encoded form: "Q: <question>\nA: <answer>"
    question: str       # retained separately for display / reranking
    answer: str
    source_file: str


def chunk_faq_row(row: dict, lang: str, source_file: str) -> Chunk:
    question = row["question"].strip()
    answer = row["answer"].strip()
    return Chunk(
        id=row["id"],
        lang=lang,
        category=row["category"],
        text=f"Q: {question}\nA: {answer}",
        question=question,
        answer=answer,
        source_file=source_file,
    )


def load_faq_chunks(path: Path, lang: str) -> Iterable[Chunk]:
    """Yield a Chunk for every non-empty line in a faqs_{lang}.jsonl file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            yield chunk_faq_row(row, lang=lang, source_file=path.name)


def chunks_to_jsonl(chunks: Iterable[Chunk], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n
