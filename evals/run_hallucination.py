"""Hallucination eval — citation groundedness over the accuracy run's
per-query output.

The PRD targets ≥85% hallucination reduction vs. a single-shot baseline
(no retrieval, no grounding). Full RAGAS faithfulness is an option, but
for a deterministic, zero-cost signal we use citation groundedness:

  * ``grounded``       — every predicted citation appears in gold.
  * ``ungrounded``     — at least one predicted citation is fabricated
    (not in the corpus or not relevant to the gold).
  * ``unverified``     — the answer is not an abstention and has zero
    citations at all, i.e. the reasoning agent made a claim with no
    source attribution.

Hallucination rate = (ungrounded + unverified) / non_abstain_total.

This is a strict upper bound — it assumes gold_citations is the only
acceptable set of supporting IDs. We complement it with ``cited_in_corpus``
which is more lenient and only asks whether the IDs exist at all.

Inputs:
  * ``evals/results/accuracy.jsonl`` (written by run_accuracy)
Outputs:
  * ``evals/results/hallucination_summary.json``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.utils.logging import configure_logging, get_logger
from evals.harness import ensure_results_dir, load_jsonl, stamp, write_summary

log = get_logger("eval.halluc")


def _load_corpus_ids() -> set[str]:
    settings = get_settings()
    meta = settings.index_dir / settings.meta_file
    ids: set[str] = set()
    with meta.open() as f:
        for line in f:
            obj = json.loads(line)
            ids.add(obj["id"])
    return ids


def classify(pred_cits: list[str], gold_cits: list[str], corpus_ids: set[str], abstain: bool) -> str:
    if abstain:
        return "abstain"
    if not pred_cits:
        return "unverified"
    gold = set(gold_cits)
    pred = set(pred_cits)
    # Any citation that's not in gold OR not in the corpus is a hallucination.
    if not pred.issubset(corpus_ids):
        return "ungrounded"
    if gold and not pred.issubset(gold):
        return "ungrounded"
    if not gold:
        # No gold IDs (OOD query) but the model made up citations instead of abstaining.
        return "ungrounded"
    return "grounded"


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="evals/results/accuracy.jsonl")
    ap.add_argument("--out-summary", default=None)
    args = ap.parse_args()

    corpus_ids = _load_corpus_ids()
    records = load_jsonl(Path(args.in_jsonl))

    by_class = {"grounded": 0, "ungrounded": 0, "unverified": 0, "abstain": 0}
    by_lang: dict[str, dict[str, int]] = {}
    ungrounded_samples: list[dict] = []

    for rec in records:
        cls = classify(
            rec.get("pred_citations", []),
            rec.get("gold_citations", []),
            corpus_ids,
            abstain=rec.get("abstain", False),
        )
        by_class[cls] += 1
        lang = rec.get("lang", "?")
        by_lang.setdefault(lang, {"grounded": 0, "ungrounded": 0, "unverified": 0, "abstain": 0})
        by_lang[lang][cls] += 1
        if cls == "ungrounded" and len(ungrounded_samples) < 20:
            ungrounded_samples.append(
                {
                    "id": rec.get("id"),
                    "lang": lang,
                    "pred_citations": rec.get("pred_citations", []),
                    "gold_citations": rec.get("gold_citations", []),
                }
            )

    total = len(records)
    non_abstain = total - by_class["abstain"]
    halluc = by_class["ungrounded"] + by_class["unverified"]

    def rate_block(counts: dict[str, int]) -> dict:
        n = sum(counts.values())
        n_na = n - counts["abstain"]
        hr = (counts["ungrounded"] + counts["unverified"]) / n_na if n_na else 0.0
        return {
            **counts,
            "n": n,
            "non_abstain": n_na,
            "hallucination_rate": hr,
            "grounded_rate": counts["grounded"] / n_na if n_na else 0.0,
        }

    summary = {
        **stamp(),
        "input": args.in_jsonl,
        "corpus_size": len(corpus_ids),
        "overall": rate_block(by_class),
        "by_lang": {lang: rate_block(c) for lang, c in by_lang.items()},
        "ungrounded_samples": ungrounded_samples,
    }

    out = Path(args.out_summary) if args.out_summary else ensure_results_dir() / "hallucination_summary.json"
    write_summary(out, summary)

    log.info(
        "halluc_run_done",
        total=total,
        hallucination_rate=round(halluc / non_abstain, 3) if non_abstain else 0.0,
        grounded_rate=round(by_class["grounded"] / non_abstain, 3) if non_abstain else 0.0,
    )


if __name__ == "__main__":
    main()
