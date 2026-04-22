"""Accuracy + citation-grounding eval over the gold QA corpus.

Metrics (per-language and aggregate):
  * citation_f1, citation_precision, citation_recall — set overlap of
    predicted vs. gold faq IDs.
  * abstain_rate — fraction of queries the pipeline refused to answer.
  * correct_abstain_rate — abstentions on OOD queries (gold_citations
    empty) counted as correct.
  * latency P50/P95 per stage + total.

Outputs:
  * ``evals/results/accuracy.jsonl`` — one line per query (see QueryRecord).
  * ``evals/results/accuracy_summary.json`` — aggregate metrics.

Modes:
  * ``--offline``      use a deterministic fake LLM (default for CI). Gives
                       an upper-bound that isolates retrieval errors.
  * ``--limit N``      first N queries from the stratified sample.
  * ``--per-lang N``   sample size per language (default 30; total 90).
  * ``--langs en,es``  restrict to a subset of languages.

Usage:
  python -m evals.run_accuracy --offline --per-lang 10
  python -m evals.run_accuracy --per-lang 30    # real Mistral
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.chains.orchestrator import SupportOrchestrator
from app.utils.logging import configure_logging, get_logger
from evals.harness import (
    QueryRecord,
    citation_prf,
    ensure_results_dir,
    latency_stats,
    load_gold,
    make_offline_orchestrator,
    run_samples,
    stamp,
    stratified_sample,
    write_jsonl,
    write_summary,
)

log = get_logger("eval.accuracy")


def summarize(records: list[QueryRecord]) -> dict:
    by_lang: dict[str, list[QueryRecord]] = {}
    for r in records:
        by_lang.setdefault(r.lang, []).append(r)

    def bucket(rs: list[QueryRecord]) -> dict:
        if not rs:
            return {}
        prs = [citation_prf(r.pred_citations, r.gold_citations) for r in rs]
        # Abstains without predicted citations shouldn't inflate precision —
        # citation_prf handles that (empty pred ⇒ 0 unless gold is also empty).
        n = len(rs)
        abstain = sum(1 for r in rs if r.abstain)
        correct_abstain = sum(
            1 for r in rs if r.abstain and not r.gold_citations
        )
        expected_abstain = sum(1 for r in rs if not r.gold_citations)
        return {
            "n": n,
            "citation_precision": sum(p for p, _, _ in prs) / n,
            "citation_recall": sum(r for _, r, _ in prs) / n,
            "citation_f1": sum(f for _, _, f in prs) / n,
            "abstain_rate": abstain / n,
            "correct_abstain_rate": (
                correct_abstain / expected_abstain if expected_abstain else 1.0
            ),
            "latency_total_ms": latency_stats(rs, "total_ms").__dict__,
            "latency_retrieval_ms": latency_stats(rs, "retrieval_ms").__dict__,
            "latency_reasoning_ms": latency_stats(rs, "reasoning_ms").__dict__,
            "latency_safety_input_ms": latency_stats(rs, "safety_input_ms").__dict__,
            "latency_safety_output_ms": latency_stats(rs, "safety_output_ms").__dict__,
        }

    return {
        **stamp(),
        "overall": bucket(records),
        "by_lang": {lang: bucket(rs) for lang, rs in by_lang.items()},
    }


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="Use deterministic fake LLM.")
    ap.add_argument("--per-lang", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="Cap total samples (0 = no cap).")
    ap.add_argument("--langs", default="en,es,hi")
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--out-summary", default=None)
    args = ap.parse_args()

    langs = tuple(args.langs.split(","))
    gold_all = load_gold(langs)
    samples = stratified_sample(gold_all, n_per_lang=args.per_lang)
    if args.limit:
        samples = samples[: args.limit]

    log.info("accuracy_run_start", n_samples=len(samples), offline=args.offline, langs=langs)

    if args.offline:
        # Oracle reasoning over the full gold corpus so the fake LLM can
        # look up any query, even ones outside the sampled subset.
        orch: SupportOrchestrator = make_offline_orchestrator(gold_all)
    else:
        orch = SupportOrchestrator()

    records: list[QueryRecord] = list(run_samples(orch, samples))

    results_dir = ensure_results_dir()
    jsonl_path = Path(args.out_jsonl) if args.out_jsonl else results_dir / "accuracy.jsonl"
    summary_path = (
        Path(args.out_summary) if args.out_summary else results_dir / "accuracy_summary.json"
    )
    write_jsonl(jsonl_path, records)
    summary = summarize(records)
    summary["offline"] = args.offline
    write_summary(summary_path, summary)

    log.info(
        "accuracy_run_done",
        n=len(records),
        f1=round(summary["overall"].get("citation_f1", 0.0), 3),
        abstain=round(summary["overall"].get("abstain_rate", 0.0), 3),
        p95=round(summary["overall"]["latency_total_ms"]["p95"], 1),
    )


if __name__ == "__main__":
    main()
