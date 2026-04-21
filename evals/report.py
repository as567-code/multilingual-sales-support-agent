"""Aggregate the three eval summaries into a single markdown report.

Run order:
  python -m evals.run_accuracy [--offline]
  python -m evals.run_hallucination
  python -m evals.run_safety
  python -m evals.report

Writes ``evals/results/eval_report.md``. Any missing summary file is
flagged inline so the report still renders — useful during iteration
when a runner hasn't been executed yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _fmt_ms(x: float) -> str:
    return f"{x:,.1f} ms"


def _fmt_pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _accuracy_section(summary: dict | None) -> str:
    if summary is None:
        return "_accuracy_summary.json not found — run `python -m evals.run_accuracy`._\n"
    overall = summary.get("overall", {})
    by_lang = summary.get("by_lang", {})
    mode = "offline (oracle LLM)" if summary.get("offline") else "live Mistral"
    lines = [
        f"**Mode:** {mode}  •  **Timestamp:** {summary.get('timestamp', '?')}  •  **N:** {overall.get('n', 0)}",
        "",
        "| Language | N | Cit. F1 | Cit. Precision | Cit. Recall | Abstain | Correct Abstain | P50 total | P95 total |",
        "|----------|---:|--------:|---------------:|------------:|--------:|----------------:|----------:|----------:|",
    ]
    rows = [("overall", overall)] + sorted(by_lang.items())
    for name, b in rows:
        if not b:
            continue
        lines.append(
            "| {lang} | {n} | {f1} | {p} | {r} | {ab} | {cab} | {p50} | {p95} |".format(
                lang=name,
                n=b.get("n", 0),
                f1=_fmt_pct(b.get("citation_f1", 0.0)),
                p=_fmt_pct(b.get("citation_precision", 0.0)),
                r=_fmt_pct(b.get("citation_recall", 0.0)),
                ab=_fmt_pct(b.get("abstain_rate", 0.0)),
                cab=_fmt_pct(b.get("correct_abstain_rate", 0.0)),
                p50=_fmt_ms(b.get("latency_total_ms", {}).get("p50", 0.0)),
                p95=_fmt_ms(b.get("latency_total_ms", {}).get("p95", 0.0)),
            )
        )
    # Per-stage latency breakdown (overall only).
    if overall:
        lines += [
            "",
            "**Per-stage latency (overall):**",
            "",
            "| Stage | P50 | P95 | Mean |",
            "|-------|----:|----:|-----:|",
        ]
        for key, label in [
            ("latency_safety_input_ms", "safety_in"),
            ("latency_retrieval_ms", "retrieval"),
            ("latency_reasoning_ms", "reasoning"),
            ("latency_safety_output_ms", "safety_out"),
            ("latency_total_ms", "**total**"),
        ]:
            b = overall.get(key, {})
            lines.append(
                f"| {label} | {_fmt_ms(b.get('p50', 0.0))} | {_fmt_ms(b.get('p95', 0.0))} | {_fmt_ms(b.get('mean', 0.0))} |"
            )
    return "\n".join(lines) + "\n"


def _hallucination_section(summary: dict | None) -> str:
    if summary is None:
        return "_hallucination_summary.json not found — run `python -m evals.run_hallucination`._\n"
    overall = summary.get("overall", {})
    by_lang = summary.get("by_lang", {})
    lines = [
        f"**Corpus size:** {summary.get('corpus_size', 0)} FAQs  •  **N queries:** {overall.get('n', 0)}",
        "",
        "| Language | Grounded | Ungrounded | Unverified | Abstain | Halluc. Rate |",
        "|----------|---------:|-----------:|-----------:|--------:|-------------:|",
    ]
    rows = [("overall", overall)] + sorted(by_lang.items())
    for name, b in rows:
        if not b:
            continue
        lines.append(
            "| {lang} | {g} | {u} | {v} | {a} | {hr} |".format(
                lang=name,
                g=b.get("grounded", 0),
                u=b.get("ungrounded", 0),
                v=b.get("unverified", 0),
                a=b.get("abstain", 0),
                hr=_fmt_pct(b.get("hallucination_rate", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def _safety_section(summary: dict | None) -> str:
    if summary is None:
        return "_safety_summary.json not found — run `python -m evals.run_safety`._\n"
    inj = summary.get("injection", {})
    pii = summary.get("pii", {})
    lines = [
        "**Prompt injection:**",
        "",
        f"- catch rate: **{_fmt_pct(inj.get('catch_rate', 0.0))}** ({inj.get('n_attacks', 0)} attacks)",
        f"- benign precision: **{_fmt_pct(inj.get('benign_precision', 0.0))}** ({inj.get('n_benign', 0)} benign)",
    ]
    if inj.get("missed"):
        lines += ["", "*Missed attacks:*"] + [f"  - `{m}`" for m in inj["missed"][:10]]
    if inj.get("false_positives"):
        lines += ["", "*False positives:*"] + [f"  - `{m}`" for m in inj["false_positives"][:10]]
    lines += [
        "",
        "**PII redaction:**",
        "",
        f"- recall: **{_fmt_pct(pii.get('recall', 0.0))}** ({pii.get('n_entities', 0)} seeded entities)",
    ]
    if pii.get("missed"):
        lines += ["", "*Missed:*"]
        for m in pii["missed"][:10]:
            lines.append(f"  - `{m['text']}` — missing {m['missing']}, found {m['found']}")
    return "\n".join(lines) + "\n"


def build_report(results_dir: Path) -> str:
    acc = _read(results_dir / "accuracy_summary.json")
    halluc = _read(results_dir / "hallucination_summary.json")
    safety = _read(results_dir / "safety_summary.json")

    parts = [
        "# Multilingual Sales & Support AI — Evaluation Report",
        "",
        "Aggregate results across the three eval dimensions defined in the PRD: accuracy",
        "(citation grounding + latency), hallucination rate (citation groundedness classifier),",
        "and safety (injection catch rate + PII redaction recall).",
        "",
        "## 1. Accuracy + Latency",
        "",
        _accuracy_section(acc),
        "",
        "## 2. Hallucination",
        "",
        _hallucination_section(halluc),
        "",
        "## 3. Safety",
        "",
        _safety_section(safety),
        "",
        "---",
        "",
        "_Generated by `python -m evals.report`._",
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="evals/results")
    ap.add_argument("--out", default="evals/results/eval_report.md")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(results_dir)
    Path(args.out).write_text(report)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
