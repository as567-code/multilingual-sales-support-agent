"""Stage-9 eval-harness smoke tests.

Exercise the three runners in offline/local mode only — the live-Mistral
accuracy run is triggered manually (not in CI) because it consumes API
credits. These tests verify:
  * harness loads the gold corpus and stratifies a sample
  * the offline accuracy runner produces JSONL + summary with real numbers
  * the safety runner hits the Stage-5 gates (≥95% injection, ≥98% PII)
  * the hallucination runner reads the accuracy JSONL and classifies records
  * the report aggregator renders markdown from the three summaries
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.harness import (
    GoldSample,
    citation_prf,
    load_gold,
    stratified_sample,
)


REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


def test_load_gold_returns_all_three_langs():
    gold = load_gold()
    langs = {g.lang for g in gold}
    assert langs == {"en", "es", "hi"}
    assert len(gold) >= 900  # 310 * 3 per the seeded corpus


def test_stratified_sample_respects_quota():
    gold = load_gold()
    sampled = stratified_sample(gold, n_per_lang=5, seed=42)
    by_lang: dict[str, int] = {}
    for g in sampled:
        by_lang[g.lang] = by_lang.get(g.lang, 0) + 1
    assert by_lang == {"en": 5, "es": 5, "hi": 5}


def test_citation_prf_edges():
    # exact match
    p, r, f = citation_prf(["a", "b"], ["a", "b"])
    assert (p, r, f) == (1.0, 1.0, 1.0)
    # empty on both sides — trivially correct (abstain on OOD)
    assert citation_prf([], []) == (1.0, 1.0, 1.0)
    # predicted empty but gold non-empty — total miss
    assert citation_prf([], ["a"]) == (0.0, 0.0, 0.0)
    # partial overlap
    p, r, f = citation_prf(["a", "b"], ["a", "c"])
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(0.5)
    assert f == pytest.approx(0.5)


# --------------------------------------------------------------------------
# runners — offline / pure-local
# --------------------------------------------------------------------------


def _run(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO, check=True, capture_output=True, text=True,
    )


def test_accuracy_runner_offline(tmp_path):
    jsonl = tmp_path / "acc.jsonl"
    summary_path = tmp_path / "acc_summary.json"
    _run(
        "evals.run_accuracy",
        "--offline",
        "--per-lang", "2",
        "--out-jsonl", str(jsonl),
        "--out-summary", str(summary_path),
    )
    assert jsonl.exists() and summary_path.exists()

    records = [json.loads(l) for l in jsonl.read_text().splitlines()]
    assert len(records) == 6  # 2 per lang * 3 langs
    assert {r["lang"] for r in records} == {"en", "es", "hi"}

    summary = json.loads(summary_path.read_text())
    assert summary["offline"] is True
    overall = summary["overall"]
    assert overall["n"] == 6
    # Oracle reasoning + retrieval should easily clear 0.6 F1 on a
    # 2-per-lang sample; we set the bar low to avoid test flakes from the
    # odd edge case in the stratified draw.
    assert overall["citation_f1"] >= 0.6
    assert overall["latency_total_ms"]["p95"] >= 0.0


def test_safety_runner_hits_gates(tmp_path):
    summary_path = tmp_path / "safety.json"
    _run("evals.run_safety", "--out-summary", str(summary_path))
    summary = json.loads(summary_path.read_text())
    assert summary["injection"]["catch_rate"] >= 0.95
    assert summary["injection"]["benign_precision"] == 1.0
    assert summary["pii"]["recall"] >= 0.98


def test_hallucination_runner_reads_accuracy(tmp_path):
    acc_jsonl = tmp_path / "acc.jsonl"
    acc_summary = tmp_path / "acc_summary.json"
    halluc_summary = tmp_path / "halluc.json"

    _run(
        "evals.run_accuracy", "--offline",
        "--per-lang", "2",
        "--out-jsonl", str(acc_jsonl),
        "--out-summary", str(acc_summary),
    )
    _run(
        "evals.run_hallucination",
        "--in-jsonl", str(acc_jsonl),
        "--out-summary", str(halluc_summary),
    )
    summary = json.loads(halluc_summary.read_text())
    overall = summary["overall"]
    assert overall["n"] == 6
    # Oracle LLM by construction only cites retrieved gold IDs, so no ungrounded.
    assert overall["ungrounded"] == 0
    assert "grounded_rate" in overall


def test_report_builds_markdown_from_summaries(tmp_path):
    # seed the three summaries
    (tmp_path / "accuracy_summary.json").write_text(json.dumps({
        "timestamp": "2026-04-21T00:00:00",
        "offline": True,
        "overall": {
            "n": 6, "citation_f1": 0.9, "citation_precision": 0.9, "citation_recall": 0.9,
            "abstain_rate": 0.0, "correct_abstain_rate": 1.0,
            "latency_total_ms": {"p50": 50.0, "p95": 100.0, "mean": 60.0},
            "latency_retrieval_ms": {"p50": 10.0, "p95": 15.0, "mean": 12.0},
            "latency_reasoning_ms": {"p50": 30.0, "p95": 70.0, "mean": 40.0},
            "latency_safety_input_ms": {"p50": 1.0, "p95": 2.0, "mean": 1.5},
            "latency_safety_output_ms": {"p50": 5.0, "p95": 10.0, "mean": 6.0},
        },
        "by_lang": {},
    }))
    (tmp_path / "hallucination_summary.json").write_text(json.dumps({
        "timestamp": "2026-04-21T00:00:00",
        "corpus_size": 100,
        "overall": {
            "grounded": 5, "ungrounded": 0, "unverified": 0, "abstain": 1,
            "n": 6, "non_abstain": 5, "hallucination_rate": 0.0, "grounded_rate": 1.0,
        },
        "by_lang": {},
    }))
    (tmp_path / "safety_summary.json").write_text(json.dumps({
        "timestamp": "2026-04-21T00:00:00",
        "injection": {"n_attacks": 21, "catch_rate": 1.0, "missed": [],
                      "n_benign": 10, "benign_precision": 1.0, "false_positives": []},
        "pii": {"n_entities": 11, "recall": 1.0, "missed": []},
    }))
    out = tmp_path / "report.md"
    _run("evals.report", "--results-dir", str(tmp_path), "--out", str(out))

    text = out.read_text()
    assert "# Multilingual Sales & Support AI" in text
    assert "Accuracy + Latency" in text
    assert "Hallucination" in text
    assert "Safety" in text
    # numbers surfaced
    assert "100.0%" in text or "100%" in text or " 1.0" in text
