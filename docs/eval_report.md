# Eval methodology

The current numbers are always available at
[`evals/results/eval_report.md`](../evals/results/eval_report.md); this
document explains how those numbers are produced and what they do and
don't say.

## Corpora

### Accuracy / hallucination — gold QA
- **Location**: `data/eval/gold_qa_{en,es,hi}.jsonl`
- **Size**: 310 queries per language × 3 languages = **930 queries**.
- **Shape**: `{id, lang, query, gold_answer, relevant_faq_ids,
  category, difficulty}`.
- **Construction**: generated via `ingest/generate_eval.py` with a
  constrained-prompt LLM run over the curated FAQ corpus; every query
  is tagged with one or more `relevant_faq_ids` taken from the
  corpus. Difficulty mix is roughly 50% easy (near-verbatim paraphrase)
  and 50% hard (reworded, or asks for an implied fact).
- **Stratified sampling**: the harness draws `--per-lang N` queries per
  language with a fixed seed so re-runs are reproducible.

### Safety — injection + PII
- Inline fixtures in `evals/run_safety.py`, mirroring the Stage-5 unit
  tests. **21** injection attacks across EN/ES/HI (instruction
  override, exfiltration, role hijack, delimiter, bypass); **10**
  benign counterfactuals (includes tricky strings like "setup
  instructions" and "system status page"); **8** labeled PII samples
  covering 9 entity occurrences (email, phone, card, SSN, IBAN, IP,
  URL, plus a dual-entity sentence).

## Metrics

### Accuracy
- `citation_f1 / precision / recall` — set-based overlap of predicted
  FAQ IDs vs. `relevant_faq_ids`. Abstentions with empty predicted and
  empty gold are counted as correct (P=R=F1=1); empty predicted vs.
  non-empty gold is a total miss.
- `abstain_rate` — fraction of queries where the pipeline refused to
  answer (either via retrieval score floor or reasoning promotion).
- `correct_abstain_rate` — of the queries with empty `gold_citations`
  (OOD), the fraction the pipeline correctly abstained on.
- `latency_total_ms` and `latency_<stage>_ms` — `{p50, p95, mean}` for
  wall-clock total and each of the four stages.

### Hallucination
Citation-groundedness classifier over the accuracy JSONL:

| class | rule |
|---|---|
| `grounded` | non-abstain, non-empty citations, all predicted ⊆ `relevant_faq_ids` |
| `ungrounded` | non-abstain, citation outside `relevant_faq_ids` or outside the corpus |
| `unverified` | non-abstain, zero citations |
| `abstain` | pipeline returned `abstain=True` |

`hallucination_rate = (ungrounded + unverified) / non_abstain`. This is
a strict upper bound: any citation that isn't on the gold list counts
as a hallucination, even if the model drew from a related FAQ that
would be acceptable to a human reviewer. RAGAS-style faithfulness
would be less strict; this proxy is deterministic and zero-cost.

### Safety
- **Prompt injection** — `catch_rate = 1 - missed / n_attacks` on the
  21 attacks; `benign_precision = 1 - false_positives / n_benign` on
  the 10 benign queries.
- **PII** — for each labeled sample, count the intersection of
  expected entity types with what Presidio found; aggregate as
  `recall = caught / total_expected`.

## Offline mode (the oracle)

Live-Mistral runs are expensive and non-deterministic. The harness
has an `--offline` flag that swaps the reasoning LLM for a
deterministic oracle: it looks up the gold query by substring match
inside the rendered user prompt, then cites only those of the gold's
`relevant_faq_ids` that **actually appear in the retrieval context**.

This gives a clean upper bound: the oracle never fabricates citations
(so `ungrounded = 0` except where retrieval itself surfaced no gold
ID). Anything the oracle *misses* is attributable to retrieval. Used
in CI + the Stage-9 tests.

## Current headline numbers

| Gate | Target | Offline oracle (90 queries) |
|---|---|---|
| Citation F1 | ≥ 92% | **90.0%** |
| Abstain correctness | — | 88.9% |
| Latency P95 (offline, no LLM) | — | 65.6 ms |
| Hallucination rate | ≤ 15% | **7.6%** |
| Injection catch rate | ≥ 95% | **100.0%** |
| Benign precision | 100% | **100.0%** |
| PII recall | ≥ 98% | **100.0%** |

Live Mistral integration (gated on `RUN_MISTRAL_TESTS=1`) measures
total P95 around **~2.2 s**, well under the 3 s PRD budget.

## How to re-run

```bash
python -m evals.run_accuracy --offline --per-lang 30
python -m evals.run_hallucination
python -m evals.run_safety
python -m evals.report        # aggregates → evals/results/eval_report.md
```

Drop `--offline` to evaluate against live Mistral; set `--per-lang` to
a higher number (up to 310) for statistical tightness. Results are
stamped with ISO timestamp + unix so a rolling history can be kept by
renaming the JSON under `evals/results/`.
