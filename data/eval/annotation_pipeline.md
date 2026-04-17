# Eval Set Annotation Pipeline

## Overview

The gold QA eval set in `data/eval/gold_qa_{en,es,hi}.jsonl` is constructed in two phases:

1. **Automated candidate generation** — `ingest/generate_eval.py` emits ~310 candidate queries per language using Mistral Small in JSON mode, with provenance fields (`relevant_faq_ids`, `category`, `difficulty`) grounded in the FAQ corpus.
2. **Human-in-the-loop review** — `ingest/annotate.py` walks each candidate and records an `accept / edit / reject / skip` decision into a sidecar `.reviewed.jsonl` file.

Source files are immutable; the sidecar is the auditable record of which candidates entered the final eval.

## Query composition

Per language (EN / ES / HI), the generator produces:

| Variant | Count | Source | `gold_answer` | `relevant_faq_ids` |
|---|---|---|---|---|
| `direct` paraphrase | ~140 | One per FAQ, near-rewrite of the FAQ question | FAQ's answer | `[faq-<cat>-NNN]` |
| `indirect` paraphrase | ~140 | One per FAQ, oblique phrasing (customer describes the situation) | FAQ's answer | `[faq-<cat>-NNN]` |
| `distractor` (out-of-domain) | 30 | Fresh generation, topic unrelated to TechNova | Canonical abstention string | `[]` |

Total per language: ~310. Total across languages: ~930 (target was ≥ 500, ≥ 165 each).

## Candidate generation prompt (summary)

- **System prompt** constrains the model to realistic customer-support phrasing in the target language, explicitly discouraging FAQ-style "Q1:" prefixes.
- **User prompt** pins the output schema (strict JSON via Mistral's `response_format={"type": "json_object"}`) and demands exactly `2 × |batch|` queries — one `direct` + one `indirect` per FAQ.
- **Difficulty** is self-labeled by the model, with the rubric given inline (`easy` = keyword overlap, `medium` = some drift, `hard` = oblique / contextual).
- **Dedup** runs per batch on normalized (lowercased, stripped) query text before write.
- **Seed:** `random.Random(42 + hash(lang))` seeds the FAQ shuffle so paraphrase ordering is reproducible.

## Two-pass solo review protocol

The PRD calls for a Cohen's kappa ≥ 0.7 inter-annotator agreement. Because this is a solo project, we approximate with a **two-pass self-review with ≥ 24-hour separation**:

1. **Pass 1** — first read-through, same day as generation.
   ```
   python -m ingest.annotate review data/eval/gold_qa_en.jsonl --pass 1
   ```
2. **Pass 2** — second read-through, ≥ 24h later, without re-reading pass-1 decisions first.
   ```
   python -m ingest.annotate review data/eval/gold_qa_en.jsonl --pass 2
   ```
3. **Agreement check** — the tool's `status` command prints the overlap:
   ```
   python -m ingest.annotate status data/eval/gold_qa_en.jsonl
   ```

This yields a decision matrix per ID. Cohen's kappa is computed post-hoc (the sidecar stores the raw labels; a small script `scripts/cohen_kappa.py` — TBD — will compute it over the 4-class label space).

### Decision semantics

| Decision | Meaning | Included in final eval? |
|---|---|---|
| `accept` | Candidate is usable as-is. | Yes |
| `edit` | Candidate is close but needs fixing; reviewer provides the corrected query or gold_answer. | Yes (edited version) |
| `reject` | Candidate is broken (wrong language, nonsensical, duplicates intent of another row). | No |
| `skip` | Undecided; revisit later. | No |

Only rows with `_review_status ∈ {accept, edit}` from Pass 2 enter the final eval that Stage 9 scores against. Pass 1 decisions feed the kappa calculation but do not gate inclusion — a candidate could be rejected in Pass 1 and accepted in Pass 2, and we'd use the Pass 2 decision.

## Known limits of this protocol

- A solo reviewer cannot produce a true inter-annotator kappa; the two-pass self-agreement is a weaker signal that captures *consistency over time* rather than *agreement across people*. This is noted explicitly in the dataset card.
- The 24-hour gap is recorded via `_review_ts` on each sidecar row so it can be verified after the fact.
- Mistral-generated "indirect" queries occasionally slip back toward direct phrasing; the reviewer demotes those to `direct` via `edit` when caught.
- Distractor queries sometimes drift too close to TechNova's domain (e.g., "what's the best laptop?" — borderline). Reviewer rejects or rewrites these.

## Reproducibility

- FAQ corpus: `ingest/generate_faqs.py` (seed deterministic by category order; temperature=0.3).
- Translations: `ingest/translate_faqs.py` (temperature=0.2, batch size 10).
- Eval queries: `ingest/generate_eval.py --seed 42` (shuffle seeded; temperature=0.5 for paraphrase, 0.7 for distractors).

Rerunning all three scripts on a fresh clone will produce a *similar* but not byte-identical corpus because Mistral outputs are not fully deterministic even at temperature 0. The committed JSONL files are the canonical artifacts.
