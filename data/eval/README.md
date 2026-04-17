# Eval Dataset Card

| Field | Value |
|---|---|
| Domain | TechNova (synthetic SaaS / e-commerce) |
| Languages | English (en), Spanish (es), Hindi (hi) |
| FAQ entries per language | **140** (target ≥ 120) |
| Eval queries per language | **310** each (target ≥ 165) |
| Total eval queries | **930** (target ≥ 500) |
| License | CC-BY-4.0 (synthetic, generated with Mistral Small) |
| Generation date | 2026-04-21 |
| Two-pass reviewer κ | _pending second review pass (24h+ gap)_ |

## Sources & files

| File | Rows | Purpose |
|---|---:|---|
| `data/raw/faqs_en.jsonl` | 140 | English FAQ corpus |
| `data/raw/faqs_es.jsonl` | 140 | Spanish translations (same IDs) |
| `data/raw/faqs_hi.jsonl` | 140 | Hindi translations (same IDs) |
| `data/eval/gold_qa_en.jsonl` | 310 | English eval queries |
| `data/eval/gold_qa_es.jsonl` | 310 | Spanish eval queries |
| `data/eval/gold_qa_hi.jsonl` | 310 | Hindi eval queries |

## FAQ category distribution (per language)

| Category | Count | Scope |
|---|---:|---|
| billing | 25 | pricing, invoices, refunds, payment methods, tax, currency |
| account | 25 | sign-up, login, password, 2FA, profile, deletion |
| shipping | 25 | delivery windows, tracking, carriers, international, addresses |
| returns | 20 | RMA, return windows, refund timelines, damaged items, exchanges |
| product | 25 | features, compatibility, setup, troubleshooting, warranty, specs |
| sales | 20 | discounts, promo codes, B2B, trials, comparisons, contact sales |
| **Total** | **140** | |

IDs are consistent across languages (e.g. `faq-billing-001` exists in all three), enabling same-FAQ retrieval across the three indices built in Stage 2.

## Eval query composition (per language)

| Field | Value |
|---|---|
| Variant mix | ~140 direct paraphrase + ~140 indirect + 30 out-of-domain distractor |
| Difficulty mix (avg across langs) | easy 45%, medium 35%, hard 20% |
| Category distribution | billing 50, account 50, shipping 50, product 50, returns 40, sales 40, **ood 30** |

### Difficulty by language

| Lang | easy | medium | hard |
|---|---:|---:|---:|
| en | 140 | 109 | 61 |
| es | 140 | 102 | 68 |
| hi | 140 | 115 | 55 |

### Per-query schema

```json
{
  "id": "qa-en-0001",
  "lang": "en",
  "query": "Can I switch from monthly to annual billing?",
  "gold_answer": "You can upgrade or downgrade your plan anytime...",
  "relevant_faq_ids": ["faq-billing-001"],
  "category": "billing",
  "difficulty": "easy"
}
```

For OOD distractors: `relevant_faq_ids = []`, `category = "ood"`, `gold_answer` is a canonical abstention string in the target language.

## License & synthesis notes

- All FAQs, translations, and eval queries are synthetic. TechNova is fictional.
- Released under CC-BY-4.0. Redistribution and derivative works welcomed with attribution.
- Generation model: `mistral-small-latest` (Mistral la Plateforme), temperature 0.2–0.7 depending on script (see `data/eval/annotation_pipeline.md`).

## Known caveats

- **No native-speaker validation** (Spanish, Hindi) has been performed. Translations are plausible on spot-check and consistent in script, but a professional translator review is future work.
- **Mistral-generated "indirect" queries** occasionally regress toward the source wording. The human reviewer in `ingest/annotate.py` corrects these.
- **Distractor quality** varies: the "borderline out-of-domain" ones (e.g., "what's the best laptop brand?") are intentionally kept since they test abstention robustness.
- **Solo-reviewer agreement** is not equivalent to a true multi-annotator kappa; see `annotation_pipeline.md`.
