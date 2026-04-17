# PRD Deviations Log

Tracks intentional deviations from `PRD_multilingual_sales_support_agent.md`. Every row must cite the motive so reviewers (and future-me) can judge whether the deviation was sound.

| Date | Deviation | Motive | Impact |
|---|---|---|---|
| 2026-04-21 | Reasoning LLM swapped from OpenAI `gpt-4o-mini` → Google `gemini-2.0-flash` (reverted same day) | Free-tier coverage for Stage 9 + Stage 10 eval budget. | Reverted: the supplied Google API key had `free_tier limit=0` quota (Cloud Console key, not AI Studio), so no requests succeeded. |
| 2026-04-21 | Reasoning LLM swapped to Mistral `mistral-small-latest` (current) | Free tier on Mistral la Plateforme covers the eval budget; multilingual smoke test passed for EN/ES/HI including proper Devanagari Hindi. | `pyproject.toml` uses `langchain-mistralai` + `mistralai`; env var `MISTRAL_API_KEY`. Resume bullet reads "Mistral Small via Mistral la Plateforme" in place of "OpenAI gpt-4o-mini". |
| 2026-04-21 | Python pin loosened from `>=3.11,<3.12` → `>=3.11,<3.13` | Host machine (macOS) has 3.12 via Homebrew, no 3.11; all pinned deps are 3.12-compatible. | Ruff `target-version` bumped to `py312`. No runtime impact. |

## Stack after deviations (current)

| Layer | PRD | Actual |
|---|---|---|
| Reasoning LLM | OpenAI gpt-4o-mini | **Mistral `mistral-small-latest`** |
| LangChain binding | `langchain-openai` | `langchain-mistralai` |
| Env var | `OPENAI_API_KEY` | `MISTRAL_API_KEY` |
| Embeddings | intfloat/multilingual-e5-base | unchanged |
| Vector store | FAISS | unchanged |
| Safety | Prompt-Guard-2 + Presidio | unchanged |
| API | FastAPI | unchanged |
| Python | 3.11 | 3.12 |

## Validation plan for the LLM swap

The PRD's 92% accuracy target was set assuming gpt-4o-mini. Mistral Small is a smaller model; the target may be harder to hit. Concrete risks and mitigations:

- **Hindi generation quality** — smoke test (single query per language) passed. Re-validate at Stage 9 on the full 165-query Hindi eval set. If per-language accuracy drops below 0.88, upgrade to `mistral-large-latest` (paid tier) or swap LLM-as-judge to OpenAI.
- **Structured output reliability** — Mistral supports JSON mode via `response_format={"type": "json_object"}`; test at Stage 4 and fall back to manual parsing + Pydantic validation if flaky.
- **Rate limits** — Mistral free tier: 1 RPS, 500k TPM, 1B TPM monthly. Stage 9 (~500 calls) and Stage 10 (~2500 calls across 5 frameworks) fit comfortably.
