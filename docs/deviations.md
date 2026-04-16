# PRD Deviations Log

Tracks intentional deviations from `PRD_multilingual_sales_support_agent.md`. Every row must cite the motive so reviewers (and future-me) can judge whether the deviation was sound.

| Date | Deviation | Motive | Impact |
|---|---|---|---|
| 2026-04-21 | Reasoning LLM swapped from OpenAI `gpt-4o-mini` → Google `gemini-2.0-flash` | Free tier on Google AI Studio (15 RPM, 1M TPM for Flash) covers the entire Stage 9 + Stage 10 eval budget; PRD §3 expected ~$10 on OpenAI. | `app/agents/reasoning.py` uses `ChatGoogleGenerativeAI`; eval `LLM-as-judge` also Gemini; README + architecture diagram updated. Resume bullet now reads "Google Gemini 2.0 Flash" instead of "OpenAI gpt-4o-mini". |

## Stack after deviations

| Layer | PRD | Actual |
|---|---|---|
| Reasoning LLM | OpenAI gpt-4o-mini | **Google gemini-2.0-flash** |
| LangChain binding | `langchain-openai` | `langchain-google-genai` |
| Env var | `OPENAI_API_KEY` | `GOOGLE_API_KEY` |
| Embeddings | intfloat/multilingual-e5-base | unchanged |
| Vector store | FAISS | unchanged |
| Safety | Prompt-Guard-2 + Presidio | unchanged |
| API | FastAPI | unchanged |

## Notes

- Gemini Flash's multilingual quality is competitive with gpt-4o-mini for EN/ES/HI FAQ QA; re-validate on the 500-query eval in Stage 9 and record any gap.
- If the free tier proves rate-limited during Stage 10 framework benchmarks, fall back to `gemini-1.5-flash-8b` (even more generous) or request a paid key.
