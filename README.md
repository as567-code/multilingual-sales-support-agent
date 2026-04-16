# Multilingual Sales & Support Conversational AI Agent

An end-to-end conversational AI agent for sales & customer support that answers user questions in **English, Spanish, and Hindi** over a curated domain FAQ corpus. Uses a three-agent pipeline (Retrieval → Reasoning → Safety) orchestrated via LangChain, with FAISS retrieval, Google Gemini reasoning, and prompt-injection / PII guardrails.

## Hero Metrics

| Metric | Target | Result |
|---|---|---|
| Answer accuracy (EN/ES/HI, 500+ queries) | ≥ 92% | _pending Stage 9_ |
| Hallucination reduction vs. single-shot baseline | ≥ 85% | _pending Stage 9_ |
| Prompt-injection catch rate | ≥ 95% | _pending Stage 5_ |
| PII redaction recall | ≥ 98% | _pending Stage 5_ |
| P95 latency | ≤ 3s | _pending Stage 7_ |
| RAG frameworks compared | ≥ 5 | _pending Stage 10_ |

All numbers are reproduced from committed JSON in `evals/results/`.

## Architecture

```
User query (EN/ES/HI)
      ↓
FastAPI /chat
      ↓
Safety Agent (in)  ──▶  Prompt-Injection Guard + PII Redactor
      ↓
Retrieval Agent    ──▶  FAISS index (multilingual-e5 embeddings)
      ↓
Reasoning Agent    ──▶  Google Gemini 2.0 Flash (zero-shot + few-shot)
      ↓
Safety Agent (out) ──▶  Factuality + PII re-scan
      ↓
   Response
```

Full write-up: [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
cp .env.example .env              # fill GOOGLE_API_KEY, HF_TOKEN
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Stage 2: build the index (after FAQs exist)
python -m ingest.build_faiss --input data/raw --out data/processed

# Stage 7: run the API
uvicorn app.main:app --reload

# Stage 8: or via Docker
docker compose up
```

## Example request

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "How do I cancel my subscription?", "session_id": "demo"}'
```

## Eval reproduction

```bash
python -m evals.run_accuracy        # overall + per-language accuracy
python -m evals.run_hallucination   # RAGAS faithfulness vs. baseline
python -m evals.run_safety          # injection + PII catch rates
python -m evals.framework_benchmark.compare  # 5-framework RAG comparison
```

All scripts write timestamped JSON to `evals/results/` and append a row to `evals/results/summary.csv`.

## Framework comparison

Benchmarks of LangChain, LlamaIndex, Haystack, txtai, and RAGatouille on the same 500-query eval set: see [`docs/framework_comparison.md`](docs/framework_comparison.md).

## License

MIT
