"""FastAPI HTTP layer — Stage 7 of the build.

Exposes the 3-agent pipeline behind two endpoints:

  * ``POST /ask``   — primary entry. Takes ``{query, lang_hint?}``, returns
    the ``AssistantResponse`` the orchestrator already produces (redacted
    answer + citations + per-stage latencies + safety flags).
  * ``GET  /health`` — liveness probe. Does NOT touch the pipeline so k8s
    readiness probes don't cascade embedding/Presidio loads on every hit.

Design notes:
  * A single ``SupportOrchestrator`` is built in the lifespan context so
    FAISS + spaCy + sentence-transformers pay their cold start once per
    process rather than once per request. Cold start is ~12s; warmed, the
    P95 is comfortably under the 3s PRD budget.
  * Each request is tagged with a UUID4 ``request_id`` and bound to the
    structlog contextvars so every log line from every agent carries it.
  * Validation is handled by Pydantic; empty queries → 422. The orchestrator
    itself is defensive (strips, detects language, short-circuits), so
    there's no additional business-logic 4xx surface here.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.chains.orchestrator import AssistantResponse, SupportOrchestrator
from app.config import get_settings
from app.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger("api")


# --------------------------------------------------------------------------
# request / response models
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    lang_hint: str | None = Field(default=None, description="Optional BCP-47-ish code: en|es|hi.")


# --------------------------------------------------------------------------
# lifespan — build the orchestrator once, reuse across requests
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Tests may pre-set app.state.orchestrator with a fake before entering
    # the TestClient context — honor that so we don't pay FAISS+spaCy cold
    # start on every test run.
    if not getattr(app.state, "orchestrator", None):
        settings = get_settings()
        log.info("api_startup", llm_model=settings.llm_model, embed_model=settings.embed_model)
        # Build eagerly: FAISS mmap + spaCy load + e5 weights — all amortized here.
        app.state.orchestrator = SupportOrchestrator(settings=settings)
    log.info("api_ready")
    yield
    log.info("api_shutdown")


app = FastAPI(
    title="Multilingual Sales & Support AI",
    version="0.1.0",
    description="EN/ES/HI 3-agent RAG pipeline (Retrieval → Reasoning → Safety).",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# middleware — per-request id + structured logging context
# --------------------------------------------------------------------------


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "request_id": request_id},
        )
    response.headers["x-request-id"] = request_id
    return response


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AssistantResponse)
def ask(payload: AskRequest, request: Request) -> AssistantResponse:
    orch: SupportOrchestrator = request.app.state.orchestrator
    result = orch.ask(payload.query, lang_hint=payload.lang_hint)
    log.info(
        "ask_handled",
        lang=result.lang,
        abstain=result.abstain,
        injection_detected=result.injection_detected,
        pii_redacted=result.pii_redacted,
        total_ms=result.latencies.total_ms,
    )
    return result
