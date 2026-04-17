"""Shared Mistral client + retry helper for ingest scripts.

Keeps rate-limit handling in one place: Mistral free tier is ~1 RPS, so we
pace requests and retry on 429/5xx with exponential backoff.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import SDKError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "mistral-small-latest")
_MIN_INTERVAL_S = 1.1  # Mistral free tier ≈ 1 RPS; stay under it.
_last_call_ts = 0.0


def _client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set in environment or .env")
    return Mistral(api_key=key)


def _pace() -> None:
    global _last_call_ts
    delta = time.monotonic() - _last_call_ts
    if delta < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - delta)
    _last_call_ts = time.monotonic()


@retry(
    retry=retry_if_exception_type(SDKError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def chat_json(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Any:
    """Call Mistral with JSON mode and return the parsed JSON object.

    Raises on non-JSON output after retries.
    """
    _pace()
    client = _client()
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("Non-JSON response (first 200 chars): %s", raw[:200])
        raise SDKError(f"Model returned non-JSON: {exc}") from exc


def chat_text(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """Plain text completion for translation tasks where JSON mode is overkill."""

    @retry(
        retry=retry_if_exception_type(SDKError),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call() -> str:
        _pace()
        client = _client()
        resp = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    return _call()
