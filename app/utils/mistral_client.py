"""Runtime Mistral client — single-query interactive path.

Distinct from ``ingest/_client.py`` (bulk-generation path with 1-RPS pacing):
the agent pipeline serves one request at a time, so we skip the pacing guard
but keep the tenacity retry on 429/5xx. Settings are read once from
``app.config``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from mistralai import Mistral
from mistralai.models import SDKError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.utils.logging import get_logger

log = get_logger("mistral_client")


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _client() -> Mistral:
    key = get_settings().mistral_api_key
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set (in .env or environment)")
    return Mistral(api_key=key)


@retry(
    retry=retry_if_exception_type(SDKError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def chat_json(
    *,
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict:
    """Call Mistral in JSON mode and return the parsed object.

    Raises SDKError after retries exhaust, or ValueError if the model output
    is not valid JSON (wrapped in SDKError so tenacity re-tries once).
    """
    m = model or get_settings().llm_model
    client = _client()
    resp = client.chat.complete(
        model=m,
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
        log.warning("non_json_response", preview=raw[:200])
        raise SDKError(f"Model returned non-JSON: {exc}") from exc
