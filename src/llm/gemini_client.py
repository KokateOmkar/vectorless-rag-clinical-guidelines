"""Shared Gemini wrapper: rate-limiting, retry/backoff, on-disk caching, model fallback.

- Stay inside the free tier: a client-side limiter caps requests per minute.
- Never pay twice: identical (model, prompt) calls are cached to data/cache/, so
  re-running the evaluation costs zero quota.
- Survive quota hiccups: retry with exponential backoff on 429/5xx, and fall back from
  the primary model to GEMINI_FALLBACK_MODEL when the daily cap is hit.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

# ---------------------------------------------------------------------------
# Rate limiter: simple global min-interval between calls (RPM -> seconds/call).
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
            self._last = time.monotonic()


_limiter = _RateLimiter(config.GEMINI_RPM)
_client: "genai.Client | None" = None


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_gemini_key())
    return _client


class QuotaExhausted(Exception):
    """Raised when a model returns a 429 / ResourceExhausted (daily or rate quota)."""


class TransientError(Exception):
    """Raised on a server-side hiccup (500/503/overloaded) — worth retrying, but NOT a
    quota wall, so it must never trigger a model fallback or end a run prematurely."""


def _raise_classified(exc: Exception) -> None:
    """Re-raise a provider error as QuotaExhausted (429) or TransientError (5xx); pass
    anything else through unchanged."""
    msg = str(exc).lower()
    if any(k in msg for k in ("429", "quota", "resource exhausted", "rate limit")):
        raise QuotaExhausted(str(exc)) from exc
    if any(k in msg for k in ("500", "503", "unavailable", "overloaded", "internal error")):
        raise TransientError(str(exc)) from exc
    raise exc


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def _cache_path(kind: str, key_material: str) -> Path:
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
    return config.CACHE_DIR / f"{kind}_{digest}.json"


def _read_cache(path: Path) -> Any | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _write_cache(path: Path, value: Any) -> None:
    path.write_text(json.dumps({"value": value}, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((QuotaExhausted, TransientError)),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _call_model(model_name: str, prompt: str, temperature: float) -> str:
    client = _get_client()
    _limiter.wait()
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return resp.text or ""
    except Exception as exc:  # noqa: BLE001 - classify into quota vs transient for retry
        _raise_classified(exc)


def generate(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str | None = None,
    use_cache: bool = True,
) -> str:
    """Generate text. Cached by (model, temperature, prompt). Falls back on quota."""
    primary = model or config.GEMINI_MODEL
    cache_key = f"{primary}|{temperature}|{prompt}"
    cache_file = _cache_path("gen", cache_key)
    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            return cached

    try:
        text = _call_model(primary, prompt, temperature)
    except QuotaExhausted:
        # Primary daily cap hit -> try the fallback model once (if a distinct one is set).
        fb = config.GEMINI_FALLBACK_MODEL
        if fb and fb != primary:
            text = _call_model(fb, prompt, temperature)
        else:
            raise
    # TransientError propagates: a 5xx blip is not a quota wall, and the fallback model
    # may be just as overloaded (or unavailable on free tier).

    if use_cache:
        _write_cache(cache_file, text)
    return text


@retry(
    retry=retry_if_exception_type((QuotaExhausted, TransientError)),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _call_multimodal(model_name: str, prompt: str, images: list[bytes], mime_type: str, temperature: float) -> str:
    client = _get_client()
    _limiter.wait()
    parts = [types.Part.from_bytes(data=img, mime_type=mime_type) for img in images]
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[prompt, *parts],
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return resp.text or ""
    except Exception as exc:  # noqa: BLE001 - classify into quota vs transient for retry
        _raise_classified(exc)


def generate_multimodal(
    prompt: str,
    images: list[bytes],
    *,
    mime_type: str = "image/png",
    temperature: float = 0.0,
    model: str | None = None,
    use_cache: bool = True,
) -> str:
    """Generate text from a prompt + page image(s) (Gemini vision). Cached by content hash."""
    primary = model or config.GEMINI_MODEL
    img_digest = hashlib.sha256(b"".join(images)).hexdigest()[:16]
    cache_key = f"{primary}|{temperature}|{mime_type}|{img_digest}|{prompt}"
    cache_file = _cache_path("mm", cache_key)
    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            return cached

    try:
        text = _call_multimodal(primary, prompt, images, mime_type, temperature)
    except QuotaExhausted:
        fb = config.GEMINI_FALLBACK_MODEL
        if fb and fb != primary:
            text = _call_multimodal(fb, prompt, images, mime_type, temperature)
        else:
            raise
    # TransientError propagates (see generate()).

    if use_cache:
        _write_cache(cache_file, text)
    return text


def generate_json(prompt: str, *, temperature: float = 0.0, model: str | None = None) -> Any:
    """Generate and parse a JSON object/array, tolerating ```json fences."""
    raw = generate(prompt, temperature=temperature, model=model)
    return _parse_json(raw)


def _parse_json(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else raw
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: find first {..} or [..] block
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = text.find(opener), text.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    continue
        raise
