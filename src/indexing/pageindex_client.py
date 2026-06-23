"""Thin PageIndex Cloud API client.

Endpoints (https://docs.pageindex.ai/api-reference):
  POST /doc/                         -> {"doc_id": "pi-..."}            (multipart file upload)
  GET  /doc/{doc_id}/                -> status poll
  GET  /doc/{doc_id}/?type=tree...   -> hierarchical tree result

Auth: header `api_key: <KEY>`.
We only ever index here; tree-search reasoning runs on our own Gemini key (free).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

import config


def _headers() -> dict[str, str]:
    return {"api_key": config.require_pageindex_key()}


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def submit(pdf_path: Path) -> str:
    """Upload a PDF for indexing; returns the PageIndex doc_id."""
    url = f"{config.PAGEINDEX_BASE_URL}/doc/"
    with open(pdf_path, "rb") as fh:
        files = {"file": (pdf_path.name, fh, "application/pdf")}
        resp = requests.post(url, headers=_headers(), files=files, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    doc_id = data.get("doc_id")
    if not doc_id:
        raise RuntimeError(f"PageIndex /doc/ returned no doc_id: {data}")
    return doc_id


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def status(doc_id: str) -> dict[str, Any]:
    """Return the raw status payload for a doc_id."""
    url = f"{config.PAGEINDEX_BASE_URL}/doc/{doc_id}/"
    resp = requests.get(url, headers=_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def get_tree(doc_id: str, *, summary: bool = True) -> dict[str, Any]:
    """Fetch the hierarchical tree (with node summaries) for a completed doc."""
    url = f"{config.PAGEINDEX_BASE_URL}/doc/{doc_id}/"
    params = {"type": "tree"}
    if summary:
        params["summary"] = "true"
    resp = requests.get(url, headers=_headers(), params=params, timeout=120)
    resp.raise_for_status()
    return resp.json()


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
def get_ocr_pages(doc_id: str) -> list[dict[str, Any]]:
    """Fetch per-page OCR markdown: [{page_index, markdown, ...}, ...].

    The tree endpoint returns structure + summaries but not full body text; the page
    markdown here is the source of grounding text we attach to nodes. GET-only, so it
    spends no indexing credits.
    """
    url = f"{config.PAGEINDEX_BASE_URL}/doc/{doc_id}/"
    resp = requests.get(url, headers=_headers(), params={"type": "ocr", "format": "page"}, timeout=180)
    resp.raise_for_status()
    return resp.json().get("result") or []


def wait_until_ready(
    doc_id: str, *, poll_interval: int = 10, timeout: int = 1800
) -> dict[str, Any]:
    """Poll until retrieval_ready/completed, or raise on timeout/failure."""
    deadline = time.monotonic() + timeout
    while True:
        payload = status(doc_id)
        state = payload.get("status", "")
        if payload.get("retrieval_ready") or state == "completed":
            return payload
        if state in {"failed", "error"}:
            raise RuntimeError(f"PageIndex indexing failed for {doc_id}: {payload}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"PageIndex indexing timed out for {doc_id} (last: {state})")
        time.sleep(poll_interval)
