"""Vectorless retrieval: LLM reasons over the PageIndex tree outline to pick nodes.

This is the heart of "PageIndex-style" retrieval — instead of embedding similarity,
we hand Gemini an indented outline (node_id · title · summary) and ask it to select
the node(s) most likely to contain the answer, with a relevance score and rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
from src import tree_utils
from src.llm import gemini_client

_PROMPT = """You are a retrieval engine for a long clinical-guideline document.
Below is the document's section TREE. Each line is: [node_id] Title — summary.

Your job: pick the {k} node(s) whose section is MOST LIKELY to contain the answer to
the user's question. Reason like a clinician scanning a table of contents.

QUESTION:
{question}

DOCUMENT TREE:
{outline}

Return ONLY a JSON array (most relevant first), each item:
{{"node_id": "<id>", "relevance": <0.0-1.0>, "reason": "<one short sentence>"}}
Pick at most {k} nodes. Do not invent node_ids that are not in the tree."""


@dataclass
class Retrieved:
    node_id: str
    title: str
    page_index: int | None
    relevance: float
    reason: str
    text: str = field(repr=False, default="")


def search(slug: str, question: str, *, k: int | None = None) -> list[Retrieved]:
    """Return the top-k retrieved nodes for a question against a document's tree."""
    k = k or config.RETRIEVAL_TOP_K
    tree = tree_utils.load_tree(slug)
    outline = tree_utils.render_outline(tree, with_summary=True)
    prompt = _PROMPT.format(k=k, question=question.strip(), outline=outline)

    try:
        raw = gemini_client.generate_json(prompt)
    except gemini_client.QuotaExhausted:
        # Don't fabricate an empty retrieval on a quota wall — let the caller halt
        # cleanly so the question is retried later instead of scored as a wrong "0".
        raise
    except Exception:  # noqa: BLE001 - genuine parse/other error: degrade to no retrieval
        raw = []
    if not isinstance(raw, list):
        raw = []

    results: list[Retrieved] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("node_id", "")).strip()
        if not nid or nid in seen:
            continue
        node = tree_utils.find_node(tree, nid)
        if node is None:  # model hallucinated an id
            continue
        seen.add(nid)
        results.append(
            Retrieved(
                node_id=nid,
                title=(node.get("title") or "").strip(),
                page_index=tree_utils.node_page(node),
                relevance=float(item.get("relevance", 0.0) or 0.0),
                reason=str(item.get("reason", "")).strip(),
                text=tree_utils.node_text(node),
            )
        )
        if len(results) >= k:
            break
    return results


def context_from(retrieved: list[Retrieved], *, max_chars: int = 12000) -> str:
    """Concatenate retrieved node texts into a bounded context block."""
    blocks: list[str] = []
    used = 0
    for r in retrieved:
        header = f"[{r.node_id}] {r.title} (page {r.page_index})"
        body = r.text
        chunk = f"{header}\n{body}"
        if used + len(chunk) > max_chars:
            chunk = chunk[: max(0, max_chars - used)]
        blocks.append(chunk)
        used += len(chunk)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(blocks)
