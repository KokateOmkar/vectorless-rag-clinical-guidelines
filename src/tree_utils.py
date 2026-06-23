"""Helpers for working with PageIndex trees and the local index manifest.

A PageIndex tree is a list of nodes; each node has:
  {"title", "node_id", "page_index", "text", "summary"?, "nodes": [child, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import config


# ---------------------------------------------------------------------------
# Loading / saving
# ---------------------------------------------------------------------------
def tree_path(slug: str) -> Path:
    return config.TREE_DIR / f"{slug}.json"


def save_tree(slug: str, tree: list[dict[str, Any]]) -> None:
    tree_path(slug).write_text(
        json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_tree(slug: str) -> list[dict[str, Any]]:
    path = tree_path(slug)
    if not path.exists():
        raise FileNotFoundError(f"No tree for '{slug}'. Run indexing first ({path}).")
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest() -> dict[str, Any]:
    if config.INDEX_MANIFEST.exists():
        return json.loads(config.INDEX_MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict[str, Any]) -> None:
    config.INDEX_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------
def iter_nodes(tree: list[dict[str, Any]], _depth: int = 0) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield (node, depth) for every node in document order."""
    for node in tree:
        yield node, _depth
        children = node.get("nodes") or []
        if children:
            yield from iter_nodes(children, _depth + 1)


def count_nodes(tree: list[dict[str, Any]]) -> int:
    return sum(1 for _ in iter_nodes(tree))


def find_node(tree: list[dict[str, Any]], node_id: str) -> dict[str, Any] | None:
    for node, _ in iter_nodes(tree):
        if node.get("node_id") == node_id:
            return node
    return None


def node_text(node: dict[str, Any]) -> str:
    return (node.get("text") or "").strip()


def attach_page_text(tree: list[dict[str, Any]], pages: list[dict[str, Any]], *, max_chars: int = 8000) -> None:
    """Populate each node's `text` from per-page OCR markdown (mutates the tree).

    PageIndex tree nodes carry only title/summary/page_index — no body text. We give
    each node the markdown for the pages it owns: from its start page up to the start
    page of the next section that begins on a later page (so a node always gets at least
    its own page). This grounds answer generation in the real document text.
    """
    page_md = {p.get("page_index"): (p.get("markdown") or "") for p in pages}
    max_page = max((p for p in page_md if p is not None), default=0)

    flat = [node for node, _ in iter_nodes(tree)]
    starts = [(n.get("page_index") or 1) for n in flat]
    for i, node in enumerate(flat):
        start = starts[i]
        end = next((starts[j] for j in range(i + 1, len(flat)) if starts[j] > start), max_page + 1)
        if end <= start:
            end = start + 1
        text = "\n\n".join(page_md.get(p, "") for p in range(start, end)).strip()
        node["text"] = text[:max_chars]


def node_page(node: dict[str, Any]) -> int | None:
    return node.get("page_index")


# ---------------------------------------------------------------------------
# Outline rendering (for LLM tree-search prompts)
# ---------------------------------------------------------------------------
def render_outline(tree: list[dict[str, Any]], *, with_summary: bool = True, max_summary: int = 240) -> str:
    """Render an indented outline of node_id · title · summary for the LLM to reason over."""
    lines: list[str] = []
    for node, depth in iter_nodes(tree):
        indent = "  " * depth
        nid = node.get("node_id", "?")
        title = (node.get("title") or "(untitled)").strip()
        line = f"{indent}- [{nid}] {title}"
        if with_summary:
            summary = (node.get("summary") or "").strip()
            if not summary:
                summary = node_text(node)[:max_summary]
            if summary:
                line += f" — {summary[:max_summary]}"
        lines.append(line)
    return "\n".join(lines)
