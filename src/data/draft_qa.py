"""Draft candidate QA pairs from each indexed tree for the user to verify.

Workflow: I generate 8-10 grounded candidate questions per document (spanning the four
question types) with reference answers and gold node/page references; the user reviews
and corrects them into the final qa_dataset.csv.
"""
from __future__ import annotations

import csv
import json
from typing import Any

import config
from src import tree_utils
from src.llm import gemini_client

QA_FIELDS = [
    "question_id", "document", "question", "reference_answer",
    "gold_node_id", "gold_section", "gold_page", "question_type",
    "answer_items", "notes",
]

_PROMPT = """You are building a QA benchmark from a USPSTF clinical guideline.
Below is the document's section tree with node ids, titles, and text excerpts.

Write {n} high-quality question-answer pairs that a clinician might ask, grounded ONLY
in the provided text. Spread them across these question_type values:
- "numeric_threshold": a specific number/cutoff (e.g., blood-pressure threshold, age to start)
- "age_band_lookup": an answer that depends on an age range or population
- "list_based": an answer that is a list of items (tests, risk factors, etc.)
- "information_extraction": a factual statement / recommendation grade / rationale

For each pair, cite the single node_id whose text contains the answer.

DOCUMENT: {title}
SECTION TREE WITH TEXT:
{outline}

Return ONLY a JSON array of objects:
{{"question": "...", "reference_answer": "...", "gold_node_id": "<id from tree>",
  "question_type": "<one of the four>", "answer_items": ["item1","item2"] }}
Use "answer_items" ONLY for list_based questions (else omit or empty)."""


def _outline_with_text(tree: list[dict[str, Any]], max_chars: int = 9000) -> str:
    """Outline that includes trimmed node text so the model can ground answers."""
    lines: list[str] = []
    used = 0
    for node, depth in tree_utils.iter_nodes(tree):
        indent = "  " * depth
        nid = node.get("node_id", "?")
        title = (node.get("title") or "").strip()
        text = tree_utils.node_text(node)[:600]
        block = f"{indent}[{nid}] {title}\n{indent}    {text}"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines)


def draft_for_document(slug: str, n: int = 9) -> list[dict[str, Any]]:
    tree = tree_utils.load_tree(slug)
    prompt = _PROMPT.format(n=n, title=config.DOCUMENTS.get(slug, slug),
                            outline=_outline_with_text(tree))
    try:
        items = gemini_client.generate_json(prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] drafting failed for {slug}: {exc}")
        return []
    if not isinstance(items, list):
        return []

    rows: list[dict[str, Any]] = []
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict):
            continue
        nid = str(it.get("gold_node_id", "")).strip()
        node = tree_utils.find_node(tree, nid)
        answer_items = it.get("answer_items") or []
        rows.append({
            "question_id": f"{slug}_{i:02d}",
            "document": slug,
            "question": str(it.get("question", "")).strip(),
            "reference_answer": str(it.get("reference_answer", "")).strip(),
            "gold_node_id": nid,
            "gold_section": (node.get("title", "").strip() if node else ""),
            "gold_page": (tree_utils.node_page(node) if node else ""),
            "question_type": str(it.get("question_type", "information_extraction")).strip(),
            "answer_items": "|".join(str(x).strip() for x in answer_items) if answer_items else "",
            "notes": "" if node else "REVIEW: gold_node_id not found in tree",
        })
    return rows


def draft_all(n_per_doc: int = 9) -> None:
    all_rows: list[dict[str, Any]] = []
    for slug in config.DOCUMENTS:
        if not tree_utils.tree_path(slug).exists():
            print(f"  [skip] {slug}: no tree yet")
            continue
        print(f"  drafting {slug} ...")
        all_rows.extend(draft_for_document(slug, n=n_per_doc))

    out_csv = config.QA_DATASET_DRAFT_CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QA_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    (config.QA_DIR / "qa_dataset_DRAFT.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(all_rows)} candidate QA pairs to {out_csv}")
    print("REVIEW and correct, then save the verified file as qa_dataset.csv")


if __name__ == "__main__":
    draft_all()
