"""Validate the verified qa_dataset.csv: schema, types, and gold-node existence."""
from __future__ import annotations

import csv

import config
from src import tree_utils
from src.data.draft_qa import QA_FIELDS


def validate(path=None) -> list[str]:
    path = path or config.QA_DATASET_CSV
    errors: list[str] = []
    if not path.exists():
        return [f"File not found: {path}"]

    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(QA_FIELDS) - set(reader.fieldnames or [])
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
            return errors

        seen_ids: set[str] = set()
        trees: dict[str, list] = {}
        for i, row in enumerate(reader, 2):  # header is line 1
            qid = row["question_id"].strip()
            if not qid:
                errors.append(f"line {i}: empty question_id")
            elif qid in seen_ids:
                errors.append(f"line {i}: duplicate question_id '{qid}'")
            seen_ids.add(qid)

            slug = row["document"].strip()
            if slug not in config.DOCUMENTS:
                errors.append(f"line {i}: unknown document '{slug}'")
            else:
                if slug not in trees and tree_utils.tree_path(slug).exists():
                    trees[slug] = tree_utils.load_tree(slug)
                tree = trees.get(slug)
                nid = row["gold_node_id"].strip()
                if tree is not None and nid and tree_utils.find_node(tree, nid) is None:
                    errors.append(f"line {i}: gold_node_id '{nid}' not in tree '{slug}'")

            qtype = row["question_type"].strip()
            if qtype not in config.QUESTION_TYPES:
                errors.append(f"line {i}: invalid question_type '{qtype}'")
            if not row["question"].strip():
                errors.append(f"line {i}: empty question")
            if not row["reference_answer"].strip():
                errors.append(f"line {i}: empty reference_answer")
            if qtype == "list_based" and not row["answer_items"].strip():
                errors.append(f"line {i}: list_based question has no answer_items")

    return errors


if __name__ == "__main__":
    errs = validate()
    if errs:
        print(f"VALIDATION FAILED ({len(errs)} issues):")
        for e in errs:
            print("  -", e)
    else:
        print("QA dataset is valid.")
