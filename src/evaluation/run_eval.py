"""Run the full evaluation over the QA dataset and aggregate results.

Pipeline per question: retrieve -> generate -> score (lexical + retrieval)
-> judge-if-borderline. Writes per-question rows to results/raw_results.jsonl and
aggregates to results/metrics_by_document.csv and metrics_by_question_type.csv.

Resumable: questions already present in raw_results.jsonl are skipped, and all Gemini
calls are cached, so re-running costs ~zero quota.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.evaluation import judge as judge_mod
from src.evaluation import metrics
from src.generation.answer import answer_question
from src.llm.gemini_client import QuotaExhausted, TransientError

RAW_PATH = config.RESULTS_DIR / "raw_results.jsonl"

NUMERIC_METRICS = [
    "exact_match",
    "token_f1",
    "fuzzy_match",
    "list_f1",
    "retrieval_hit",
    "retrieval_recall",
    "judge_score",
    "final_correct",
]


def _load_qa() -> list[dict[str, Any]]:
    if not config.QA_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"QA dataset not found at {config.QA_DATASET_CSV}. "
            "Run `python -m src.cli draft-qa` and verify it first."
        )
    with open(config.QA_DATASET_CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _done_ids() -> set[str]:
    if not RAW_PATH.exists():
        return set()
    ids = set()
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["question_id"])
    return ids


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    qid = row["question_id"]
    slug = row["document"]
    question = row["question"]
    reference = row["reference_answer"]
    qtype = row.get("question_type", "information_extraction")
    gold_ids = [row["gold_node_id"]] if row.get("gold_node_id") else []
    gold_page = int(row["gold_page"]) if str(row.get("gold_page", "")).strip().isdigit() else None

    result = answer_question(slug, question, question_type=qtype)
    pred = result.answer

    em = metrics.exact_match(pred, reference)
    f1 = metrics.token_f1(pred, reference)
    fuzzy = metrics.fuzzy_match(pred, reference)

    list_f1 = ""
    if qtype == "list_based" and row.get("answer_items"):
        gold_items = [s for s in str(row["answer_items"]).split("|") if s.strip()]
        list_f1 = metrics.list_component_f1(pred, gold_items)["f1"]

    ret_ids = result.used_node_ids
    ret_hit = metrics.retrieval_hit(ret_ids, gold_ids)
    if ret_hit == 0.0 and gold_page is not None:
        ret_hit = metrics.page_hit(result.used_pages, gold_page)
    ret_recall = metrics.retrieval_recall(ret_ids, gold_ids)

    # Borderline -> LLM judge
    judge_score = ""
    judge_verdict = ""
    if judge_mod.is_borderline(f1, fuzzy):
        j = judge_mod.judge(question, reference, pred)
        judge_score, judge_verdict = j.score, j.verdict
        judge_mod.log_judgment({
            "question_id": qid, "question": question, "reference": reference,
            "prediction": pred, "token_f1": f1, "fuzzy_match": fuzzy,
            "verdict": j.verdict, "reasoning": j.reasoning,
        })

    # Final correctness: lexical clearly-correct OR judge says (partial+) correct.
    final_correct = 1.0 if (
        em == 1.0 or f1 >= config.JUDGE_F1_THRESHOLD
        or (judge_score != "" and judge_score >= 0.5)
    ) else 0.0

    return {
        "question_id": qid,
        "document": slug,
        "question_type": qtype,
        "prediction": pred,
        "reference": reference,
        "grounding_mode": result.grounding_mode,
        "retrieved_node_ids": ",".join(ret_ids),
        "confidence": round(result.confidence, 3),
        "exact_match": em,
        "token_f1": round(f1, 4),
        "fuzzy_match": round(fuzzy, 4),
        "list_f1": list_f1 if list_f1 == "" else round(float(list_f1), 4),
        "retrieval_hit": ret_hit,
        "retrieval_recall": round(ret_recall, 4),
        "judge_score": judge_score,
        "judge_verdict": judge_verdict,
        "final_correct": final_correct,
    }


def run(*, limit: int | None = None) -> None:
    qa = _load_qa()
    done = _done_ids()
    pending = [r for r in qa if r["question_id"] not in done]
    if limit:
        pending = pending[:limit]

    print(f"{len(qa)} questions total; {len(done)} done; evaluating {len(pending)} now.")
    halted = False
    with open(RAW_PATH, "a", encoding="utf-8") as fh:
        for i, row in enumerate(pending, 1):
            try:
                rec = evaluate_row(row)
            except (QuotaExhausted, TransientError) as exc:
                # Stop cleanly WITHOUT writing a row for this question so it is retried
                # (not scored as 0) on the next run. Distinguish the two causes:
                #   QuotaExhausted -> daily free-tier cap; wait for reset.
                #   TransientError -> model overloaded (5xx); just re-run shortly.
                reached = len(done) + i - 1
                if isinstance(exc, QuotaExhausted):
                    print(f"\n[quota] Daily Gemini quota reached at {row['question_id']} "
                          f"({reached}/{len(qa)} done). Re-run after the quota resets to resume.")
                else:
                    print(f"\n[transient] Gemini overloaded (5xx) at {row['question_id']} "
                          f"({reached}/{len(qa)} done). Re-run `python -m src.cli eval` shortly to resume.")
                print(f"[detail] {str(exc)[:160]}")
                halted = True
                break
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{i}/{len(pending)}] {rec['question_id']} "
                  f"f1={rec['token_f1']} fuzzy={rec['fuzzy_match']} hit={rec['retrieval_hit']}")

    aggregate()
    if halted:
        print("\nPartial run: aggregates above cover completed questions only.")


def aggregate() -> None:
    """Build per-document and per-question-type metric tables from raw results."""
    if not RAW_PATH.exists():
        print("No raw results to aggregate.")
        return
    rows = [json.loads(l) for l in RAW_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    for col in NUMERIC_METRICS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    metric_cols = [c for c in NUMERIC_METRICS if c in df]

    by_doc = df.groupby("document")[metric_cols].mean().round(4)
    by_doc["n"] = df.groupby("document").size()
    by_doc.to_csv(config.RESULTS_DIR / "metrics_by_document.csv")

    by_type = df.groupby("question_type")[metric_cols].mean().round(4)
    by_type["n"] = df.groupby("question_type").size()
    by_type.to_csv(config.RESULTS_DIR / "metrics_by_question_type.csv")

    overall = df[metric_cols].mean().round(4).to_dict()
    overall["n"] = len(df)
    (config.RESULTS_DIR / "metrics_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print("\nOverall:", json.dumps(overall, indent=2))
    print(f"Wrote metrics_by_document.csv, metrics_by_question_type.csv to {config.RESULTS_DIR}")


if __name__ == "__main__":
    run()
