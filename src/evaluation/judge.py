"""LLM-as-judge for borderline answers, with reasoning logged for transparency.

Only invoked when the lexical metrics are inconclusive — token-F1 doesn't clearly mark
the answer correct, yet it still lexically resembles the reference (high fuzzy ratio),
which is the signature of a correct paraphrase. This keeps judge calls (and quota) low
while catching the cases lexical metrics miss.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import config
from src.llm import gemini_client

_PROMPT = """You are grading a model's answer against a reference answer for a clinical
guideline question. Judge MEANING, not wording — paraphrases that preserve the clinical
facts (numbers, thresholds, age bands, listed items) are correct.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

MODEL ANSWER:
{prediction}

Return ONLY JSON:
{{"verdict": "correct" | "partial" | "incorrect", "reasoning": "<one or two sentences>"}}"""


@dataclass
class Judgment:
    verdict: str
    reasoning: str
    score: float  # correct=1.0, partial=0.5, incorrect=0.0


_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def is_borderline(token_f1: float, fuzzy_match: float) -> bool:
    """Inconclusive lexical signal -> worth judging.

    token-F1 below the 'clearly correct' threshold, but a high fuzzy ratio shows the
    answer still resembles the reference (likely a correct paraphrase or reordering).
    """
    return token_f1 < config.JUDGE_F1_THRESHOLD and fuzzy_match >= config.JUDGE_FUZZY_TRIGGER


def judge(question: str, reference: str, prediction: str) -> Judgment:
    prompt = _PROMPT.format(question=question, reference=reference, prediction=prediction)
    try:
        data = gemini_client.generate_json(prompt)
        verdict = str(data.get("verdict", "incorrect")).lower().strip()
        reasoning = str(data.get("reasoning", "")).strip()
    except Exception as exc:  # noqa: BLE001
        verdict, reasoning = "incorrect", f"judge error: {exc}"
    if verdict not in _SCORE:
        verdict = "incorrect"
    return Judgment(verdict=verdict, reasoning=reasoning, score=_SCORE[verdict])


def log_judgment(record: dict) -> None:
    path = config.RESULTS_DIR / "judge_log.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
