"""Generation- and retrieval-level metrics. Pure functions, no network calls.

Mirrors what RAG papers report:
  - Token-level F1, Exact Match (SQuAD-style normalization)
  - Fuzzy match (rapidfuzz token-sort ratio)
  - List-Component F1 for multi-item answers
  - Retrieval Hit@k / Recall@k
All functions are pure and offline-testable. An LLM-as-judge (see judge.py) covers
correct paraphrases that these lexical metrics miss.
"""
from __future__ import annotations

import re
import string
from collections import Counter

from rapidfuzz import fuzz

_ARTICLES = {"a", "an", "the"}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """SQuAD-style: lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------
def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-level F1."""
    pred_toks, gold_toks = _tokens(pred), _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def fuzzy_match(pred: str, gold: str) -> float:
    """Token-sort fuzzy ratio in [0, 1]."""
    return fuzz.token_sort_ratio(normalize(pred), normalize(gold)) / 100.0


# ---------------------------------------------------------------------------
# List-Component F1
# ---------------------------------------------------------------------------
def split_items(text: str) -> list[str]:
    """Split a list-style answer into items on commas/semicolons/newlines/bullets."""
    parts = re.split(r"[\n;,]|(?:\s+and\s+)|(?:^|\s)[-*•]\s+", text)
    items = [normalize(p) for p in parts if p and normalize(p)]
    return items


def list_component_f1(pred: str, gold_items: list[str], *, fuzz_threshold: float = 0.8) -> dict[str, float]:
    """Precision/recall/F1 over list items, matching with fuzzy overlap."""
    pred_items = split_items(pred)
    gold = [normalize(g) for g in gold_items if normalize(g)]
    if not gold and not pred_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold or not pred_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gold: set[int] = set()
    tp = 0
    for p in pred_items:
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            # token_set_ratio credits a gold term embedded in a longer predicted
            # chunk (e.g. "risk factors obesity" vs "obesity"), which token_sort
            # would miss.
            if fuzz.token_set_ratio(p, g) / 100.0 >= fuzz_threshold:
                matched_gold.add(gi)
                tp += 1
                break
    precision = tp / len(pred_items)
    recall = tp / len(gold)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def retrieval_hit(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """1.0 if any gold node was retrieved (hit), else 0.0."""
    return 1.0 if set(retrieved_ids) & set(gold_ids) else 0.0


def retrieval_recall(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Fraction of gold nodes that were retrieved."""
    gold = set(gold_ids)
    if not gold:
        return 1.0
    return len(set(retrieved_ids) & gold) / len(gold)


def page_hit(retrieved_pages: list[int], gold_page: int | None) -> float:
    """Fallback retrieval signal when node ids differ: did we land on the gold page?"""
    if gold_page is None:
        return 0.0
    return 1.0 if gold_page in set(retrieved_pages) else 0.0
