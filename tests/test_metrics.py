"""Offline unit tests for the evaluation metrics (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import metrics


def test_exact_match_normalization():
    assert metrics.exact_match("The answer is 130 mm Hg.", "answer is 130 mm hg") == 1.0
    assert metrics.exact_match("120", "130") == 0.0


def test_token_f1():
    assert metrics.token_f1("blood pressure 130", "blood pressure 130") == 1.0
    assert metrics.token_f1("", "something") == 0.0
    partial = metrics.token_f1("screen adults aged 45 to 75", "screen adults 45 75 years")
    assert 0.0 < partial < 1.0


def test_fuzzy_match():
    assert metrics.fuzzy_match("colonoscopy every 10 years", "colonoscopy every 10 years") == 1.0
    assert metrics.fuzzy_match("colonoscopy", "mammography") < 0.5


def test_list_component_f1_perfect():
    pred = "Risk factors: obesity, hypertension, family history"
    gold = ["obesity", "hypertension", "family history"]
    out = metrics.list_component_f1(pred, gold)
    assert out["f1"] == 1.0


def test_list_component_f1_partial():
    pred = "obesity and smoking"
    gold = ["obesity", "hypertension", "family history"]
    out = metrics.list_component_f1(pred, gold)
    assert 0.0 < out["recall"] < 1.0
    assert out["precision"] <= 1.0


def test_retrieval_metrics():
    assert metrics.retrieval_hit(["0003", "0007"], ["0007"]) == 1.0
    assert metrics.retrieval_hit(["0003"], ["0007"]) == 0.0
    assert metrics.retrieval_recall(["0003", "0007"], ["0007", "0009"]) == 0.5
    assert metrics.page_hit([4, 5, 6], 5) == 1.0
    assert metrics.page_hit([4, 5, 6], 9) == 0.0


def test_split_items():
    items = metrics.split_items("colonoscopy, FIT; sigmoidoscopy and CT colonography")
    assert "colonoscopy" in items
    assert len(items) >= 4
