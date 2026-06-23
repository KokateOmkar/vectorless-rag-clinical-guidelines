"""Render the results tables into charts saved under results/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config

PLOT_METRICS = ["token_f1", "fuzzy_match", "retrieval_hit", "final_correct"]


def _bar(df: pd.DataFrame, index_label: str, title: str, outfile: str) -> None:
    cols = [c for c in PLOT_METRICS if c in df.columns]
    if df.empty or not cols:
        print(f"  [skip] {outfile}: no data")
        return
    ax = df[cols].plot(kind="bar", figsize=(10, 5))
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.set_xlabel(index_label)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    path = config.FIGURES_DIR / outfile
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  wrote {path}")


def make_report() -> None:
    doc_csv = config.RESULTS_DIR / "metrics_by_document.csv"
    type_csv = config.RESULTS_DIR / "metrics_by_question_type.csv"
    if doc_csv.exists():
        _bar(pd.read_csv(doc_csv, index_col=0), "document",
             "PageIndex RAG — metrics per document", "metrics_by_document.png")
    if type_csv.exists():
        _bar(pd.read_csv(type_csv, index_col=0), "question_type",
             "PageIndex RAG — metrics per question type", "metrics_by_question_type.png")
    print("Report figures written to", config.FIGURES_DIR)


if __name__ == "__main__":
    make_report()
