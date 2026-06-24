"""Central configuration: paths, API settings, model names, and tunable constants.

Every free-tier-sensitive value is defined here (and overridable via environment
variables) so quotas can be retuned in one place if PageIndex or Gemini change theirs.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (this file's directory).
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
TREE_DIR = DATA_DIR / "pageindex_trees"
CACHE_DIR = DATA_DIR / "cache"
QA_DIR = DATA_DIR / "qa"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INDEX_MANIFEST = DATA_DIR / "index_manifest.json"
QA_DATASET_CSV = QA_DIR / "qa_dataset.csv"

# Create local-only dirs that aren't guaranteed to exist on a fresh clone.
for _d in (TREE_DIR, CACHE_DIR, QA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# The 5 USPSTF documents (slug -> human-readable title)
# ---------------------------------------------------------------------------
DOCUMENTS: dict[str, str] = {
    "colorectal_cancer_screening": "Colorectal Cancer: Screening",
    "breast_cancer_screening": "Breast Cancer: Screening",
    "hypertension_screening": "Hypertension in Adults: Screening",
    "depression_suicide_screening": "Depression and Suicide Risk in Adults: Screening",
    "prediabetes_diabetes_screening": "Prediabetes and Type 2 Diabetes: Screening",
}

# ---------------------------------------------------------------------------
# PageIndex API
# ---------------------------------------------------------------------------
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_BASE_URL = os.getenv("PAGEINDEX_BASE_URL", "https://api.pageindex.ai")
# Free-tier guard: refuse to submit if total pages would exceed this.
PAGEINDEX_FREE_PAGE_CAP = int(os.getenv("PAGEINDEX_FREE_PAGE_CAP", "200"))

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")
# Client-side rate limit (requests/minute). Keep below the free-tier RPM.
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "8"))

# ---------------------------------------------------------------------------
# Retrieval / evaluation knobs
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))

# Grounding: how Gemini receives the retrieved page(s) when answering.
#   "text"   -> PageIndex per-page OCR markdown attached to nodes
#   "vision" -> render gold PDF page(s) and send as a multimodal call
#   "hybrid" -> text by default, vision for layout-sensitive question types below
GROUNDING_MODE = os.getenv("GROUNDING_MODE", "hybrid")
VISION_QUESTION_TYPES = set(
    os.getenv("VISION_QUESTION_TYPES", "numeric_threshold,age_band_lookup,list_based").split(",")
)
# DPI for rendering PDF pages to images in vision mode.
VISION_PAGE_DPI = int(os.getenv("VISION_PAGE_DPI", "150"))
# Judge trigger (no embeddings): an answer that token-F1 doesn't clearly mark correct
# (< JUDGE_F1_THRESHOLD) but that still resembles the reference lexically
# (fuzzy >= JUDGE_FUZZY_TRIGGER) is "borderline" -> send it to the LLM judge to catch
# correct paraphrases that lexical metrics miss.
JUDGE_F1_THRESHOLD = float(os.getenv("JUDGE_F1_THRESHOLD", "0.6"))
JUDGE_FUZZY_TRIGGER = float(os.getenv("JUDGE_FUZZY_TRIGGER", "0.5"))

QUESTION_TYPES = [
    "information_extraction",
    "numeric_threshold",
    "age_band_lookup",
    "list_based",
]


def require_pageindex_key() -> str:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "PAGEINDEX_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return PAGEINDEX_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return GEMINI_API_KEY
