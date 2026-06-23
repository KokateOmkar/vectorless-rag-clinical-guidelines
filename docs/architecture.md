# Architecture

## Overview

This project evaluates **PageIndex** (vectorless, reasoning-based RAG) on long, structured
USPSTF clinical-guideline PDFs. The pipeline has four stages: **index → retrieve → generate
→ evaluate**.

## 1. Indexing (PageIndex Cloud, one-time)

`src/indexing/build_index.py` uploads each PDF (`POST /doc/`), polls until the tree is ready,
and saves the hierarchical tree JSON to `data/pageindex_trees/<slug>.json` plus a manifest
(`data/index_manifest.json`). A **page-cap guard** refuses to submit if the pages to index
would exceed the free-tier cap (200), preventing surprise credit spend. Indexing is
**idempotent** — already-indexed docs are skipped.

A PageIndex tree node looks like:

```json
{ "title": "Practice Considerations", "node_id": "0012", "page_index": 4,
  "text": "…", "summary": "…", "nodes": [ /* children */ ] }
```

## 2. Retrieval — vectorless tree search

`src/retrieval/tree_search.py` renders the tree as an indented outline
(`[node_id] title — summary`) and asks Gemini to return the top-k `node_id`s most likely to
contain the answer, each with a relevance score and a one-line rationale. Hallucinated ids
(not present in the tree) are dropped. This is the core "vectorless" step — **no embeddings
are used for retrieval**.

## 3. Generation — hybrid grounding

PageIndex tree nodes carry only `title`/`summary`/`page_index` — **no body text**. We therefore
fetch PageIndex's free per-page OCR markdown (`GET ?type=ocr&format=page`, no credits) at index
time and attach it to each node (`tree_utils.attach_page_text`).

`src/generation/answer.py` then answers in one of two grounding modes (config `GROUNDING_MODE`):

- **text** (default): condition Gemini on the retrieved nodes' attached markdown.
- **vision**: render the gold PDF page(s) to PNG (`page_render.py`, PyMuPDF) and send them to
  Gemini as a multimodal call — more robust for tables/figures.
- **hybrid**: text by default, but vision for layout-sensitive question types
  (`numeric_threshold`, `age_band_lookup`, `list_based` — config `VISION_QUESTION_TYPES`).

It returns the answer, cited node ids/pages, a confidence (top retrieval relevance), and the
`grounding_mode` used. All Gemini calls (text and vision) are cached.

## 4. Evaluation

`src/evaluation/` scores each answer against the hand-verified QA benchmark:

- **Retrieval:** Hit@k / Recall@k (node-id match, with a gold-page fallback).
- **Generation:** token-F1, Exact Match, fuzzy match.
- **List questions:** List-Component F1 over matched items.
- **LLM-as-judge:** invoked **only for borderline cases** (token-F1 doesn't clearly mark
  the answer correct, yet fuzzy ratio is high — the signature of a correct paraphrase);
  reasoning is logged to `results/judge_log.jsonl`.

Results aggregate **per document** and **per question type** into CSVs; `make_report.py`
renders bar charts.

## Cross-cutting: the Gemini client

`src/llm/gemini_client.py` (built on the maintained `google-genai` SDK) centralizes:

- a **token-bucket rate limiter** (stay under free-tier RPM),
- **retry/backoff** on 429 / resource-exhausted,
- **on-disk response caching** keyed by `(model, prompt)` so re-runs cost ~0 quota,
- **model fallback** `gemini-2.5-flash → gemini-2.0-flash` when the daily cap is hit.

All tunable numbers live in `config.py`.
