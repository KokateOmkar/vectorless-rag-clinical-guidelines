"""Answer generation with hybrid grounding.

- Retrieval (tree-search) selects the relevant node(s) and their page(s).
- TEXT grounding: condition Gemini on the nodes' attached OCR-markdown text.
- VISION grounding: send the gold PDF page image(s) to Gemini (better for tables/figures).
- HYBRID (default): text, but switch to vision for layout-sensitive question types
  (numeric_threshold, age_band_lookup, list_based) — see config.VISION_QUESTION_TYPES.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from src.generation import page_render
from src.llm import gemini_client
from src.retrieval import tree_search
from src.retrieval.tree_search import Retrieved

_TEXT_PROMPT = """You are answering a question using ONLY the excerpts from a clinical
guideline below. Be concise and faithful. If the answer is a number, threshold, age
band, or list, state it exactly as written in the source. If the excerpts do not
contain the answer, say "Not found in the provided sections."

QUESTION:
{question}

SOURCE EXCERPTS:
{context}

ANSWER:"""

_VISION_PROMPT = """You are answering a question using ONLY the attached page image(s)
from a clinical guideline. Read tables and figures carefully. Be concise and faithful.
If the answer is a number, threshold, age band, or list, state it exactly as shown. If
the page(s) do not contain the answer, say "Not found in the provided pages."

QUESTION:
{question}

ANSWER:"""


@dataclass
class AnswerResult:
    answer: str
    retrieved: list[Retrieved]
    confidence: float
    grounding_mode: str

    @property
    def used_node_ids(self) -> list[str]:
        return [r.node_id for r in self.retrieved]

    @property
    def used_pages(self) -> list[int]:
        return [r.page_index for r in self.retrieved if r.page_index is not None]


def _choose_mode(question_type: str | None) -> str:
    mode = config.GROUNDING_MODE
    if mode in {"text", "vision"}:
        return mode
    # hybrid: vision only for layout-sensitive types
    if question_type and question_type in config.VISION_QUESTION_TYPES:
        return "vision"
    return "text"


def answer_question(
    slug: str,
    question: str,
    *,
    k: int | None = None,
    question_type: str | None = None,
) -> AnswerResult:
    """Full retrieve -> ground -> generate pipeline for one question."""
    retrieved = tree_search.search(slug, question, k=k)
    if not retrieved:
        return AnswerResult("Not found in the provided sections.", [], 0.0, "none")

    confidence = max((r.relevance for r in retrieved), default=0.0)
    mode = _choose_mode(question_type)

    if mode == "vision":
        pages = [r.page_index for r in retrieved if r.page_index is not None]
        images = page_render.render_pages(slug, pages)
        if images:
            prompt = _VISION_PROMPT.format(question=question.strip())
            text = gemini_client.generate_multimodal(prompt, images).strip()
            return AnswerResult(text, retrieved, confidence, "vision")
        # fall through to text if rendering wasn't possible (e.g. uploaded doc)

    context = tree_search.context_from(retrieved)
    prompt = _TEXT_PROMPT.format(question=question.strip(), context=context)
    text = gemini_client.generate(prompt).strip()
    return AnswerResult(text, retrieved, confidence, "text")
