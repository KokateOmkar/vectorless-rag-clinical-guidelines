"""Render PDF pages to PNG bytes for Gemini vision grounding (PyMuPDF)."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

import config


def render_pages(slug: str, page_indices: list[int], *, dpi: int | None = None) -> list[bytes]:
    """Render the given 1-based page numbers of a doc's PDF to PNG bytes.

    Unknown/out-of-range pages are skipped. Pages are de-duplicated and ordered.
    """
    pdf_path = config.PDF_DIR / f"{slug}.pdf"
    if not pdf_path.exists():
        return []
    dpi = dpi or config.VISION_PAGE_DPI
    wanted = sorted({p for p in page_indices if isinstance(p, int) and p >= 1})

    images: list[bytes] = []
    with fitz.open(str(pdf_path)) as doc:
        for p in wanted:
            idx = p - 1  # PageIndex page_index is 1-based; PyMuPDF is 0-based
            if 0 <= idx < doc.page_count:
                pix = doc.load_page(idx).get_pixmap(dpi=dpi)
                images.append(pix.tobytes("png"))
    return images
