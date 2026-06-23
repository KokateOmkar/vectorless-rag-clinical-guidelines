"""Index the USPSTF PDFs with PageIndex and cache the resulting trees locally.

Idempotent: skips any document whose tree JSON already exists. Enforces the
free-tier page cap BEFORE spending any credits.
"""
from __future__ import annotations

import datetime as dt

from pypdf import PdfReader

import config
from src import tree_utils
from src.indexing import pageindex_client


def count_pages(slug: str) -> int:
    pdf = config.PDF_DIR / f"{slug}.pdf"
    return len(PdfReader(str(pdf)).pages)


def page_report() -> dict[str, int]:
    """Return {slug: pages} for all configured documents."""
    return {slug: count_pages(slug) for slug in config.DOCUMENTS}


def _pages_needing_index(manifest: dict) -> int:
    """Pages we'd actually submit (skip docs already indexed)."""
    pages = page_report()
    return sum(p for slug, p in pages.items() if slug not in manifest)


def _fetch_and_save_tree(slug: str, doc_id: str, pages: int, manifest: dict) -> dict:
    """Fetch tree + per-page OCR text, attach text to nodes, save, update manifest.

    GET-only — costs no PageIndex credits, so it is safe to re-run on an existing doc_id.
    """
    tree = pageindex_client.get_tree(doc_id, summary=True).get("result") or []
    ocr_pages = pageindex_client.get_ocr_pages(doc_id)
    tree_utils.attach_page_text(tree, ocr_pages)
    tree_utils.save_tree(slug, tree)

    grounded = sum(1 for n, _ in tree_utils.iter_nodes(tree) if tree_utils.node_text(n))
    manifest[slug] = {
        "doc_id": doc_id,
        "title": config.DOCUMENTS[slug],
        "pages": pages,
        "n_nodes": tree_utils.count_nodes(tree),
        "indexed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    tree_utils.save_manifest(manifest)
    print(f"  saved tree ({manifest[slug]['n_nodes']} nodes, {grounded} with text) "
          f"-> {tree_utils.tree_path(slug).name}")
    return manifest


def enrich_existing(slug: str) -> dict:
    """Re-fetch tree + OCR text for an already-indexed doc (no submit, no credits)."""
    if slug not in config.DOCUMENTS:
        raise ValueError(f"Unknown slug '{slug}'.")
    manifest = tree_utils.load_manifest()
    if slug not in manifest:
        raise RuntimeError(f"{slug} not in manifest — index it first.")
    doc_id = manifest[slug]["doc_id"]
    print(f"[enrich] {slug} ({doc_id}) — fetching tree + OCR text (no credits) ...")
    return _fetch_and_save_tree(slug, doc_id, manifest[slug]["pages"], manifest)


def index_one(slug: str, *, force: bool = False) -> dict:
    """Index a single document by slug; updates the shared manifest. Idempotent."""
    if slug not in config.DOCUMENTS:
        raise ValueError(f"Unknown slug '{slug}'. Known: {list(config.DOCUMENTS)}")
    manifest = tree_utils.load_manifest()
    pages = count_pages(slug)
    print(f"[{slug}] {pages} pages")

    if not force and slug in manifest and tree_utils.tree_path(slug).exists():
        print(f"[skip] {slug} already indexed ({manifest[slug]['doc_id']})")
        return manifest

    print(f"[submit] {slug} ...")
    doc_id = pageindex_client.submit(config.PDF_DIR / f"{slug}.pdf")
    print(f"  doc_id={doc_id}; waiting for tree ...")
    pageindex_client.wait_until_ready(doc_id)
    return _fetch_and_save_tree(slug, doc_id, pages, manifest)


def index_all(*, force: bool = False) -> dict:
    """Submit each PDF, poll, save its tree, and update the manifest."""
    manifest = {} if force else tree_utils.load_manifest()
    pages = page_report()
    total = sum(pages.values())

    print("Page counts:")
    for slug, p in pages.items():
        print(f"  {slug:38s} {p:>3d}")
    print(f"  {'TOTAL':38s} {total:>3d}  (free-tier cap: {config.PAGEINDEX_FREE_PAGE_CAP})")

    to_index = _pages_needing_index(manifest) if not force else total
    if to_index > config.PAGEINDEX_FREE_PAGE_CAP:
        raise RuntimeError(
            f"Would submit {to_index} pages, exceeding the free-tier cap of "
            f"{config.PAGEINDEX_FREE_PAGE_CAP}. Aborting to avoid surprise credit spend."
        )

    for slug in config.DOCUMENTS:
        tree_file = tree_utils.tree_path(slug)
        if not force and slug in manifest and tree_file.exists():
            print(f"[skip] {slug} already indexed ({manifest[slug]['doc_id']})")
            continue

        print(f"[submit] {slug} ({pages[slug]} pages) ...")
        doc_id = pageindex_client.submit(config.PDF_DIR / f"{slug}.pdf")
        print(f"  doc_id={doc_id}; waiting for tree ...")
        pageindex_client.wait_until_ready(doc_id)
        _fetch_and_save_tree(slug, doc_id, pages[slug], manifest)

    print("Indexing complete.")
    return manifest


if __name__ == "__main__":
    index_all()
