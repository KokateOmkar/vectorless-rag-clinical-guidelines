"""Streamlit demo for the PageIndex vectorless-RAG project.

Tab 1 (Explore): ask a free-text question against one of the 5 indexed USPSTF guidelines
                 (or upload your own PDF -> live PageIndex indexing), and see the answer,
                 the retrieved tree node(s)/pages, and a confidence indicator.
Tab 2 (Benchmark): show the precomputed evaluation results (no API calls).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import tree_utils  # noqa: E402

st.set_page_config(page_title="PageIndex Vectorless RAG", layout="wide")
st.title("🌲 PageIndex — Vectorless RAG on USPSTF Clinical Guidelines")
st.caption("Reasoning-based tree-search retrieval (no embeddings) + Gemini, evaluated on 5 USPSTF screening guidelines.")

explore_tab, bench_tab = st.tabs(["🔎 Explore", "📊 Benchmark results"])


# ---------------------------------------------------------------------------
# Explore tab
# ---------------------------------------------------------------------------
with explore_tab:
    manifest = tree_utils.load_manifest()
    indexed = list(manifest.keys())

    source = st.radio("Document source", ["Pick an indexed guideline", "Upload a new PDF"], horizontal=True)

    slug = None
    if source == "Pick an indexed guideline":
        if not indexed:
            st.warning("No documents indexed yet. Run `python -m src.cli index` first.")
        else:
            label_to_slug = {config.DOCUMENTS.get(s, s): s for s in indexed}
            label = st.selectbox("Guideline", list(label_to_slug.keys()))
            slug = label_to_slug[label]
    else:
        st.info("Uploading indexes the PDF via PageIndex and **spends ~1 credit per page** "
                "of your free-tier quota.")
        uploaded = st.file_uploader("PDF", type=["pdf"])
        if uploaded is not None and st.button("Index this PDF"):
            from src.indexing import pageindex_client
            tmp = config.CACHE_DIR / uploaded.name
            tmp.write_bytes(uploaded.getvalue())
            with st.spinner("Indexing with PageIndex (OCR + tree build)…"):
                doc_id = pageindex_client.submit(tmp)
                pageindex_client.wait_until_ready(doc_id)
                tree = pageindex_client.get_tree(doc_id, summary=True).get("result", [])
                slug = "upload_" + uploaded.name.rsplit(".", 1)[0].replace(" ", "_")
                tree_utils.save_tree(slug, tree)
            st.success(f"Indexed as '{slug}' ({tree_utils.count_nodes(tree)} nodes).")
            st.session_state["uploaded_slug"] = slug
        slug = st.session_state.get("uploaded_slug", slug)

    question = st.text_input("Ask a question", placeholder="At what age does screening start?")
    k = st.slider("Nodes to retrieve (k)", 1, 6, config.RETRIEVAL_TOP_K)

    if st.button("Answer", type="primary") and slug and question.strip():
        from src.generation.answer import answer_question
        with st.spinner("Tree-search retrieval + generation…"):
            t0 = time.time()
            res = answer_question(slug, question, k=k)
            dt = time.time() - t0

        st.subheader("Answer")
        st.write(res.answer)

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence (top relevance)", f"{res.confidence:.2f}")
        c2.metric("Grounding", res.grounding_mode)
        c3.metric("Latency", f"{dt:.1f}s")

        st.subheader("Retrieved tree nodes")
        for r in res.retrieved:
            with st.expander(f"[{r.node_id}] {r.title} — page {r.page_index} (relevance {r.relevance:.2f})"):
                st.caption(r.reason)
                st.write((r.text or "")[:1500] + ("…" if len(r.text or "") > 1500 else ""))


# ---------------------------------------------------------------------------
# Benchmark tab
# ---------------------------------------------------------------------------
with bench_tab:
    st.subheader("Precomputed evaluation — 5 USPSTF guidelines")
    overall = config.RESULTS_DIR / "metrics_overall.json"
    if overall.exists():
        import json
        st.json(json.loads(overall.read_text(encoding="utf-8")))

    doc_csv = config.RESULTS_DIR / "metrics_by_document.csv"
    type_csv = config.RESULTS_DIR / "metrics_by_question_type.csv"
    if doc_csv.exists():
        st.markdown("**Per document**")
        st.dataframe(pd.read_csv(doc_csv, index_col=0))
    if type_csv.exists():
        st.markdown("**Per question type**")
        st.dataframe(pd.read_csv(type_csv, index_col=0))

    for fig in ["metrics_by_document.png", "metrics_by_question_type.png"]:
        p = config.FIGURES_DIR / fig
        if p.exists():
            st.image(str(p))

    if not doc_csv.exists():
        st.info("No results yet. Run `python -m src.cli eval` then `python -m src.cli report`.")
