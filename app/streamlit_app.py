"""Streamlit demo for the PageIndex vectorless-RAG project.

Two tabs:
  - Benchmark report : the precomputed 15-question evaluation (no API calls) — headline
                       metrics, a pass/fail table, and a per-question drill-down.
  - Ask a guideline  : pick one of the indexed USPSTF PDFs and ask a question live, with a
                       running history and the precomputed Q&As for that guideline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import tree_utils  # noqa: E402

st.set_page_config(page_title="PageIndex Vectorless RAG", page_icon="🌲", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached — no API calls)
# ---------------------------------------------------------------------------
@st.cache_data
def load_overall() -> dict:
    path = config.RESULTS_DIR / "metrics_overall.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def load_results() -> pd.DataFrame:
    """Per-question results joined with the question text and gold section."""
    raw = config.RESULTS_DIR / "raw_results.jsonl"
    if not raw.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if config.QA_DATASET_CSV.exists():
        qa = pd.read_csv(config.QA_DATASET_CSV)[
            ["question_id", "question", "gold_section", "gold_page", "gold_node_id"]
        ]
        df = df.merge(qa, on="question_id", how="left")
    return df


@st.cache_data
def section_titles(slug: str, node_ids_csv: str) -> list[str]:
    """Map retrieved node ids to '[id] Title' for display."""
    try:
        tree = tree_utils.load_tree(slug)
    except FileNotFoundError:
        return []
    out = []
    for nid in [x for x in (node_ids_csv or "").split(",") if x]:
        node = tree_utils.find_node(tree, nid)
        out.append(f"[{nid}] {node.get('title') if node else '?'}")
    return out


def tick(value) -> str:
    return "✅" if float(value or 0) == 1.0 else "❌"


GROUND_BADGE = {"vision": "👁 vision", "text": "📄 text", "none": "—"}

st.title("🌲 PageIndex — Vectorless RAG on USPSTF Clinical Guidelines")
st.caption(
    "Reasoning-based tree-search retrieval (no embeddings, no vector DB) + Gemini, "
    "evaluated on long USPSTF screening guidelines."
)

results = load_results()
overall = load_overall()

bench_tab, ask_tab = st.tabs(["📊 Benchmark report", "🔎 Ask a guideline"])


# ---------------------------------------------------------------------------
# Benchmark tab
# ---------------------------------------------------------------------------
with bench_tab:
    if results.empty:
        st.info("No results yet. Run `python -m src.cli eval` then `python -m src.cli report`.")
    else:
        n = int(overall.get("n", len(results)))
        n_docs = results["document"].nunique()
        st.subheader("How well does tree-search retrieval answer clinical questions?")
        st.caption(
            f"Validated end-to-end on **{n} questions** across **{n_docs} guidelines**, entirely "
            "on free-tier APIs. Each question is retrieved, answered, and graded automatically."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answers correct", f"{overall.get('final_correct', 0) * 100:.0f}%")
        c2.metric("Right section found", f"{overall.get('retrieval_hit', 0) * 100:.0f}%")
        c3.metric("Token-F1", f"{overall.get('token_f1', 0):.2f}")
        c4.metric("Questions", n)

        st.divider()

        # Filters
        f1, f2, f3 = st.columns([2, 2, 1])
        doc_opts = {"All guidelines": "All"}
        for s in sorted(results["document"].unique()):
            doc_opts[config.DOCUMENTS.get(s, s)] = s
        pick_doc = doc_opts[f1.selectbox("Guideline", list(doc_opts), key="f_doc")]
        types = sorted(results["question_type"].unique())
        pick_types = f2.multiselect("Question type", types, default=types, key="f_types")
        only_misses = f3.checkbox("Only misses", value=False, key="f_misses")

        view = results.copy()
        if pick_doc != "All":
            view = view[view["document"] == pick_doc]
        view = view[view["question_type"].isin(pick_types)]
        if only_misses:
            view = view[view["final_correct"] != 1.0]

        correct = int(view["final_correct"].sum())
        hits = int(view["retrieval_hit"].sum())
        st.markdown(
            f"**{correct}/{len(view)}** answers correct &nbsp;·&nbsp; "
            f"**{hits}/{len(view)}** retrieved the right section"
        )

        table = pd.DataFrame({
            "Question": view["question"].fillna(view["question_id"]),
            "Type": view["question_type"],
            "Section": view["retrieval_hit"].map(tick),
            "Answer": view["final_correct"].map(tick),
            "Token-F1": view["token_f1"].astype(float),
            "Grounding": view["grounding_mode"].map(lambda g: GROUND_BADGE.get(g, g)),
        })
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
                "Section": st.column_config.TextColumn("Section ✓", help="Did retrieval find the gold section?"),
                "Answer": st.column_config.TextColumn("Answer ✓", help="Was the final answer judged correct?"),
                "Token-F1": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
            },
        )

        # Per-question drill-down
        st.divider()
        st.markdown("#### Inspect a question")
        if len(view):
            labels = {f"{tick(row.final_correct)}  {row.question or row.question_id}": row.question_id
                      for row in view.itertuples()}
            pick = st.selectbox("Pick one", list(labels), key="inspect")
            row = view[view["question_id"] == labels[pick]].iloc[0]

            left, right = st.columns(2)
            with left:
                st.markdown(f"**Question** &nbsp; `{row['question_type']}`")
                st.write(row["question"])
                st.markdown("**Reference answer**")
                st.info(row["reference"])
                st.markdown(f"**Gold section** (page {row.get('gold_page', '?')})")
                st.write(f"`{row.get('gold_node_id', '?')}` — {row.get('gold_section', '?')}")
            with right:
                ok = row["final_correct"] == 1.0
                st.markdown("**Model answer**")
                (st.success if ok else st.error)(row["prediction"])
                st.markdown(f"**Retrieved sections** — {tick(row['retrieval_hit'])} "
                            f"{'hit' if row['retrieval_hit'] == 1.0 else 'miss'}")
                for s in section_titles(row["document"], row["retrieved_node_ids"]) or ["(none)"]:
                    st.write("•", s)
                meta = (f"grounding: **{GROUND_BADGE.get(row['grounding_mode'], row['grounding_mode'])}** "
                        f"· token-F1 **{float(row['token_f1']):.2f}** · fuzzy **{float(row['fuzzy_match']):.2f}**")
                if str(row.get("judge_verdict") or "").strip():
                    meta += f" · judge: **{row['judge_verdict']}**"
                st.caption(meta)

        # Charts
        st.divider()
        cols = st.columns(2)
        for col, fig in zip(cols, ["metrics_by_document.png", "metrics_by_question_type.png"]):
            p = config.FIGURES_DIR / fig
            if p.exists():
                col.image(str(p), use_column_width=True)


# ---------------------------------------------------------------------------
# Ask tab
# ---------------------------------------------------------------------------
with ask_tab:
    manifest = tree_utils.load_manifest()
    indexed = list(manifest.keys())
    if not indexed:
        st.warning("No documents indexed yet. Run `python -m src.cli index` first.")
        st.stop()

    label_to_slug = {config.DOCUMENTS.get(s, s): s for s in indexed}
    label = st.selectbox("Choose a guideline", list(label_to_slug), key="ask_doc")
    slug = label_to_slug[label]

    question = st.text_input("Ask a question", placeholder="At what age does screening start?", key="ask_q")
    k = st.slider("Sections to retrieve (k)", 1, 6, config.RETRIEVAL_TOP_K, key="ask_k")
    go = st.button("Answer", type="primary", key="ask_go")

    st.caption("Live answering uses the Gemini free tier (~20 requests/day). If it's rate-limited, "
               "the Benchmark tab shows precomputed results.")

    if go and question.strip():
        from src.generation.answer import answer_question
        from src.llm.gemini_client import QuotaExhausted, TransientError
        try:
            with st.spinner("Tree-search retrieval + generation…"):
                t0 = time.time()
                res = answer_question(slug, question, k=k)
                dt = time.time() - t0
            st.session_state.setdefault("history", []).insert(0, {
                "Guideline": label, "Question": question, "Answer": res.answer,
                "Grounding": GROUND_BADGE.get(res.grounding_mode, res.grounding_mode),
                "Confidence": round(res.confidence, 2),
            })

            st.markdown("**Answer**")
            st.success(res.answer)
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{res.confidence:.2f}")
            m2.metric("Grounding", GROUND_BADGE.get(res.grounding_mode, res.grounding_mode))
            m3.metric("Latency", f"{dt:.1f}s")
            st.markdown("**Retrieved sections**")
            for r in res.retrieved:
                with st.expander(f"[{r.node_id}] {r.title} — page {r.page_index} (relevance {r.relevance:.2f})"):
                    st.caption(r.reason)
                    snippet = (r.text or "")[:1200]
                    st.write(snippet + ("…" if len(r.text or "") > 1200 else ""))
        except (QuotaExhausted, TransientError):
            st.warning("Gemini is rate-limited or busy right now (free tier). Try again later, "
                       "or browse the precomputed results below and in the Benchmark tab.")

    # Session history
    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.markdown("#### This session")
        st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)

    # Precomputed Q&As for the selected guideline (always available, no API)
    if not results.empty:
        doc_rows = results[results["document"] == slug]
        if len(doc_rows):
            st.divider()
            st.markdown(f"#### Precomputed Q&As for *{label}* ({len(doc_rows)})")
            for row in doc_rows.itertuples():
                with st.expander(f"{tick(row.final_correct)}  {row.question or row.question_id}"):
                    st.markdown(f"**Model answer:** {row.prediction}")
                    st.markdown(f"**Reference:** {row.reference}")
                    st.caption(f"section retrieved {tick(row.retrieval_hit)} · "
                               f"grounding {GROUND_BADGE.get(row.grounding_mode, row.grounding_mode)} · "
                               f"token-F1 {float(row.token_f1):.2f}")
        else:
            st.divider()
            st.caption(f"No precomputed questions for *{label}* yet — free-tier daily limits capped "
                       "the validated set to colorectal and breast.")
