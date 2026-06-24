"""Streamlit demo for the vectorless-RAG-clinical-guidelines project.

Two tabs:
  - Benchmark report : the precomputed evaluation (no API calls) — headline metrics, a
                       pass/fail table, a per-question drill-down, and native charts.
  - Ask a guideline  : pick an indexed USPSTF PDF and ask a question live, with a session
                       history and the precomputed Q&As for that guideline.
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

REPO = "vectorless-rag-clinical-guidelines"
st.set_page_config(page_title=REPO, layout="wide")


# ---------------------------------------------------------------------------
# Cached data loaders (no API calls)
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
def load_csv(name: str) -> pd.DataFrame:
    path = config.RESULTS_DIR / name
    return pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()


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


def ground(g: str) -> str:
    return "—" if g in (None, "none", "") else g


results = load_results()
overall = load_overall()

st.title("Vectorless RAG — Clinical Guidelines")
st.caption(
    "Reasoning-based tree-search retrieval (no embeddings, no vector database) over long "
    "USPSTF screening guidelines, with PageIndex for indexing and Gemini for retrieval and "
    "generation.  ·  github.com/KokateOmkar/" + REPO
)

bench_tab, ask_tab = st.tabs(["Benchmark report", "Ask a guideline"])


# ---------------------------------------------------------------------------
# Benchmark tab
# ---------------------------------------------------------------------------
with bench_tab:
    if results.empty:
        st.info("No results yet. Run `python -m src.cli eval` then `python -m src.cli report`.")
    else:
        n = int(overall.get("n", len(results)))
        n_docs = results["document"].nunique()
        st.caption(
            f"Validated end-to-end on **{n} questions** across **{n_docs} guidelines**, entirely "
            "on free-tier APIs. Each question is retrieved, answered, and graded automatically."
        )

        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Answers correct", f"{overall.get('final_correct', 0) * 100:.0f}%")
            c2.metric("Right section found", f"{overall.get('retrieval_hit', 0) * 100:.0f}%")
            c3.metric("Token-F1", f"{overall.get('token_f1', 0):.2f}")
            c4.metric("Questions graded", n)

        # Filters
        with st.container(border=True):
            st.markdown("**Filters**")
            fc1, fc2, fc3 = st.columns([2, 2, 1])
            doc_opts = {"All guidelines": "All"}
            for s in sorted(results["document"].unique()):
                doc_opts[config.DOCUMENTS.get(s, s)] = s
            pick_doc = doc_opts[fc1.selectbox("Guideline", list(doc_opts), key="f_doc")]
            types = sorted(results["question_type"].unique())
            pick_types = fc2.multiselect("Question type", types, default=types, key="f_types")
            only_misses = fc3.checkbox("Only misses", value=False, key="f_misses")

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
            "Section found": view["retrieval_hit"].astype(bool),
            "Answer correct": view["final_correct"].astype(bool),
            "Token-F1": view["token_f1"].astype(float),
            "Grounding": view["grounding_mode"].map(ground),
        })
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Question": st.column_config.TextColumn(width="large"),
                "Section found": st.column_config.CheckboxColumn(
                    "Section found", help="Did retrieval find the gold section?", disabled=True),
                "Answer correct": st.column_config.CheckboxColumn(
                    "Answer correct", help="Was the final answer judged correct?", disabled=True),
                "Token-F1": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0),
            },
        )

        # Per-question drill-down
        with st.container(border=True):
            st.markdown("**Inspect a question**")
            if len(view):
                labels = {f"{row.question or row.question_id}": row.question_id
                          for row in view.itertuples()}
                pick_col, _ = st.columns([3, 2])
                pick = pick_col.selectbox("Question", list(labels), key="inspect")
                row = view[view["question_id"] == labels[pick]].iloc[0]

                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Question**  ·  `{row['question_type']}`")
                    st.write(row["question"])
                    st.markdown("**Reference answer**")
                    st.info(row["reference"])
                    st.markdown(f"**Gold section**  ·  page {row.get('gold_page', '?')}")
                    st.write(f"`{row.get('gold_node_id', '?')}` — {row.get('gold_section', '?')}")
                with right:
                    ok = row["final_correct"] == 1.0
                    st.markdown("**Model answer**")
                    (st.success if ok else st.error)(row["prediction"])
                    found = "found" if row["retrieval_hit"] == 1.0 else "not found"
                    st.markdown(f"**Retrieved sections**  ·  gold section {found}")
                    for s in section_titles(row["document"], row["retrieved_node_ids"]) or ["(none)"]:
                        st.write("-", s)
                    meta = (f"grounding **{ground(row['grounding_mode'])}**  ·  "
                            f"token-F1 **{float(row['token_f1']):.2f}**  ·  "
                            f"fuzzy **{float(row['fuzzy_match']):.2f}**")
                    if str(row.get("judge_verdict") or "").strip():
                        meta += f"  ·  judge **{row['judge_verdict']}**"
                    st.caption(meta)

        # Native charts (no image files — always render, never stale)
        with st.container(border=True):
            st.markdown("**Performance by category**")
            metric_map = {"Answer correct": "final_correct",
                          "Token-F1": "token_f1",
                          "Right section": "retrieval_hit"}
            choice = st.radio("Metric", list(metric_map), horizontal=True, key="chart_metric")
            mcol = metric_map[choice]
            by_type = load_csv("metrics_by_question_type.csv")
            by_doc = load_csv("metrics_by_document.csv")
            ch1, ch2 = st.columns(2)
            if mcol in by_type:
                ch1.caption("By question type")
                ch1.bar_chart(by_type[mcol], height=260)
            if mcol in by_doc:
                by_doc = by_doc.rename(index=lambda s: config.DOCUMENTS.get(s, s))
                ch2.caption("By guideline")
                ch2.bar_chart(by_doc[mcol], height=260)


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

    with st.container(border=True):
        top, _ = st.columns([3, 2])
        label = top.selectbox("Choose a guideline", list(label_to_slug), key="ask_doc")
        slug = label_to_slug[label]
        question = st.text_input("Ask a question", placeholder="At what age does screening start?", key="ask_q")
        opt, btn = st.columns([3, 1])
        k = opt.slider("Sections to retrieve (k)", 1, 6, config.RETRIEVAL_TOP_K, key="ask_k")
        go = btn.button("Answer", type="primary", key="ask_go", use_container_width=True)
        st.caption("Live answering uses the Gemini free tier (~20 requests/day). If it is "
                   "rate-limited, the precomputed Q&As below and the Benchmark tab still work.")

    if go and question.strip():
        from src.generation.answer import answer_question
        from src.llm.gemini_client import QuotaExhausted, TransientError
        try:
            with st.spinner("Tree-search retrieval and generation…"):
                t0 = time.time()
                res = answer_question(slug, question, k=k)
                dt = time.time() - t0
            st.session_state.setdefault("history", []).insert(0, {
                "Guideline": label, "Question": question, "Answer": res.answer,
                "Grounding": ground(res.grounding_mode), "Confidence": round(res.confidence, 2),
            })
            with st.container(border=True):
                st.markdown("**Answer**")
                st.success(res.answer)
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{res.confidence:.2f}")
                m2.metric("Grounding", ground(res.grounding_mode))
                m3.metric("Latency", f"{dt:.1f}s")
                st.markdown("**Retrieved sections**")
                for r in res.retrieved:
                    with st.expander(f"[{r.node_id}] {r.title} — page {r.page_index} "
                                     f"(relevance {r.relevance:.2f})"):
                        st.caption(r.reason)
                        snippet = (r.text or "")[:1200]
                        st.write(snippet + ("…" if len(r.text or "") > 1200 else ""))
        except (QuotaExhausted, TransientError):
            st.warning("Gemini is rate-limited or busy right now (free tier). Try again later, "
                       "or browse the precomputed results below and in the Benchmark tab.")

    history = st.session_state.get("history", [])
    if history:
        with st.container(border=True):
            st.markdown("**This session**")
            st.dataframe(pd.DataFrame(history), hide_index=True, use_container_width=True)

    if not results.empty:
        doc_rows = results[results["document"] == slug]
        with st.container(border=True):
            if len(doc_rows):
                st.markdown(f"**Precomputed Q&As for {label}** ({len(doc_rows)})")
                for row in doc_rows.itertuples():
                    with st.expander(row.question or row.question_id):
                        body = f"**Answer:** {row.prediction}"
                        (st.success if row.final_correct == 1.0 else st.error)(body)
                        st.markdown(f"**Reference:** {row.reference}")
                        found = "found" if row.retrieval_hit == 1.0 else "not found"
                        st.caption(f"gold section {found}  ·  grounding {ground(row.grounding_mode)}"
                                   f"  ·  token-F1 {float(row.token_f1):.2f}")
            else:
                st.caption(f"No precomputed questions for {label} yet — free-tier daily limits "
                           "capped the validated set to colorectal and breast.")
