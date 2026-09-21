"""
app/main.py
------------
Streamlit entry point. Routes between the three UI modes described in the
README: Chat UI (intent capture + final Q&A), Form UI (Bronze/Silver/Gold
STTM review), and the Dashboard (Gold-layer charts).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import streamlit as st

from agents import bronze_agent, gold_agent, orchestrator, reporter, silver_agent
from core.hitl import read_audit_log, render_sttm_review
from core.llm import LLMClient
from core.pipeline import PipelineState, ReviewDecision, Stage
from core.sttm import sttm_to_markdown

st.set_page_config(page_title="E-commerce Returns Analyzer", layout="wide", page_icon="📦")

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample")


@st.cache_resource
def get_llm() -> LLMClient:
    return LLMClient()


def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "bronze_rows" not in st.session_state:
        st.session_state.bronze_rows = None
    if "silver_rows" not in st.session_state:
        st.session_state.silver_rows = None
    if "gold_rows" not in st.session_state:
        st.session_state.gold_rows = None


def sidebar_status(state: PipelineState | None):
    with st.sidebar:
        st.header("Pipeline Status")
        llm = get_llm()
        st.caption(f"LLM mode: {'🟢 live (gpt-4o via GitHub Models)' if llm.is_live else '🟡 offline fallback (set GITHUB_TOKEN)'}")
        if state is None:
            st.info("No active run yet. State your intent in the chat to begin.")
            return
        st.caption(f"Run ID: `{state.run_id}`")
        stages = [Stage.BRONZE, Stage.SILVER, Stage.GOLD, Stage.INSIGHTS]
        for s in stages:
            review = {
                Stage.BRONZE: state.bronze_review,
                Stage.SILVER: state.silver_review,
                Stage.GOLD: state.gold_review,
            }.get(s)
            if s == Stage.INSIGHTS:
                icon = "✅" if state.stage == Stage.INSIGHTS else "⏳"
            else:
                icon = {
                    ReviewDecision.APPROVED: "✅",
                    ReviewDecision.APPROVED_WITH_NOTES: "✅",
                    ReviewDecision.REJECTED: "🔁",
                    ReviewDecision.PENDING: "⏳",
                }[review.decision]
            st.write(f"{icon} {s.value.capitalize()}")

        if st.button("↩️ Start a new analysis"):
            for key in ("pipeline", "chat_history", "bronze_rows", "silver_rows", "gold_rows"):
                st.session_state[key] = None if key != "chat_history" else []
            st.rerun()

        with st.expander("Audit log (this run)"):
            for event in read_audit_log(state.run_id):
                st.json(event, expanded=False)


# ----------------------------------------------------------------------
# Phase 1 — Chat UI for intent capture
# ----------------------------------------------------------------------
def render_intent_chat():
    st.title("📦 E-commerce Returns Analyzer")
    st.caption(
        "State your business intent in plain language. The Supervisor Agent will parse "
        "it and drive the Bronze → Silver → Gold pipeline, with a human review gate at "
        "every layer."
    )

    with st.expander("💡 Example prompts"):
        st.markdown(
            "- *Find which product categories have the highest return rates and estimate "
            "how much revenue each category is losing per month. Focus on the last 6 months.*\n"
            "- *Show me return rate trends by category over the last quarter.*\n"
            "- *Which categories have the worst revenue erosion from returns?*"
        )

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    prompt = st.chat_input("Describe what you want to understand about returns...")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        llm = get_llm()
        params = orchestrator.parse_intent(llm, prompt)
        state = orchestrator.initialize_pipeline(prompt, params)
        st.session_state.pipeline = state

        summary = params.get("summary", "Intent parsed.")
        ack = (
            f"Got it. **{summary}**\n\n"
            f"- Categories: {', '.join(params.get('categories', ['all']))}\n"
            f"- KPIs: {', '.join(params.get('kpis', []))}\n"
            f"- Time window: {params.get('time_window_days', 180)} days\n\n"
            "Next, upload your raw CSV files so the Bronze STTM Agent can propose a schema mapping."
        )
        st.session_state.chat_history.append(("assistant", ack))
        st.rerun()


# ----------------------------------------------------------------------
# Phase 2 — Bronze Review (Form UI)
# ----------------------------------------------------------------------
def render_bronze(state: PipelineState):
    st.title("🥉 Bronze Layer — Raw Ingestion")
    st.caption("Upload your CSV exports (orders, returns, products) or use the bundled sample data.")

    use_sample = st.checkbox("Use bundled sample data (data/sample/*.csv)", value=True)

    file_paths: dict[str, str] = {}
    if use_sample:
        for name in ("orders_sample", "returns_sample", "products_sample"):
            path = os.path.join(SAMPLE_DIR, f"{name}.csv")
            if os.path.exists(path):
                file_paths[name] = path
        if not file_paths:
            st.error("Sample data not found. Run `python data/sample/generate_sample_data.py` first.")
            return
        st.success(f"Loaded {len(file_paths)} sample files.")
    else:
        uploads = st.file_uploader(
            "Upload CSV files", type="csv", accept_multiple_files=True
        )
        if uploads:
            upload_dir = os.path.join(SAMPLE_DIR, "..", "_uploads", state.run_id)
            os.makedirs(upload_dir, exist_ok=True)
            for f in uploads:
                name = os.path.splitext(f.name)[0]
                path = os.path.join(upload_dir, f"{name}.csv")
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                file_paths[name] = path

    if not file_paths:
        st.info("Waiting for files...")
        return

    state.uploaded_files = file_paths

    if st.session_state.bronze_rows is None or st.button("🔄 Re-analyze schema"):
        with st.spinner("Bronze STTM Agent analyzing schema..."):
            llm = get_llm()
            profile, rows, used_llm = bronze_agent.profile_and_propose(llm, file_paths)
            state.profile = profile
            st.session_state.bronze_rows = rows
            st.session_state.bronze_used_llm = used_llm

    if st.session_state.get("bronze_used_llm") is False:
        st.caption("⚠️ Generated via rule-based fallback (LLM offline or unavailable).")

    with st.expander("📊 Raw file profile (schema, nulls, cardinality)"):
        for table, prof in state.profile.items():
            st.write(f"**{table}** — {prof['row_count']} rows, {prof['column_count']} columns")
            st.dataframe(pd.DataFrame(prof["columns"]), use_container_width=True, hide_index=True)

    def on_reject(feedback: str):
        llm = get_llm()
        rows, used_llm = bronze_agent.generate_bronze_sttm(llm, state.profile, feedback=feedback)
        st.session_state.bronze_rows = rows
        st.session_state.bronze_used_llm = used_llm

    decision = render_sttm_review(state, "bronze", st.session_state.bronze_rows, on_reject)

    if decision in ("approved", "approved_with_notes"):
        with st.spinner("Materializing Bronze layer to data/bronze/ (Parquet)..."):
            state.bronze_path = bronze_agent.materialize_bronze(file_paths, state.run_id)
        state.stage = Stage.SILVER
        st.rerun()
    elif decision == "rejected":
        st.rerun()


# ----------------------------------------------------------------------
# Phase 3 — Silver Review (Form UI)
# ----------------------------------------------------------------------
def render_silver(state: PipelineState):
    st.title("🥈 Silver Layer — Clean & Normalize")
    st.caption("Cleaning rules: null handling, deduplication, category normalization, derived columns.")

    bronze_tables = silver_agent.load_bronze(state.bronze_path)

    if st.session_state.silver_rows is None:
        with st.spinner("Silver STTM Agent proposing cleaning rules..."):
            llm = get_llm()
            rows, used_llm = silver_agent.generate_silver_sttm(llm, bronze_tables)
            st.session_state.silver_rows = rows
            st.session_state.silver_used_llm = used_llm

    if st.session_state.get("silver_used_llm") is False:
        st.caption("⚠️ Generated via rule-based fallback (LLM offline or unavailable).")

    def on_reject(feedback: str):
        llm = get_llm()
        rows, used_llm = silver_agent.generate_silver_sttm(llm, bronze_tables, feedback=feedback)
        st.session_state.silver_rows = rows
        st.session_state.silver_used_llm = used_llm

    decision = render_sttm_review(state, "silver", st.session_state.silver_rows, on_reject)

    if decision in ("approved", "approved_with_notes"):
        with st.spinner("Materializing Silver layer to data/silver/ (Parquet)..."):
            state.silver_path = silver_agent.materialize_silver(state.bronze_path, state.run_id)
        state.stage = Stage.GOLD
        st.rerun()
    elif decision == "rejected":
        st.rerun()


# ----------------------------------------------------------------------
# Phase 4 — Gold Review (Form UI)
# ----------------------------------------------------------------------
def render_gold(state: PipelineState):
    st.title("🥇 Gold Layer — Analytical Aggregates")
    st.caption("This is the most consequential review — it defines how KPIs are calculated.")

    if st.session_state.gold_rows is None:
        with st.spinner("Gold STTM Agent proposing KPI definitions..."):
            llm = get_llm()
            rows, used_llm = gold_agent.generate_gold_sttm(llm, state.intent_params)
            st.session_state.gold_rows = rows
            st.session_state.gold_used_llm = used_llm

    if st.session_state.get("gold_used_llm") is False:
        st.caption("⚠️ Generated via rule-based fallback (LLM offline or unavailable).")

    def on_reject(feedback: str):
        llm = get_llm()
        rows, used_llm = gold_agent.generate_gold_sttm(llm, state.intent_params, feedback=feedback)
        st.session_state.gold_rows = rows
        st.session_state.gold_used_llm = used_llm

    decision = render_sttm_review(state, "gold", st.session_state.gold_rows, on_reject)

    if decision in ("approved", "approved_with_notes"):
        with st.spinner("Materializing Gold layer to data/gold/ (Parquet)..."):
            window = state.intent_params.get("time_window_days", 180)
            state.gold_path = gold_agent.materialize_gold(state.silver_path, state.run_id, window)
        state.stage = Stage.INSIGHTS
        st.rerun()
    elif decision == "rejected":
        st.rerun()


# ----------------------------------------------------------------------
# Phase 5 — Explore Results (Chat UI + Dashboard)
# ----------------------------------------------------------------------
def render_insights(state: PipelineState):
    st.title("📈 Insights & Dashboard")
    tables = reporter.load_gold(state.gold_path)
    kpis = tables.get("category_kpis")

    if kpis is None or kpis.empty:
        st.warning("No Gold data available yet.")
        return

    callouts = reporter.top_callouts(tables)
    c1, c2, c3 = st.columns(3)
    c1.metric("Worst return-rate category", callouts.get("worst_return_rate_category", "—"),
              f"{callouts.get('worst_return_rate', 0):.1%}")
    c2.metric("Biggest revenue loss", callouts.get("biggest_revenue_loss_category", "—"),
              f"${callouts.get('biggest_revenue_loss', 0):,.0f}")
    c3.metric("Categories analyzed", len(kpis))

    tab1, tab2, tab3, tab4 = st.tabs(["Return Rate by Category", "Revenue Erosion", "Trends (30/60/90d)", "Return Reasons"])

    with tab1:
        fig = px.bar(
            kpis.sort_values("return_rate", ascending=False),
            x="category", y="return_rate", title="Return Rate by Category",
            labels={"return_rate": "Return Rate", "category": "Category"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.bar(
            kpis.sort_values("revenue_erosion", ascending=False),
            x="category", y="revenue_erosion", title="Revenue Lost to Returns by Category",
            labels={"revenue_erosion": "Revenue Erosion ($)", "category": "Category"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        trends = tables.get("return_rate_trends")
        if trends is not None and not trends.empty:
            fig = px.line(
                trends, x="window_days", y="return_rate", color="category", markers=True,
                title="Return Rate Trend — 30/60/90-Day Windows",
                labels={"window_days": "Trailing Window (days)", "return_rate": "Return Rate"},
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trend data available.")

    with tab4:
        reasons = tables.get("return_reason_breakdown")
        if reasons is not None and not reasons.empty:
            fig = px.bar(
                reasons, x="category", y="count", color="return_reason", barmode="stack",
                title="Return Reason Breakdown by Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No return-reason data available.")

    st.divider()
    st.subheader("💬 Ask a follow-up question")
    for role, msg in st.session_state.chat_history:
        if role in ("user_q", "assistant_a"):
            with st.chat_message("user" if role == "user_q" else "assistant"):
                st.write(msg)

    question = st.chat_input("e.g. Which return reasons are most common in Electronics?")
    if question:
        st.session_state.chat_history.append(("user_q", question))
        llm = get_llm()
        answer = reporter.answer_question(llm, question, tables)
        st.session_state.chat_history.append(("assistant_a", answer))
        st.rerun()


def main():
    init_session()
    state: PipelineState | None = st.session_state.pipeline
    sidebar_status(state)

    if state is None:
        render_intent_chat()
        return

    if state.stage == Stage.BRONZE:
        render_bronze(state)
    elif state.stage == Stage.SILVER:
        render_silver(state)
    elif state.stage == Stage.GOLD:
        render_gold(state)
    elif state.stage == Stage.INSIGHTS:
        render_insights(state)


if __name__ == "__main__":
    main()
