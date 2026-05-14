"""
App.py — Streamlit Frontend for Review Analyzer
Talks to the FastAPI backend (main.py) via HTTP.
Set API_BASE in .env or as an environment variable.
"""

import os
import time
import io

import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Review Analyzer",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Review Analyzer")
st.caption("Sentiment · Fake Detection · Duplicate Detection · ABSA · Insights")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**API:** `{API_BASE}`")

    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        st.success(f"API online · {health.get('device','?')} · {health.get('ram_mb','?')} MB")
    except Exception:
        st.error("API offline — is the backend running?")

    st.divider()
    mode = st.radio("Mode", ["Single Review", "Batch CSV", "Duplicate Detection"])


# ── Single Review ─────────────────────────────────────────────────────────────

if mode == "Single Review":
    st.subheader("Analyze a single review")

    col1, col2 = st.columns([3, 1])
    with col1:
        text = st.text_area("Review text", height=120, placeholder="Type or paste a review…")
    with col2:
        st.markdown("&nbsp;")
        run_absa = st.checkbox("Run ABSA", value=False, help="Aspect-Based Sentiment Analysis — slower")
        analyze  = st.button("Analyze ▶", use_container_width=True)

    if analyze and text.strip():
        with st.spinner("Analyzing…"):
            try:
                r = requests.post(f"{API_BASE}/analyze_text", json={"text": text}, timeout=30)
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        c1, c2, c3, c4 = st.columns(4)
        sentiment = result.get("sentiment", "?")
        sent_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(sentiment, "⚪")
        c1.metric("Sentiment", f"{sent_color} {sentiment}")
        c2.metric("Confidence", f"{result.get('confidence', 0)*100:.1f}%")
        fake = result.get("fake_review", "?")
        c3.metric("Fake Review", "🚩 Fake" if fake == "Fake" else "✅ Real")
        c4.metric("Product", result.get("product", "?"))

        if result.get("fake_reasons"):
            with st.expander("Fake detection breakdown"):
                for rule in result["fake_reasons"]:
                    triggered = rule.get("triggered", False)
                    icon = "🔴" if triggered else "⚪"
                    matched = f" — matched: `{rule['matched']}`" if rule.get("matched") else ""
                    st.markdown(f"{icon} **{rule['rule']}** (weight {rule['weight']}){matched}")
                    st.caption(rule.get("description", ""))

        if run_absa:
            with st.spinner("Running ABSA…"):
                try:
                    ar = requests.post(f"{API_BASE}/absa", json={"text": text}, timeout=60)
                    ar.raise_for_status()
                    aspects = ar.json().get("aspects", [])
                except Exception as e:
                    st.warning(f"ABSA failed: {e}")
                    aspects = []

            if aspects:
                st.subheader("Aspect-Based Sentiment")
                for asp in aspects:
                    sent  = asp["sentiment"]
                    icon  = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}.get(sent, "⚪")
                    conf  = asp["confidence"]
                    snip  = asp.get("snippet") or ""
                    st.markdown(
                        f"{icon} **{asp['aspect']}** — {sent} ({conf*100:.0f}%)"
                        + (f"\n> _{snip}_" if snip else "")
                    )


# ── Batch CSV ─────────────────────────────────────────────────────────────────

elif mode == "Batch CSV":
    st.subheader("Analyze a CSV of reviews")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            preview_df = pd.read_csv(uploaded, nrows=5)
            st.dataframe(preview_df, use_container_width=True)
            columns = list(preview_df.columns)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        col = st.selectbox("Review column", columns)

        if st.button("Start Analysis ▶", use_container_width=True):
            uploaded.seek(0)
            with st.spinner("Uploading…"):
                try:
                    r = requests.post(
                        f"{API_BASE}/upload_csv",
                        files={"file": (uploaded.name, uploaded, "text/csv")},
                        data={"column": col},
                        timeout=30,
                    )
                    r.raise_for_status()
                    info = r.json()
                    st.success(f"Processing {info['total_reviews']:,} reviews…")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.stop()

            progress_bar = st.progress(0)
            status_text  = st.empty()

            while True:
                try:
                    prog = requests.get(f"{API_BASE}/progress", timeout=5).json()
                except Exception:
                    time.sleep(2)
                    continue

                pct = prog.get("percent", 0) / 100
                progress_bar.progress(min(pct, 1.0))
                status_text.caption(
                    f"{prog['processed']:,} / {prog['total']:,} reviews · "
                    f"{prog['speed']} rev/s · ETA {prog['eta']}s"
                )

                if not prog.get("running") and prog["processed"] >= prog["total"]:
                    break
                time.sleep(1)

            status_text.empty()
            progress_bar.progress(1.0)
            st.success("Done!")

        # ── Results ───────────────────────────────────────────────────────────
        if st.button("Load Results", use_container_width=True):
            try:
                res = requests.get(f"{API_BASE}/results", timeout=10).json()
                results = res.get("results", [])
            except Exception as e:
                st.error(f"Could not fetch results: {e}")
                st.stop()

            if not results:
                st.info("No results yet — run an analysis first.")
                st.stop()

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                sent_counts = df["sentiment"].value_counts().reset_index()
                sent_counts.columns = ["sentiment", "count"]
                fig = px.pie(
                    sent_counts, names="sentiment", values="count",
                    title="Sentiment distribution",
                    color="sentiment",
                    color_discrete_map={"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"},
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                prod_counts = df["product"].value_counts().head(8).reset_index()
                prod_counts.columns = ["product", "count"]
                fig2 = px.bar(prod_counts, x="count", y="product", orientation="h", title="Top product categories")
                st.plotly_chart(fig2, use_container_width=True)

            fake_pct = (df["fake_review"] == "Fake").mean() * 100
            st.metric("Fake review rate", f"{fake_pct:.1f}%")

            # ── Insights ──────────────────────────────────────────────────────
            if st.button("Generate AI Insights ✨"):
                with st.spinner("Generating insights…"):
                    try:
                        ins = requests.get(f"{API_BASE}/insights", timeout=45).json()
                        st.info(ins.get("summary", "No summary returned."))
                    except Exception as e:
                        st.warning(f"Insights failed: {e}")

            # ── Download ──────────────────────────────────────────────────────
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download results CSV",
                data=csv_bytes,
                file_name="review_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── Duplicate Detection ───────────────────────────────────────────────────────

elif mode == "Duplicate Detection":
    st.subheader("Duplicate Detection")
    st.caption("Runs on the last batch result. Upload and analyze a CSV first.")

    c1, c2 = st.columns(2)
    near_threshold = c1.slider("Near-duplicate threshold (Jaccard)", 0.5, 1.0, 0.80, 0.05)
    sem_threshold  = c2.slider("Semantic threshold (TF-IDF cosine)", 0.5, 1.0, 0.85, 0.05)

    if st.button("Run Duplicate Detection ▶", use_container_width=True):
        with st.spinner("Detecting duplicates…"):
            try:
                r = requests.get(
                    f"{API_BASE}/duplicates",
                    params={"near_threshold": near_threshold, "sem_threshold": sem_threshold},
                    timeout=120,
                )
                r.raise_for_status()
                report = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total reviews",   report["total"])
        m2.metric("Originals",       report["originals"])
        m3.metric("Duplicates",      report["duplicates"])
        m4.metric("Dedup rate",      f"{report['dedup_rate_pct']}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("Exact",    report["exact_count"])
        c2.metric("Near",     report["near_count"])
        c3.metric("Semantic", report["semantic_count"])

        if report["results"]:
            dup_df = pd.DataFrame(report["results"])
            st.dataframe(dup_df, use_container_width=True)

            csv_bytes = dup_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download duplicates CSV",
                data=csv_bytes,
                file_name="duplicates.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.success("No duplicates found!")