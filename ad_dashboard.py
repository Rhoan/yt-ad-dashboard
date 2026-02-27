"""
ad_dashboard.py — YouTube Ad Analysis Dashboard (4 tabs, port 8502)

Reads ads_cache.json produced by batch_analyzer.py and visualizes aggregate
patterns across 20 kitchen/home-remodeling ads.

Launch:
    /c/Users/rohan/anaconda3/python.exe -m streamlit run ad_dashboard.py \
        --server.port 8502 --server.headless true
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from batch_analyzer import CACHE_PATH, run_batch

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="YouTube Ad Analysis Dashboard",
    layout="wide",
)

# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(ttl=0)
def load_results() -> pd.DataFrame:
    """Read ads_cache.json → flat DataFrame. Returns empty DF if missing."""
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Normalize emotional_appeal list → pipe-joined string
    if "emotional_appeal" in df.columns:
        df["emotional_appeal_str"] = df["emotional_appeal"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else (str(x) if x else "")
        )
    else:
        df["emotional_appeal_str"] = ""

    # Ensure boolean columns are actual booleans
    for col in ["has_cta", "has_before_after", "has_price_mention"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: bool(x) if x is not None else False)

    # Coerce numerics
    for col in [
        "view_count", "like_count", "comment_count", "duration",
        "ad_length_seconds", "input_tokens", "output_tokens",
        "cost_usd", "elapsed_sec",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _reload():
    st.cache_data.clear()
    st.rerun()


# =============================================================================
# Sidebar
# =============================================================================
df_all = load_results()

with st.sidebar:
    st.title("YouTube Ad Dashboard")
    st.markdown("---")

    # Run buttons — only shown when running locally
    _is_local = not st.context.headers.get("host", "").endswith(".streamlit.app")
    if _is_local:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Run Mock", use_container_width=True):
                with st.spinner("Running mock analysis..."):
                    run_batch(mock=True)
                _reload()
        with c2:
            if st.button("Run Real\n~$2.00", use_container_width=True, type="primary"):
                with st.spinner("Running real analysis (~$2.00)..."):
                    run_batch(mock=False)
                _reload()

        if st.button("Reset Cache", use_container_width=True):
            if CACHE_PATH.exists():
                CACHE_PATH.unlink()
            _reload()
    else:
        st.caption("Analysis runs locally only.")

    st.markdown("---")

    if df_all.empty:
        st.info("No data yet. Click 'Run Mock' to populate.")
        st.stop()

    # Filters (only shown when data is present)
    st.subheader("Filters")

    tone_vals = sorted(df_all["tone"].dropna().unique().tolist())
    sel_tones = st.multiselect("Tone", tone_vals, default=tone_vals)

    theme_vals = sorted(df_all["theme"].dropna().unique().tolist())
    sel_themes = st.multiselect("Theme", theme_vals, default=theme_vals)

    cta_filter = st.radio("Has CTA", ["All", "Yes", "No"], horizontal=True)

    max_views = int(df_all["view_count"].max()) if not df_all["view_count"].isna().all() else 0
    min_views = st.slider("Min Views", 0, max(max_views, 1), 0, step=1_000)

# =============================================================================
# Apply filters
# =============================================================================
df = df_all.copy()

if sel_tones:
    df = df[df["tone"].isin(sel_tones)]
if sel_themes:
    df = df[df["theme"].isin(sel_themes)]
if cta_filter == "Yes":
    df = df[df["has_cta"] == True]
elif cta_filter == "No":
    df = df[df["has_cta"] == False]
df = df[df["view_count"].fillna(0) >= min_views]

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", "Label Analytics", "Video Comparison", "Video Inspector"
])

# =============================================================================
# TAB 1 — Overview
# =============================================================================
with tab1:
    st.header("Overview")

    if df.empty:
        st.warning("No videos match the current filters.")
    else:
        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Videos Analyzed", len(df))
        with k2:
            total_cost = df["cost_usd"].sum()
            st.metric("Total API Cost", f"${total_cost:.4f}")
        with k3:
            avg_len = df["ad_length_seconds"].mean()
            st.metric("Avg Ad Length", f"{avg_len:.0f}s" if not np.isnan(avg_len) else "N/A")
        with k4:
            pct_cta = df["has_cta"].mean() * 100
            st.metric("% With CTA", f"{pct_cta:.0f}%")

        st.markdown("---")

        # Tone bar + Theme pie
        cl, cr = st.columns(2)
        with cl:
            tc = df["tone"].value_counts().reset_index()
            tc.columns = ["tone", "count"]
            fig = px.bar(tc, x="tone", y="count", color="tone",
                         title="Tone Distribution",
                         labels={"tone": "Tone", "count": "Count"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with cr:
            thc = df["theme"].value_counts().reset_index()
            thc.columns = ["theme", "count"]
            fig = px.pie(thc, names="theme", values="count",
                         title="Theme Breakdown", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # Hook type + Target audience
        cl2, cr2 = st.columns(2)
        with cl2:
            hk = df["hook_type"].value_counts().reset_index()
            hk.columns = ["hook_type", "count"]
            fig = px.bar(hk, x="hook_type", y="count", color="hook_type",
                         title="Hook Type Distribution",
                         labels={"hook_type": "Hook Type", "count": "Count"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with cr2:
            aud = df["target_audience"].value_counts().reset_index()
            aud.columns = ["target_audience", "count"]
            fig = px.bar(aud, x="target_audience", y="count", color="target_audience",
                         title="Target Audience Distribution",
                         labels={"target_audience": "Audience", "count": "Count"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — Label Analytics
# =============================================================================
with tab2:
    st.header("Label Analytics")

    if df.empty:
        st.warning("No videos match the current filters.")
    else:
        # Section A — Categorical labels (3-column grid)
        st.subheader("Categorical Labels")
        CAT_LABELS = [
            "tone", "theme", "hook_type", "narrator_type", "pacing",
            "color_palette", "music_mood", "setting", "cta_type",
            "target_audience", "product_reveal_timing",
        ]
        grid_cols = st.columns(3)
        for i, label in enumerate(CAT_LABELS):
            if label not in df.columns:
                continue
            cnt = df[label].value_counts().reset_index()
            cnt.columns = [label, "count"]
            fig = px.bar(cnt, x=label, y="count",
                         title=label.replace("_", " ").title(),
                         labels={label: "", "count": "Count"})
            fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
            grid_cols[i % 3].plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Section B — Boolean labels grouped bar
        st.subheader("Boolean Labels")
        BOOL_LABELS = ["has_cta", "has_before_after", "has_price_mention"]
        bool_rows = []
        for col in BOOL_LABELS:
            if col not in df.columns:
                continue
            yes_n = int(df[col].sum())
            bool_rows.append({
                "label": col.replace("_", " ").title(),
                "Yes": yes_n,
                "No": len(df) - yes_n,
            })
        if bool_rows:
            bdf = pd.DataFrame(bool_rows).melt(
                id_vars="label", var_name="value", value_name="count"
            )
            fig = px.bar(bdf, x="label", y="count", color="value", barmode="group",
                         title="Boolean Label Counts",
                         labels={"label": "", "count": "Count", "value": ""})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Section C — Emotional Appeals
        st.subheader("Emotional Appeals")
        if "emotional_appeal_str" in df.columns:
            appeals = [
                item.strip()
                for val in df["emotional_appeal_str"].dropna()
                for item in val.split(" | ")
                if item.strip()
            ]
            if appeals:
                ap_cnt = pd.Series(appeals).value_counts().reset_index()
                ap_cnt.columns = ["appeal", "count"]
                fig = px.bar(ap_cnt, x="appeal", y="count", color="appeal",
                             title="Emotional Appeal Frequency",
                             labels={"appeal": "Appeal", "count": "Count"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Section D — Ad Length Distribution
        st.subheader("Ad Length Distribution")
        if "ad_length_seconds" in df.columns:
            fig = px.histogram(df, x="ad_length_seconds", nbins=20,
                               title="Ad Length Distribution",
                               labels={"ad_length_seconds": "Length (seconds)"})
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3 — Video Comparison
# =============================================================================
with tab3:
    st.header("Video Comparison")

    if df.empty:
        st.warning("No videos match the current filters.")
    else:
        # Sortable table
        st.subheader("All Videos")
        TABLE_COLS = [
            "title", "uploader", "platform", "view_count", "duration",
            "tone", "theme", "has_cta", "cost_usd", "webpage_url",
        ]
        table_df = df[[c for c in TABLE_COLS if c in df.columns]].copy()
        if "has_cta" in table_df.columns:
            table_df["has_cta"] = table_df["has_cta"].map({True: "Yes", False: "No"})

        col_cfg = {}
        if "webpage_url" in table_df.columns:
            col_cfg["webpage_url"] = st.column_config.LinkColumn("URL")
        if "title" in table_df.columns:
            col_cfg["title"] = st.column_config.TextColumn("Title", width="large")

        st.dataframe(table_df, column_config=col_cfg, use_container_width=True)

        st.markdown("---")

        # Scatter: views vs ad_length  |  views vs has_before_after
        sc_l, sc_r = st.columns(2)
        with sc_l:
            if {"view_count", "ad_length_seconds"}.issubset(df.columns):
                fig = px.scatter(
                    df, x="ad_length_seconds", y="view_count",
                    color="tone" if "tone" in df.columns else None,
                    hover_data=["title"] if "title" in df.columns else None,
                    title="Views vs Ad Length",
                    labels={"ad_length_seconds": "Ad Length (s)", "view_count": "Views"},
                )
                st.plotly_chart(fig, use_container_width=True)

        with sc_r:
            if {"view_count", "has_before_after"}.issubset(df.columns):
                sdf = df.copy()
                rng = np.random.default_rng(42)
                sdf["ba_jitter"] = (
                    sdf["has_before_after"].astype(float)
                    + rng.uniform(-0.1, 0.1, size=len(sdf))
                )
                fig = px.scatter(
                    sdf, x="ba_jitter", y="view_count",
                    color="theme" if "theme" in sdf.columns else None,
                    hover_data=["title"] if "title" in sdf.columns else None,
                    title="Views vs Has Before/After",
                    labels={"ba_jitter": "Has Before/After  (0 = No, 1 = Yes)",
                            "view_count": "Views"},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Cost per video
        if "cost_usd" in df.columns:
            cost_df = df[["title", "cost_usd", "tone"]].copy()
            cost_df = cost_df.sort_values("cost_usd", ascending=True)
            cost_df["short_title"] = cost_df["title"].str[:40]
            fig = px.bar(
                cost_df, x="short_title", y="cost_usd", color="tone",
                title="API Cost per Video",
                labels={"short_title": "Video", "cost_usd": "Cost (USD)"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 4 — Video Inspector
# =============================================================================
with tab4:
    st.header("Video Inspector")

    if df.empty:
        st.warning("No videos match the current filters.")
    else:
        selected_title = st.selectbox("Select a video", df["title"].tolist())
        row = df[df["title"] == selected_title].iloc[0]

        col_l, col_r = st.columns(2)

        with col_l:
            ad_len = row.get("ad_length_seconds")
            st.metric("Ad Length", f"{int(ad_len)}s" if pd.notna(ad_len) else "N/A")

            m1, m2 = st.columns(2)
            with m1:
                views = row.get("view_count")
                st.metric("Views", f"{int(views):,}" if pd.notna(views) else "N/A")
            with m2:
                cost = row.get("cost_usd")
                st.metric("API Cost", f"${float(cost):.4f}" if pd.notna(cost) else "N/A")

            # 16-label table
            LABEL_KEYS = [
                "tone", "emotional_appeal_str", "theme", "product_reveal_timing",
                "hook_type", "narrator_type", "pacing", "color_palette",
                "music_mood", "setting", "has_cta", "cta_type",
                "has_before_after", "has_price_mention", "ad_length_seconds",
                "target_audience",
            ]
            label_rows = []
            for k in LABEL_KEYS:
                if k not in row.index:
                    continue
                val = row[k]
                display_key = (
                    k.replace("_str", "")
                     .replace("_", " ")
                     .title()
                )
                if isinstance(val, bool):
                    display_val = "Yes" if val else "No"
                elif not isinstance(val, (list, dict)) and pd.isna(val):
                    display_val = "N/A"
                else:
                    display_val = str(val)
                label_rows.append({"Label": display_key, "Value": display_val})

            st.table(pd.DataFrame(label_rows))

            with st.expander("Transcript"):
                st.write(
                    "Transcripts are not stored in the cache. "
                    "Re-run analysis to capture them."
                )

        with col_r:
            url = row.get("webpage_url", "")
            if url:
                st.markdown(f"### [Watch on YouTube]({url})")

            st.markdown(f"**Channel:** {row.get('uploader', 'N/A')}")
            st.markdown(f"**Upload Date:** {row.get('upload_date', 'N/A')}")

            m3, m4 = st.columns(2)
            likes = row.get("like_count")
            comments = row.get("comment_count")
            with m3:
                st.metric("Likes", f"{int(likes):,}" if pd.notna(likes) else "N/A")
            with m4:
                st.metric("Comments", f"{int(comments):,}" if pd.notna(comments) else "N/A")

            st.markdown("---")
            st.markdown("**Token Usage**")
            in_tok = row.get("input_tokens")
            out_tok = row.get("output_tokens")
            t1, t2 = st.columns(2)
            with t1:
                st.metric("Input Tokens", f"{int(in_tok):,}" if pd.notna(in_tok) else "N/A")
            with t2:
                st.metric("Output Tokens", f"{int(out_tok):,}" if pd.notna(out_tok) else "N/A")

            err = row.get("error")
            if err and err != "None" and pd.notna(err):
                st.error(f"Analysis error: {err}")
