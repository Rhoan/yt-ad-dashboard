"""
ad_dashboard.py — YouTube Ad Analysis Dashboard (5 tabs, port 8502)

Reads ads_cache.json produced by batch_analyzer.py and visualizes aggregate
patterns across home-remodeling ads.

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
# ML vocabulary — fixed to match batch_analyzer.py constants
# =============================================================================
CAT_VOCAB = {
    "tone":                  ["calm", "inspirational", "serious", "upbeat", "urgent"],
    "theme":                 ["before_after_transformation", "lifestyle", "price_offer",
                              "problem_solution", "product_demo", "testimonial"],
    "hook_type":             ["celebrity", "offer", "pain_point", "question",
                              "shocking_stat", "visual_transformation"],
    "narrator_type":         ["customer_testimonial", "mixed", "on_screen_talent",
                              "text_only", "voiceover"],
    "pacing":                ["fast_cuts", "medium", "slow_cinematic"],
    "color_palette":         ["cool", "high_contrast", "neutral", "warm"],
    "music_mood":            ["calm", "dramatic", "none", "tense", "upbeat"],
    "setting":               ["exterior", "interior", "mixed", "studio"],
    "cta_type":              ["limited_time_offer", "none", "phone_number",
                              "visit_store", "website"],
    "product_reveal_timing": ["early (<10s)", "late (>30s)", "mid (10-30s)", "never"],
    "target_audience":       ["budget_conscious", "diy", "families",
                              "homeowners_general", "luxury"],
}
CAT_COLS  = list(CAT_VOCAB.keys())
BOOL_COLS = ["has_cta", "has_before_after", "has_price_mention"]


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

    if "emotional_appeal" in df.columns:
        df["emotional_appeal_str"] = df["emotional_appeal"].apply(
            lambda x: " | ".join(x) if isinstance(x, list) else (str(x) if x else "")
        )
    else:
        df["emotional_appeal_str"] = ""

    for col in ["has_cta", "has_before_after", "has_price_mention"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: bool(x) if x is not None else False)

    for col in [
        "view_count", "like_count", "comment_count", "duration",
        "ad_length_seconds", "input_tokens", "output_tokens",
        "cost_usd", "elapsed_sec",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner="Training models (one-time, ~1 sec)...")
def build_models(cache_key: str, data_json: str) -> dict:
    """
    Fit Random Forest, KMeans, PCA on the full dataset.
    cache_key changes whenever ads_cache.json changes.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr

    df = pd.read_json(data_json)
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in ["view_count", "like_count", "comment_count", "ad_length_seconds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Build fixed-vocabulary OHE matrix
    ohe_parts, ohe_names = [], []
    for col in CAT_COLS:
        for val in CAT_VOCAB[col]:
            ohe_parts.append((df[col].fillna("") == val).astype(int).values)
            ohe_names.append(f"{col}__{val}")
    for col in BOOL_COLS:
        ohe_parts.append(df[col].astype(int).values)
        ohe_names.append(col)

    X = np.column_stack(ohe_parts).astype(float)
    y = np.log1p(df["view_count"].values)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=1)
    rf.fit(X, y)
    train_r2 = float(r2_score(y, rf.predict(X)))

    # KMeans
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Spearman correlations vs view_count
    spearman = {}
    for feat in ["like_count", "comment_count", "ad_length_seconds"] + BOOL_COLS:
        col_data = df[feat].fillna(0).astype(float).values
        rho, p = spearmanr(col_data, df["view_count"].values)
        spearman[feat] = (round(float(rho), 4), round(float(p), 4))

    v = df["view_count"].values
    thresholds = {
        90: float(np.percentile(v, 90)),
        75: float(np.percentile(v, 75)),
        50: float(np.percentile(v, 50)),
    }

    return {
        "X": X, "y": y, "ohe_names": ohe_names,
        "rf": rf, "train_r2": train_r2,
        "km": km, "cluster_labels": cluster_labels.tolist(),
        "pca": pca, "X_pca": X_pca.tolist(),
        "spearman": spearman,
        "thresholds": thresholds,
        "n": len(df),
    }


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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Label Analytics", "Video Comparison", "Video Inspector",
    "Insights & Predictions",
])

# =============================================================================
# TAB 1 — Overview
# =============================================================================
with tab1:
    st.header("Overview")

    if df.empty:
        st.warning("No videos match the current filters.")
    else:
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
                    labels={"ba_jitter": "Has Before/After  (0=No, 1=Yes)",
                            "view_count": "Views"},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

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
                display_key = k.replace("_str", "").replace("_", " ").title()
                if isinstance(val, bool):
                    display_val = "Yes" if val else "No"
                elif not isinstance(val, (list, dict)) and pd.isna(val):
                    display_val = "N/A"
                else:
                    display_val = str(val)
                label_rows.append({"Label": display_key, "Value": display_val})

            st.table(pd.DataFrame(label_rows))

            with st.expander("Transcript"):
                st.write("Transcripts are not stored in the cache.")

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

# =============================================================================
# TAB 5 — Insights & Predictions
# =============================================================================
with tab5:
    st.header("Insights & Predictions")
    st.caption("Models trained on the full dataset. Sections 1–2 respect sidebar filters.")

    if df_all.empty:
        st.warning("No data yet.")
    else:
        # Build / load cached models
        cache_key = f"{len(df_all)}_{df_all['id'].str.cat(sep='')[:64]}"
        m = build_models(cache_key, df_all.to_json())

        thresholds = m["thresholds"]
        p50, p75, p90 = thresholds[50], thresholds[75], thresholds[90]

        def get_tier(v):
            if v >= p90: return "Top 10%"
            if v >= p75: return "Top 25%"
            if v >= p50: return "Top 50%"
            return "Below Median"

        # -------------------------------------------------------------------
        # Section 1 — Performance Overview
        # -------------------------------------------------------------------
        st.subheader("1. Performance Overview")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Videos in View", len(df), delta=f"of {len(df_all)} total")
        k2.metric("Median Views", f"{int(df['view_count'].median()):,}")
        k3.metric("Top 10% Threshold", f"{int(p90):,}",
                  help="Computed from full dataset")
        pct_top = (df["view_count"] >= p75).mean() * 100
        k4.metric("% Top-Quartile", f"{pct_top:.0f}%")

        chart_l, chart_r = st.columns(2)

        with chart_l:
            views_clean = df["view_count"].dropna().values
            if len(views_clean) > 1:
                lo_v = max(views_clean.min(), 100)
                hi_v = views_clean.max() + 1
                log_bins = np.logspace(np.log10(lo_v), np.log10(hi_v), 35)
                counts, edges = np.histogram(views_clean, bins=log_bins)
                centers = np.sqrt(edges[:-1] * edges[1:])
                hist_df = pd.DataFrame({"views": centers, "count": counts})
                fig = px.bar(hist_df, x="views", y="count", log_x=True,
                             title="View Count Distribution (log scale)",
                             labels={"views": "Views", "count": "# Ads"})
                fig.update_traces(marker_color="#4a90d9")
                for pct_val, color, label in [
                    (p50, "blue",   f"p50: {int(p50):,}"),
                    (p75, "orange", f"p75: {int(p75):,}"),
                    (p90, "red",    f"p90: {int(p90):,}"),
                ]:
                    if lo_v <= pct_val <= hi_v:
                        fig.add_vline(x=pct_val, line_dash="dash", line_color=color,
                                      annotation_text=label)
                st.plotly_chart(fig, use_container_width=True)

        with chart_r:
            tier_counts = df["view_count"].apply(get_tier).value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            tier_order = ["Top 10%", "Top 25%", "Top 50%", "Below Median"]
            tier_color_map = {
                "Top 10%": "#e63946", "Top 25%": "#f4a261",
                "Top 50%": "#f9c74f", "Below Median": "#adb5bd",
            }
            tier_counts["tier"] = pd.Categorical(
                tier_counts["tier"], categories=tier_order, ordered=True
            )
            tier_counts = tier_counts.sort_values("tier")
            fig = px.bar(tier_counts, x="tier", y="count", color="tier",
                         color_discrete_map=tier_color_map,
                         title="Performance Tier Breakdown",
                         labels={"tier": "", "count": "# Ads"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------------
        # Section 2 — What Drives Views
        # -------------------------------------------------------------------
        st.subheader("2. What Drives Views")
        st.caption("Median views per label value, sorted highest first. Red bar = best-performing value for that label.")

        grid = st.columns(3)
        for i, cat in enumerate(CAT_COLS):
            if cat not in df.columns:
                continue
            med = (
                df.dropna(subset=[cat])
                  .groupby(cat)["view_count"]
                  .agg(median_views="median", n="count")
                  .reset_index()
                  .sort_values("median_views", ascending=False)
            )
            if med.empty:
                continue
            med["color"] = ["#e63946"] + ["#4a90d9"] * (len(med) - 1)
            fig = px.bar(
                med, x=cat, y="median_views",
                color="color", color_discrete_map="identity",
                hover_data={"n": True, "color": False},
                title=cat.replace("_", " ").title(),
                labels={cat: "", "median_views": "Median Views"},
            )
            fig.update_layout(showlegend=False, margin=dict(t=40, b=10))
            grid[i % 3].plotly_chart(fig, use_container_width=True)

        st.markdown("**Boolean Label Impact on Median Views**")
        bool_disp = st.columns(3)
        overall_med = float(df["view_count"].median())
        for i, col in enumerate(BOOL_COLS):
            if col not in df.columns:
                continue
            t_med = df[df[col] == True]["view_count"].median()
            f_med = df[df[col] == False]["view_count"].median()
            if pd.isna(t_med): t_med = 0
            if pd.isna(f_med): f_med = 0
            lift = (t_med / overall_med - 1) * 100 if overall_med else 0
            sign = "+" if lift >= 0 else ""
            bool_disp[i].metric(
                col.replace("_", " ").title(),
                f"{int(t_med):,} (True)",
                delta=f"{sign}{lift:.0f}% vs median",
            )
            bool_disp[i].caption(f"False: {int(f_med):,} median views")

        st.markdown("---")

        # -------------------------------------------------------------------
        # Section 3 — Feature Importance & Correlations
        # -------------------------------------------------------------------
        st.subheader("3. Feature Importance & Correlations")
        st.caption(
            f"Random Forest trained on {m['n']} ads (train R²={m['train_r2']:.2f}). "
            "n=203 is small — importances show directional signal, not precise ranking. "
            "Out-of-sample predictive power is near zero."
        )

        imp_df = (
            pd.DataFrame({"feature": m["ohe_names"],
                          "importance": m["rf"].feature_importances_})
            .sort_values("importance", ascending=True)
            .tail(20)
        )
        imp_df["display"] = imp_df["feature"].str.replace("__", ": ", regex=False)
        fig = px.bar(
            imp_df, x="importance", y="display", orientation="h",
            title="Top 20 Feature Importances (Random Forest)",
            labels={"importance": "Importance", "display": ""},
            color="importance", color_continuous_scale="Blues",
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

        spear_rows = []
        for feat, (rho, p_val) in m["spearman"].items():
            spear_rows.append({
                "feature": feat.replace("_", " ").title(),
                "rho": rho,
                "p_val": p_val,
                "sig": "★ p<0.05" if p_val < 0.05 else "n.s.",
                "color": "#2a9d8f" if rho > 0 else "#e76f51",
                "opacity": 1.0 if p_val < 0.05 else 0.45,
            })
        spear_df = pd.DataFrame(spear_rows).sort_values("rho")
        fig = px.bar(
            spear_df, x="rho", y="feature", orientation="h",
            color="color", color_discrete_map="identity",
            title="Spearman Correlation with View Count",
            labels={"rho": "Spearman ρ", "feature": ""},
            hover_data={"sig": True, "p_val": True, "color": False},
        )
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(showlegend=False)
        fig.update_traces(marker_opacity=spear_df["opacity"].tolist())
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Faded bars = not statistically significant (p ≥ 0.05) at n=203.")

        st.markdown("---")

        # -------------------------------------------------------------------
        # Section 4 — Ad Archetypes
        # -------------------------------------------------------------------
        st.subheader("4. Ad Archetypes")

        cluster_labels_arr = np.array(m["cluster_labels"])
        X_pca_arr = np.array(m["X_pca"])

        pca_df = pd.DataFrame({
            "PC1":    X_pca_arr[:, 0],
            "PC2":    X_pca_arr[:, 1],
            "Cluster": [f"Cluster {c}" for c in cluster_labels_arr],
            "views":  df_all["view_count"].fillna(0).values,
            "title":  df_all["title"].values,
            "tone":   df_all["tone"].values,
            "theme":  df_all["theme"].values,
        })
        exp_var = m["pca"].explained_variance_ratio_
        fig = px.scatter(
            pca_df, x="PC1", y="PC2", color="Cluster",
            hover_data=["tone", "theme", "views", "title"],
            title="Ad Archetypes — PCA 2D (KMeans k=4)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            xaxis_title=f"PC1 ({exp_var[0]*100:.1f}% variance)",
            yaxis_title=f"PC2 ({exp_var[1]*100:.1f}% variance)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"PCA explains {(exp_var[0]+exp_var[1])*100:.0f}% of total variance. "
            "Clusters capture dominant label co-occurrences."
        )

        cluster_rows = []
        for c in range(4):
            mask = cluster_labels_arr == c
            sub = df_all[mask]
            row_d = {
                "Cluster": f"Cluster {c}",
                "Size": int(mask.sum()),
                "Median Views": f"{int(sub['view_count'].median()):,}",
            }
            for col in ["tone", "theme", "hook_type", "pacing"]:
                if col in sub.columns and not sub[col].isna().all():
                    row_d[col.replace("_", " ").title()] = sub[col].mode().iloc[0]
            cluster_rows.append(row_d)
        st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------------
        # Section 5 — Success Predictor
        # -------------------------------------------------------------------
        st.subheader("5. Success Predictor")
        st.caption(
            "Pick label values to estimate where your ad would rank among the 203 analyzed. "
            "Unlisted labels (color palette, music mood, setting, cta type, product reveal) "
            "are treated as neutral."
        )

        with st.form("predictor_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                p_tone     = st.selectbox("Tone",            sorted(CAT_VOCAB["tone"]))
                p_theme    = st.selectbox("Theme",           sorted(CAT_VOCAB["theme"]))
                p_hook     = st.selectbox("Hook Type",       sorted(CAT_VOCAB["hook_type"]))
            with fc2:
                p_pacing   = st.selectbox("Pacing",          sorted(CAT_VOCAB["pacing"]))
                p_audience = st.selectbox("Target Audience", sorted(CAT_VOCAB["target_audience"]))
                p_narrator = st.selectbox("Narrator Type",   sorted(CAT_VOCAB["narrator_type"]))
            with fc3:
                p_ba       = st.checkbox("Has Before/After", value=True)
                p_cta      = st.checkbox("Has CTA",          value=True)
                p_price    = st.checkbox("Has Price Mention")
                p_length   = st.slider("Ad Length (s)", 15, 180, 30, step=5)
            submitted = st.form_submit_button("Predict Performance", type="primary")

        if submitted:
            form_vals = {
                "tone": p_tone, "theme": p_theme, "hook_type": p_hook,
                "narrator_type": p_narrator, "pacing": p_pacing,
                "target_audience": p_audience,
                "has_cta": p_cta, "has_before_after": p_ba,
                "has_price_mention": p_price,
            }
            ohe_names = m["ohe_names"]
            x_row = np.zeros(len(ohe_names), dtype=float)
            for idx, name in enumerate(ohe_names):
                if "__" in name:
                    feat, val = name.split("__", 1)
                    if form_vals.get(feat) == val:
                        x_row[idx] = 1.0
                else:
                    x_row[idx] = float(form_vals.get(name, False))

            rf = m["rf"]
            pred_log   = float(rf.predict(x_row.reshape(1, -1))[0])
            pred_views = float(np.expm1(pred_log))
            actual_v   = df_all["view_count"].fillna(0).values
            percentile = float(np.mean(actual_v < pred_views) * 100)

            tree_preds = np.array([
                t.predict(x_row.reshape(1, -1))[0] for t in rf.estimators_
            ])
            lo_p = float(np.expm1(np.percentile(tree_preds, 10)))
            hi_p = float(np.expm1(np.percentile(tree_preds, 90)))

            tier_label  = get_tier(pred_views)
            tier_emoji  = {
                "Top 10%": "🔴", "Top 25%": "🟠",
                "Top 50%": "🟡", "Below Median": "⚪",
            }
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Percentile", f"{percentile:.0f}th")
            r2.metric("Expected Views", f"{int(pred_views):,}",
                      delta=f"{pred_views / float(np.median(actual_v)):.1f}x median")
            r3.metric("80% Prediction Range", f"{int(lo_p):,} – {int(hi_p):,}")
            st.markdown(f"### Predicted tier: {tier_emoji[tier_label]} **{tier_label}**")

            if hi_p > 0 and hi_p / max(lo_p, 1) > 100:
                st.warning(
                    "High uncertainty — this combination is underrepresented in the training data. "
                    "The 80% range spans more than 100x."
                )

            # Top 3 improvements
            improvements = []
            for cat in ["tone", "theme", "hook_type", "pacing", "narrator_type", "target_audience"]:
                cur_val = form_vals.get(cat, "")
                for alt_val in CAT_VOCAB[cat]:
                    if alt_val == cur_val:
                        continue
                    x_alt = x_row.copy()
                    for idx, name in enumerate(ohe_names):
                        if name.startswith(cat + "__"):
                            x_alt[idx] = 0.0
                    alt_col = f"{cat}__{alt_val}"
                    if alt_col in ohe_names:
                        x_alt[ohe_names.index(alt_col)] = 1.0
                    alt_log  = float(rf.predict(x_alt.reshape(1, -1))[0])
                    lift_log = alt_log - pred_log
                    if lift_log > 0.05:
                        lift_pct = int(
                            (np.expm1(pred_log + lift_log) / max(np.expm1(pred_log), 1) - 1) * 100
                        )
                        improvements.append((cat, cur_val, alt_val, lift_pct))

            improvements.sort(key=lambda x: -x[3])
            if improvements:
                st.markdown("**Top 3 Recommended Changes**")
                for cat, cur, alt, lift_pct in improvements[:3]:
                    st.info(
                        f"Change **{cat.replace('_', ' ').title()}** from "
                        f"`{cur}` → `{alt}`  →  **+{lift_pct}% expected views**"
                    )
            else:
                st.success("Your combination is already near-optimal for the modelled features.")

        st.markdown("---")

        # -------------------------------------------------------------------
        # Section 6 — Key Insights
        # -------------------------------------------------------------------
        st.subheader("6. Key Insights")
        st.caption("All insights computed from the full 203-ad dataset.")

        overall_med_all = float(df_all["view_count"].median())

        def _fmt(v): return f"{int(v):,}"
        def _mult(a, b): return f"{a/b:.1f}x" if b else "N/A"

        insights = []

        # Best hook
        hook_med = df_all.groupby("hook_type")["view_count"].median()
        best_hook = hook_med.idxmax()
        worst_hook = hook_med.idxmin()
        insights.append(
            f"**Hook type has the biggest spread:** Ads with a "
            f"**{best_hook.replace('_',' ')}** hook average "
            f"**{_mult(hook_med.max(), hook_med.min())}** more views than "
            f"**{worst_hook.replace('_',' ')}** hooks "
            f"({_fmt(hook_med.max())} vs {_fmt(hook_med.min())} median)."
        )

        # CTA effect
        cta_t = df_all[df_all["has_cta"] == True]["view_count"].median()
        cta_f = df_all[df_all["has_cta"] == False]["view_count"].median()
        if pd.notna(cta_t) and pd.notna(cta_f):
            if cta_f > cta_t:
                insights.append(
                    f"**CTA paradox:** Ads *without* a CTA have "
                    f"**{_mult(cta_f, cta_t)}** higher median views "
                    f"({_fmt(cta_f)} vs {_fmt(cta_t)}). "
                    "High-view content is often organic/branded, not direct-response."
                )
            else:
                insights.append(
                    f"**CTA lift:** Ads *with* a CTA outperform those without by "
                    f"**{_mult(cta_t, cta_f)}** ({_fmt(cta_t)} vs {_fmt(cta_f)})."
                )

        # Best theme
        theme_med = df_all.groupby("theme")["view_count"].median()
        best_theme = theme_med.idxmax()
        insights.append(
            f"**Best theme:** **{best_theme.replace('_',' ').title()}** outperforms all "
            f"other themes with **{_fmt(int(theme_med.max()))} median views** "
            f"({_mult(theme_med.max(), overall_med_all)} the overall median)."
        )

        # Pacing
        pace_med = df_all.groupby("pacing")["view_count"].median()
        if len(pace_med) >= 2:
            best_pace = pace_med.idxmax()
            worst_pace = pace_med.idxmin()
            insights.append(
                f"**Pacing edge:** **{best_pace.replace('_',' ').title()}** ads generate "
                f"**{_mult(pace_med.max(), pace_med.min())}** more views than "
                f"**{worst_pace.replace('_',' ')}** "
                f"({_fmt(pace_med.max())} vs {_fmt(pace_med.min())} median)."
            )

        # Before/after
        ba_t = df_all[df_all["has_before_after"] == True]["view_count"].median()
        ba_f = df_all[df_all["has_before_after"] == False]["view_count"].median()
        ba_pct = (df_all["has_before_after"] == True).mean() * 100
        if pd.notna(ba_t) and pd.notna(ba_f):
            direction = "more" if ba_t > ba_f else "fewer"
            insights.append(
                f"**Before/After:** {ba_pct:.0f}% of ads use before/after visuals. "
                f"These get **{_mult(max(ba_t,ba_f), min(ba_t,ba_f))}** {direction} views "
                f"than ads without ({_fmt(ba_t)} vs {_fmt(ba_f)})."
            )

        for text in insights:
            st.info(text)
