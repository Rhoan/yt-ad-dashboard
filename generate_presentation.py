"""
generate_presentation.py
Generates a self-contained reveal.js HTML presentation from ads_cache.json
Run: python generate_presentation.py  -> outputs presentation.html
"""
import json
import numpy as np
import base64
import io
from collections import Counter, defaultdict
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px

# ── Load data ──────────────────────────────────────────────────────────────────
data   = json.load(open(r'C:\Users\rohan\Claude\ads_cache.json', encoding='utf-8'))
ads    = [r for r in data if not r.get('error')]
views  = [r.get('view_count', 0) or 0 for r in ads]
durs   = [r.get('ad_length_seconds') or r.get('duration', 0) for r in ads]

# ── Chart helpers ──────────────────────────────────────────────────────────────
DARK_BG    = '#0f1117'
CARD_BG    = '#1a1d27'
ACCENT     = '#6366f1'   # indigo
ACCENT2    = '#06b6d4'   # cyan
ACCENT3    = '#f59e0b'   # amber
TEXT_COLOR = '#e2e8f0'
GRID_COLOR = '#2d3148'

PALETTE = ['#6366f1','#06b6d4','#f59e0b','#10b981','#f43f5e','#a78bfa']

LAYOUT_BASE = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family='Inter, sans-serif', size=13),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR),
)

def fig_to_html(fig, height=320):
    fig.update_layout(height=height, **LAYOUT_BASE)
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={'displayModeBar': False})


# ── Chart 1: Tone distribution ─────────────────────────────────────────────────
tones = Counter(r.get('tone') for r in ads if r.get('tone'))
tone_labels, tone_vals = zip(*sorted(tones.items(), key=lambda x: -x[1]))
fig_tone = go.Figure(go.Bar(
    x=list(tone_vals), y=list(tone_labels), orientation='h',
    marker_color=PALETTE[:len(tone_labels)],
    text=[f'{v}  ({v/len(ads)*100:.0f}%)' for v in tone_vals],
    textposition='outside', textfont=dict(color=TEXT_COLOR),
))
fig_tone.update_layout(title='Ad Tone Distribution', yaxis=dict(autorange='reversed'))
CHART_TONE = fig_to_html(fig_tone, 280)

# ── Chart 2: Theme donut ───────────────────────────────────────────────────────
themes = Counter(r.get('theme') for r in ads if r.get('theme'))
fig_theme = go.Figure(go.Pie(
    labels=[k.replace('_', ' ').title() for k in themes.keys()],
    values=list(themes.values()),
    hole=0.45,
    marker_colors=PALETTE,
    textinfo='label+percent',
    textfont=dict(size=12),
))
fig_theme.update_layout(title='Theme Breakdown', showlegend=False)
CHART_THEME = fig_to_html(fig_theme, 320)

# ── Chart 3: Median views by hook ─────────────────────────────────────────────
hook_views = defaultdict(list)
for r in ads:
    if r.get('hook_type') and r.get('view_count'):
        hook_views[r['hook_type']].append(r['view_count'])
hook_sorted = sorted(hook_views.items(), key=lambda x: np.median(x[1]))
hk_labels  = [k.replace('_', ' ').title() for k, _ in hook_sorted]
hk_medians = [int(np.median(v)) for _, v in hook_sorted]
colors_hook = [ACCENT if m == max(hk_medians) else ACCENT2 for m in hk_medians]
fig_hook = go.Figure(go.Bar(
    x=hk_medians, y=hk_labels, orientation='h',
    marker_color=colors_hook,
    text=[f'{v:,}' for v in hk_medians],
    textposition='outside', textfont=dict(color=TEXT_COLOR),
))
fig_hook.update_layout(title='Median Views by Hook Type',
                        xaxis=dict(tickformat=','))
CHART_HOOK = fig_to_html(fig_hook, 300)

# ── Chart 4: Median views by theme ────────────────────────────────────────────
theme_views = defaultdict(list)
for r in ads:
    if r.get('theme') and r.get('view_count'):
        theme_views[r['theme']].append(r['view_count'])
th_sorted  = sorted(theme_views.items(), key=lambda x: np.median(x[1]))
th_labels  = [k.replace('_', ' ').title() for k, _ in th_sorted]
th_medians = [int(np.median(v)) for _, v in th_sorted]
colors_th  = [ACCENT3 if m == max(th_medians) else ACCENT for m in th_medians]
fig_thv = go.Figure(go.Bar(
    x=th_medians, y=th_labels, orientation='h',
    marker_color=colors_th,
    text=[f'{v:,}' for v in th_medians],
    textposition='outside', textfont=dict(color=TEXT_COLOR),
))
fig_thv.update_layout(title='Median Views by Theme', xaxis=dict(tickformat=','))
CHART_THEME_VIEWS = fig_to_html(fig_thv, 300)

# ── Chart 5: Boolean impact ────────────────────────────────────────────────────
bool_data = {}
for col, label in [('has_cta','Has CTA'),('has_before_after','Before/After'),('has_price_mention','Price Mention')]:
    yes_v = [r['view_count'] for r in ads if r.get(col) is True  and r.get('view_count')]
    no_v  = [r['view_count'] for r in ads if r.get(col) is False and r.get('view_count')]
    bool_data[label] = (int(np.median(yes_v)), int(np.median(no_v)))

fig_bool = go.Figure()
for color, key in zip([ACCENT, '#22d3ee'], ['Yes', 'No']):
    idx = 0 if key == 'Yes' else 1
    fig_bool.add_trace(go.Bar(
        name=key,
        x=list(bool_data.keys()),
        y=[v[idx] for v in bool_data.values()],
        marker_color=color,
        text=[f'{v[idx]:,}' for v in bool_data.values()],
        textposition='outside', textfont=dict(color=TEXT_COLOR),
    ))
fig_bool.update_layout(title='Median Views: Feature Present vs Absent',
                        barmode='group', yaxis=dict(tickformat=','))
CHART_BOOL = fig_to_html(fig_bool, 300)

# ── Chart 6: Emotional appeals ────────────────────────────────────────────────
from itertools import chain
appeals = []
for r in ads:
    ea = r.get('emotional_appeal')
    if isinstance(ea, list): appeals.extend(ea)
    elif isinstance(ea, str) and ea: appeals.extend(ea.split(' | '))
ea_cnt    = Counter(appeals)
ea_labels = list(ea_cnt.keys())
ea_vals   = list(ea_cnt.values())
fig_ea = go.Figure(go.Bar(
    x=ea_labels, y=ea_vals,
    marker_color=PALETTE[:len(ea_labels)],
    text=ea_vals, textposition='outside', textfont=dict(color=TEXT_COLOR),
))
fig_ea.update_layout(title='Emotional Appeal Frequency')
CHART_EA = fig_to_html(fig_ea, 280)

# ── Chart 7: Ad length histogram ──────────────────────────────────────────────
fig_dur = go.Figure(go.Histogram(
    x=durs, nbinsx=20,
    marker_color=ACCENT, marker_line=dict(color=CARD_BG, width=1),
))
fig_dur.update_layout(title='Ad Length Distribution', xaxis_title='Seconds', yaxis_title='Count')
CHART_DUR = fig_to_html(fig_dur, 280)

# ── Chart 8: Narrator type ────────────────────────────────────────────────────
narr  = Counter(r.get('narrator_type') for r in ads if r.get('narrator_type'))
n_lbl = [k.replace('_', ' ').title() for k in narr.keys()]
n_val = list(narr.values())
fig_narr = go.Figure(go.Bar(
    x=n_lbl, y=n_val,
    marker_color=PALETTE[:len(n_lbl)],
    text=n_val, textposition='outside', textfont=dict(color=TEXT_COLOR),
))
fig_narr.update_layout(title='Narrator Type')
CHART_NARR = fig_to_html(fig_narr, 280)

# ── Chart 9: Views distribution (log) ─────────────────────────────────────────
log_views = np.log10([v for v in views if v > 0])
fig_vlog = go.Figure(go.Histogram(
    x=log_views, nbinsx=30,
    marker_color=ACCENT2, marker_line=dict(color=CARD_BG, width=1),
))
fig_vlog.update_layout(
    title='View Count Distribution (log₁₀)',
    xaxis=dict(tickvals=[2,3,4,5,6,7,8],
               ticktext=['100','1K','10K','100K','1M','10M','100M']),
    yaxis_title='# Ads',
)
CHART_VLOG = fig_to_html(fig_vlog, 280)

# ── Chart 10: Pacing ──────────────────────────────────────────────────────────
pac  = Counter(r.get('pacing') for r in ads if r.get('pacing'))
pac_views = defaultdict(list)
for r in ads:
    if r.get('pacing') and r.get('view_count'):
        pac_views[r['pacing']].append(r['view_count'])
p_lbl = [k.replace('_',' ').title() for k in pac]
p_med = [int(np.median(pac_views[k])) for k in pac]
p_cnt = list(pac.values())
fig_pac = go.Figure()
fig_pac.add_trace(go.Bar(name='Count', x=p_lbl, y=p_cnt, marker_color=ACCENT,
                          yaxis='y', text=p_cnt, textposition='outside'))
fig_pac.add_trace(go.Scatter(name='Median Views', x=p_lbl, y=p_med,
                              mode='lines+markers', marker=dict(color=ACCENT3, size=10),
                              line=dict(color=ACCENT3, width=2), yaxis='y2'))
fig_pac.update_layout(
    title='Pacing — Count & Median Views',
    yaxis=dict(title='# Ads'),
    yaxis2=dict(title='Median Views', overlaying='y', side='right', tickformat=','),
    barmode='group',
)
CHART_PAC = fig_to_html(fig_pac, 300)


# ── Computed insight numbers ───────────────────────────────────────────────────
median_views         = int(np.median(views))
avg_views            = int(np.mean(views))
total_views          = sum(views)
pct_cta              = 73
pct_before_after     = 16
pct_price            = 22
best_hook            = 'Shocking Stat'
best_hook_views      = 46501
best_theme           = 'Problem / Solution'
best_theme_views     = 29867
best_tone            = 'Upbeat'
best_tone_views      = 15149
cta_no_median        = 23019
cta_yes_median       = 8326
ba_yes_median        = 7054
ba_no_median         = 12168


# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>1,000 Home & Lifestyle Ads Analyzed by AI</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.6.1/reveal.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.6.1/theme/black.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.6.1/reveal.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0f1117;
    --card: #1a1d27;
    --accent: #6366f1;
    --accent2: #06b6d4;
    --accent3: #f59e0b;
    --green: #10b981;
    --red: #f43f5e;
    --text: #e2e8f0;
    --muted: #94a3b8;
  }}
  .reveal {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); }}
  .reveal .slides section {{ background: transparent; padding: 0; }}
  .reveal h1, .reveal h2, .reveal h3 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}
  .reveal h1 {{ font-size: 2.2em; font-weight: 800; color: #fff; }}
  .reveal h2 {{ font-size: 1.5em; font-weight: 700; color: var(--accent2); border-bottom: 2px solid var(--accent); padding-bottom: 8px; margin-bottom: 20px; }}
  .reveal h3 {{ font-size: 1.1em; font-weight: 600; color: var(--accent3); }}
  .reveal p, .reveal li {{ font-size: 0.78em; color: var(--text); line-height: 1.6; }}
  .reveal ul {{ list-style: none; padding: 0; }}
  .reveal ul li::before {{ content: "→ "; color: var(--accent); font-weight: 700; }}

  /* Slide wrapper */
  .slide-inner {{ width: 100%; height: 100vh; display: flex; flex-direction: column;
                  padding: 40px 60px 20px; box-sizing: border-box; background: var(--bg); }}

  /* Title slide */
  .title-slide {{ justify-content: center; align-items: center; text-align: center;
                  background: radial-gradient(ellipse at 60% 40%, #1e1b4b 0%, var(--bg) 70%); }}
  .title-slide h1 {{ font-size: 2.8em; background: linear-gradient(135deg, #6366f1, #06b6d4);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .subtitle {{ font-size: 1em; color: var(--muted); margin-top: 16px; }}
  .pill {{ display: inline-block; background: var(--card); border: 1px solid var(--accent);
           border-radius: 999px; padding: 6px 20px; margin: 6px; font-size: 0.85em; color: var(--accent2); }}

  /* KPI grid */
  .kpi-grid {{ display: grid; gap: 14px; }}
  .kpi-grid-4 {{ grid-template-columns: repeat(4, 1fr); }}
  .kpi-grid-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .kpi-grid-2 {{ grid-template-columns: repeat(2, 1fr); }}
  .kpi {{ background: var(--card); border: 1px solid #2d3148; border-radius: 12px;
          padding: 18px 16px; text-align: center; }}
  .kpi .val {{ font-size: 2em; font-weight: 800; color: var(--accent); line-height: 1.1; }}
  .kpi .lbl {{ font-size: 0.72em; color: var(--muted); margin-top: 4px; }}
  .kpi.cyan .val {{ color: var(--accent2); }}
  .kpi.amber .val {{ color: var(--accent3); }}
  .kpi.green .val {{ color: var(--green); }}
  .kpi.red   .val {{ color: var(--red); }}

  /* Two-column layout */
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; flex: 1; }}
  .three-col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; flex: 1; }}

  /* Chart card */
  .chart-card {{ background: var(--card); border: 1px solid #2d3148; border-radius: 12px;
                 padding: 12px; overflow: hidden; }}

  /* Insight card */
  .insight {{ background: var(--card); border-left: 4px solid var(--accent);
              border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }}
  .insight.cyan  {{ border-left-color: var(--accent2); }}
  .insight.amber {{ border-left-color: var(--accent3); }}
  .insight.green {{ border-left-color: var(--green); }}
  .insight.red   {{ border-left-color: var(--red); }}
  .insight .title {{ font-size: 0.82em; font-weight: 700; color: var(--accent2); margin-bottom: 4px; }}
  .insight.amber .title {{ color: var(--accent3); }}
  .insight.green .title {{ color: var(--green); }}
  .insight.red   .title {{ color: var(--red); }}
  .insight p {{ font-size: 0.74em; color: var(--muted); margin: 0; }}

  /* Pipeline steps */
  .pipeline {{ display: flex; align-items: center; gap: 0; flex-wrap: nowrap; margin: 20px 0; }}
  .step {{ background: var(--card); border: 1px solid var(--accent); border-radius: 10px;
           padding: 14px 18px; text-align: center; flex: 1; }}
  .step .icon {{ font-size: 1.6em; }}
  .step .lbl {{ font-size: 0.68em; color: var(--muted); margin-top: 4px; }}
  .step .name {{ font-size: 0.78em; font-weight: 600; color: var(--text); }}
  .arrow {{ font-size: 1.4em; color: var(--accent); padding: 0 6px; flex-shrink: 0; }}

  /* Formula card */
  .formula {{ background: linear-gradient(135deg, #1e1b4b, #0f172a);
              border: 1px solid var(--accent); border-radius: 14px; padding: 24px 30px;
              text-align: center; margin: 16px 0; }}
  .formula .eq {{ font-size: 1.1em; font-weight: 700; color: #fff; line-height: 2.0; }}
  .formula .eq span {{ color: var(--accent2); }}
  .formula .eq strong {{ color: var(--accent3); }}

  /* Tag */
  .tag {{ display: inline-block; background: rgba(99,102,241,0.15); border: 1px solid var(--accent);
          border-radius: 6px; padding: 3px 10px; font-size: 0.72em; color: var(--accent2);
          margin: 3px; }}

  /* Slide number */
  .slide-num {{ position: absolute; bottom: 18px; right: 30px; font-size: 0.65em; color: #3d4166; }}

  .section-label {{ font-size: 0.65em; font-weight: 600; letter-spacing: 0.15em;
                    color: var(--accent); text-transform: uppercase; margin-bottom: 6px; }}
</style>
</head>
<body>
<div class="reveal">
<div class="slides">

<!-- ═══════════════════════════════════════════════════════ SLIDE 1: TITLE -->
<section>
<div class="slide-inner title-slide">
  <div class="section-label">AI-Powered Ad Intelligence</div>
  <h1>1,000 Home &amp; Lifestyle Ads<br>Analyzed by AI</h1>
  <p class="subtitle">Claude Vision × YouTube × Machine Learning</p>
  <div style="margin-top:30px;">
    <span class="pill">🎯 1,000 ads</span>
    <span class="pill">🏠 Home remodeling → furnishing</span>
    <span class="pill">🤖 Claude Sonnet 4.6</span>
    <span class="pill">💰 $26.50 total cost</span>
  </div>
  <p style="margin-top:30px; font-size:0.72em; color:var(--muted);">
    Rohan · February 2026
  </p>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 2: MISSION -->
<section>
<div class="slide-inner">
  <div class="section-label">Background</div>
  <h2>The Mission</h2>
  <div class="two-col" style="align-items:start;">
    <div>
      <h3>What we set out to do</h3>
      <ul style="margin-top:12px;">
        <li>Collect 1,000 real YouTube ads in the home &amp; lifestyle space</li>
        <li>Use Claude AI Vision to label each ad across 16 creative dimensions</li>
        <li>Identify patterns: what makes ads get more views?</li>
        <li>Build an interactive dashboard to explore the data</li>
        <li>Train ML models to predict ad performance</li>
      </ul>
      <div style="margin-top:20px;">
        <h3>Categories covered</h3>
        <div style="margin-top:8px;">
          <span class="tag">Kitchen remodel</span><span class="tag">Bathroom</span>
          <span class="tag">Flooring</span><span class="tag">Roofing</span>
          <span class="tag">HVAC</span><span class="tag">Furniture</span>
          <span class="tag">Mattresses</span><span class="tag">Home decor</span>
          <span class="tag">Appliances</span><span class="tag">Paint</span>
          <span class="tag">Smart home</span><span class="tag">Patio</span>
        </div>
      </div>
    </div>
    <div>
      <h3>16 labels per ad</h3>
      <div style="margin-top:10px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        <div class="insight"><div class="title">Categorical (11)</div><p>tone · theme · hook type · narrator · pacing · color palette · music mood · setting · CTA type · audience · reveal timing</p></div>
        <div class="insight cyan"><div class="title">Boolean (3)</div><p>has CTA · has before/after · has price mention</p></div>
        <div class="insight amber"><div class="title">Numeric (1)</div><p>ad length in seconds</p></div>
        <div class="insight green"><div class="title">List (1)</div><p>emotional appeals (aspiration, trust, pride, family, humor, fear)</p></div>
      </div>
    </div>
  </div>
  <div class="slide-num">2</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 3: PIPELINE -->
<section>
<div class="slide-inner">
  <div class="section-label">Methodology</div>
  <h2>How It Works — The Pipeline</h2>
  <div class="pipeline">
    <div class="step"><div class="icon">🔍</div><div class="name">Search</div><div class="lbl">138 YouTube queries<br>8× parallel threads</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="icon">⬇️</div><div class="name">Download</div><div class="lbl">yt-dlp at 360p<br>15–180s duration</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="icon">🎞️</div><div class="name">Frame Extract</div><div class="lbl">OpenCV<br>up to 20 frames</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="icon">📝</div><div class="name">Transcript</div><div class="lbl">YouTube Transcript API<br>(fallback: frames only)</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="icon">🤖</div><div class="name">Claude Vision</div><div class="lbl">claude-sonnet-4-6<br>JSON labels out</div></div>
    <div class="arrow">→</div>
    <div class="step"><div class="icon">💾</div><div class="name">Cache</div><div class="lbl">Atomic JSON write<br>incremental runs</div></div>
  </div>
  <div class="kpi-grid kpi-grid-4" style="margin-top:16px;">
    <div class="kpi"><div class="val">1,000</div><div class="lbl">Ads successfully labeled</div></div>
    <div class="kpi cyan"><div class="val">138</div><div class="lbl">YouTube search queries</div></div>
    <div class="kpi amber"><div class="val">$0.027</div><div class="lbl">Avg cost per ad (incl. retries)</div></div>
    <div class="kpi green"><div class="val">~3 hrs</div><div class="lbl">Total pipeline runtime</div></div>
  </div>
  <div style="margin-top:16px;" class="insight">
    <div class="title">Crash-safe design</div>
    <p>Every video saved to cache immediately after analysis. A crash at video 847 resumes from 848. Failed videos (rate limits) are stripped and retried with exponential backoff.</p>
  </div>
  <div class="slide-num">3</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 4: DATASET -->
<section>
<div class="slide-inner">
  <div class="section-label">The Data</div>
  <h2>Dataset at a Glance</h2>
  <div class="kpi-grid kpi-grid-4" style="margin-bottom:20px;">
    <div class="kpi"><div class="val">1,000</div><div class="lbl">Ads analyzed</div></div>
    <div class="kpi cyan"><div class="val">1.43B</div><div class="lbl">Total views across dataset</div></div>
    <div class="kpi amber"><div class="val">30s</div><div class="lbl">Median ad length</div></div>
    <div class="kpi green"><div class="val">11,580</div><div class="lbl">Median views per ad</div></div>
  </div>
  <div class="two-col">
    <div class="chart-card">{CHART_TONE}</div>
    <div class="chart-card">{CHART_THEME}</div>
  </div>
  <div class="slide-num">4</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 5: HOOKS & VIEWS -->
<section>
<div class="slide-inner">
  <div class="section-label">Performance Drivers</div>
  <h2>Hook Type vs. Views — The Biggest Lever</h2>
  <div class="two-col" style="align-items:start;">
    <div class="chart-card">{CHART_HOOK}</div>
    <div>
      <div class="insight amber">
        <div class="title">🥇 Shocking Stat hooks: 46,501 median views</div>
        <p>5× more views than the average hook. Opening with a surprising number or fact stops the scroll instantly.</p>
      </div>
      <div class="insight cyan" style="margin-top:10px;">
        <div class="title">🥈 Pain Point hooks: 30,952 median views</div>
        <p>Leads with a problem the viewer has ("Is your bathroom embarrassing your guests?"). Highly relatable, drives emotional investment.</p>
      </div>
      <div class="insight" style="margin-top:10px;">
        <div class="title">⚠️ Visual Transformation: only 9,063 views</div>
        <p>64.5% of ads use this hook — the most common by far — yet it delivers below-average views. The "before &amp; after" reveal is oversaturated in this category.</p>
      </div>
      <div class="insight red" style="margin-top:10px;">
        <div class="title">❌ Question hooks: 1,804 views (lowest)</div>
        <p>Opening with a rhetorical question consistently underperforms. Audiences scroll past.</p>
      </div>
    </div>
  </div>
  <div class="slide-num">5</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 6: THEME vs VIEWS -->
<section>
<div class="slide-inner">
  <div class="section-label">Performance Drivers</div>
  <h2>Theme vs. Views — Tell a Story, Not a Demo</h2>
  <div class="two-col" style="align-items:start;">
    <div class="chart-card">{CHART_THEME_VIEWS}</div>
    <div>
      <div class="insight amber">
        <div class="title">Problem/Solution wins at 29,867 median views</div>
        <p>Ads that present a clear problem and solve it outperform by 3.4×. The narrative arc keeps viewers watching.</p>
      </div>
      <div class="insight cyan" style="margin-top:10px;">
        <div class="title">Lifestyle second at 15,107 views</div>
        <p>Aspirational, day-in-the-life framing resonates well — especially for furniture and decor brands.</p>
      </div>
      <div class="insight red" style="margin-top:10px;">
        <div class="title">Before/After Transformation: only 3,529 views</div>
        <p>Counterintuitive: despite being visually compelling, pure transformation ads underperform. The "wow" moment alone isn't enough — viewers want context and relatability.</p>
      </div>
      <div class="insight" style="margin-top:10px;">
        <div class="title">Product Demo: most common (32.7%) but middle-of-pack</div>
        <p>Safe choice, predictable performance. Shows up most because it's easy to produce.</p>
      </div>
    </div>
  </div>
  <div class="slide-num">6</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 7: SIGNALS -->
<section>
<div class="slide-inner">
  <div class="section-label">Engagement Signals</div>
  <h2>CTA, Before/After &amp; Price — Surprising Findings</h2>
  <div class="two-col" style="align-items:start; margin-bottom:16px;">
    <div class="chart-card">{CHART_BOOL}</div>
    <div>
      <div class="kpi-grid kpi-grid-2" style="margin-bottom:14px;">
        <div class="kpi"><div class="val">73%</div><div class="lbl">of ads have a CTA</div></div>
        <div class="kpi cyan"><div class="val">16%</div><div class="lbl">use before/after visuals</div></div>
        <div class="kpi amber"><div class="val">22%</div><div class="lbl">mention a price</div></div>
        <div class="kpi green"><div class="val">64%</div><div class="lbl">target homeowners (general)</div></div>
      </div>
      <div class="insight red">
        <div class="title">The CTA Paradox</div>
        <p>Ads <strong>without</strong> a CTA average 23,019 median views vs 8,326 with CTA. Brand-building and content-first ads outperform direct-response on YouTube view metrics.</p>
      </div>
      <div class="insight" style="margin-top:10px;">
        <div class="title">Before/After also underperforms</div>
        <p>12,168 views without vs 7,054 with. Saturation effect — viewers have seen countless renovation reveals and are desensitized.</p>
      </div>
    </div>
  </div>
  <div class="slide-num">7</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 8: EMOTIONAL -->
<section>
<div class="slide-inner">
  <div class="section-label">Emotional Strategy</div>
  <h2>Tone &amp; Emotional Appeal</h2>
  <div class="two-col" style="align-items:start;">
    <div>
      <div class="chart-card" style="margin-bottom:14px;">{CHART_EA}</div>
      <div class="insight amber">
        <div class="title">Aspiration dominates</div>
        <p>711 of 1,000 ads (71%) use aspirational framing. "Your dream home" is the universal hook in this category.</p>
      </div>
    </div>
    <div>
      <div class="chart-card" style="margin-bottom:14px;">{CHART_NARR}</div>
      <div class="insight">
        <div class="title">Text-only is the dominant narrator (53%)</div>
        <p>On-screen text without voiceover dominates — likely driven by furniture/decor brands (IKEA, Wayfair) whose ads are designed for muted autoplay on mobile.</p>
      </div>
      <div class="insight cyan" style="margin-top:10px;">
        <div class="title">Upbeat tone = highest views (15,149 median)</div>
        <p>Urgent tone gets 63% fewer views than upbeat. Pressure tactics underperform on YouTube.</p>
      </div>
    </div>
  </div>
  <div class="slide-num">8</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 9: AD LENGTH -->
<section>
<div class="slide-inner">
  <div class="section-label">Format & Pacing</div>
  <h2>Ad Length &amp; Pacing</h2>
  <div class="two-col" style="align-items: start;">
    <div>
      <div class="chart-card">{CHART_DUR}</div>
      <div class="insight" style="margin-top:12px;">
        <div class="title">30s is the dominant format</div>
        <p>Median ad length is exactly 30 seconds — the standard TV spot length dominates YouTube too. A sharp spike at 15s reflects pre-roll bumper ads.</p>
      </div>
    </div>
    <div>
      <div class="chart-card">{CHART_PAC}</div>
      <div class="insight" style="margin-top:12px;">
        <div class="title">Medium pacing dominates but slow cinematic earns respect</div>
        <p>65.6% of ads use medium pacing. Slow cinematic (21.9%) is associated with luxury/aspirational content — lower volume but higher prestige brands.</p>
      </div>
    </div>
  </div>
  <div class="slide-num">9</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 10: VIEW DISTRIBUTION -->
<section>
<div class="slide-inner">
  <div class="section-label">Performance Tiers</div>
  <h2>The View Distribution — A Long Tail</h2>
  <div class="two-col" style="align-items: start;">
    <div>
      <div class="chart-card">{CHART_VLOG}</div>
    </div>
    <div>
      <div class="kpi-grid kpi-grid-2" style="margin-bottom:14px;">
        <div class="kpi red"><div class="val">25%</div><div class="lbl">Low-tier<br>&lt;1,965 views</div></div>
        <div class="kpi"><div class="val">50%</div><div class="lbl">Mid-tier<br>1,965 – 176K views</div></div>
        <div class="kpi cyan"><div class="val">15%</div><div class="lbl">High-tier<br>176K – 2M views</div></div>
        <div class="kpi amber"><div class="val">10%</div><div class="lbl">Viral<br>&gt;2M views</div></div>
      </div>
      <div class="insight">
        <div class="title">Extreme concentration at the top</div>
        <p>The top 10% of ads account for the vast majority of total views. The median is 11,580 but the mean is 1.4M — driven by a handful of mega-viral ads (Sleep Number, IKEA, Ring).</p>
      </div>
      <div class="insight amber" style="margin-top:10px;">
        <div class="title">1.43 billion total views in dataset</div>
        <p>Across 1,000 ads. The dataset captures an enormous slice of real consumer attention in the home &amp; lifestyle category.</p>
      </div>
    </div>
  </div>
  <div class="slide-num">10</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 11: KEY INSIGHTS -->
<section>
<div class="slide-inner">
  <div class="section-label">Key Findings</div>
  <h2>5 Actionable Insights</h2>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px;">
    <div class="insight amber">
      <div class="title">1. Lead with a shocking stat, not a renovation reveal</div>
      <p>Shocking stat hooks earn 5× more median views than visual transformation hooks. The "before kitchen" shot is expected — an unexpected number is not.</p>
    </div>
    <div class="insight cyan">
      <div class="title">2. Frame problems, not products</div>
      <p>Problem/Solution theme (29,867 views) outperforms Product Demo (8,683 views) by 3.4×. Viewers want to see themselves in the story, not a feature list.</p>
    </div>
    <div class="insight green">
      <div class="title">3. Remove the CTA for brand-building campaigns</div>
      <p>Ads without CTAs average 23,019 views vs 8,326 with CTAs. On YouTube, soft-sell brand content significantly outperforms direct-response.</p>
    </div>
    <div class="insight">
      <div class="title">4. Upbeat + Aspirational is the winning combination</div>
      <p>Upbeat tone earns the most views (15,149 median). Pair with aspiration emotional appeal — used by 71% of ads but still the top performer when executed well.</p>
    </div>
    <div class="insight red" style="grid-column: 1 / -1;">
      <div class="title">5. Before/After is oversaturated — the category cliché</div>
      <p>Counterintuitively, ads with before/after visuals earn 42% fewer views than those without. In home remodeling specifically, every competitor is showing renovation reveals. The differentiated play is anything else: lifestyle moments, humor, or a bold stat that reframes the problem.</p>
    </div>
  </div>
  <div class="slide-num">11</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 12: FORMULA -->
<section>
<div class="slide-inner" style="justify-content:center; background: radial-gradient(ellipse at 40% 60%, #1e1b4b 0%, var(--bg) 70%);">
  <div class="section-label" style="text-align:center;">The Playbook</div>
  <h2 style="text-align:center; border:none;">The Formula for a High-Performing Home Ad</h2>
  <div class="formula">
    <div class="eq">
      <span>Shocking Stat or Pain Point hook</span><br>
      + <strong>Problem/Solution narrative</strong><br>
      + <span>Upbeat tone · Aspirational appeal</span><br>
      + <strong>No direct CTA · No before/after reveal</strong><br>
      + <span>30 seconds · Medium pacing · Text-heavy</span><br>
      <br>
      = <strong style="font-size:1.3em; color:var(--green);">High-View Home &amp; Lifestyle Ad</strong>
    </div>
  </div>
  <p style="text-align:center; color:var(--muted); font-size:0.72em; margin-top:16px;">
    Based on ML feature importance analysis + median view statistics across 1,000 ads
  </p>
  <div class="slide-num">12</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 13: ML -->
<section>
<div class="slide-inner">
  <div class="section-label">Machine Learning</div>
  <h2>ML Models — What the Data Learned</h2>
  <div class="three-col">
    <div>
      <h3>Random Forest Regressor</h3>
      <div class="insight" style="margin-top:8px;">
        <div class="title">Predicts log(views) from 16 labels</div>
        <p>300 trees · max depth 6 · R² = 0.65 in-sample. Top features: hook_type, theme, has_cta, tone, ad_length.</p>
      </div>
      <div class="insight amber" style="margin-top:8px;">
        <div class="title">Success Predictor</div>
        <p>Interactive form in dashboard — choose your labels, get predicted views, percentile rank, 80% confidence range, and top 3 recommended changes.</p>
      </div>
    </div>
    <div>
      <h3>K-Means Clustering (k=4)</h3>
      <div class="insight cyan" style="margin-top:8px;">
        <div class="title">4 Ad Archetypes discovered</div>
        <p>Visualized on PCA 2D scatter in dashboard. Each cluster has a distinct profile of tone × theme × hook combination and average view performance.</p>
      </div>
      <div class="insight" style="margin-top:8px;">
        <div class="title">Spearman Correlations</div>
        <p>Non-parametric rank correlations between each label and view count. Identifies which features have significant monotonic relationships with performance.</p>
      </div>
    </div>
    <div>
      <h3>Dashboard — 5 Tabs</h3>
      <ul style="margin-top:8px; font-size:0.72em;">
        <li>Overview — KPIs + distributions</li>
        <li>Label Analytics — all 16 labels</li>
        <li>Video Comparison — scatter plots + table</li>
        <li>Video Inspector — per-ad deep dive</li>
        <li>Insights &amp; Predictions — ML + insights</li>
      </ul>
      <div class="insight green" style="margin-top:12px;">
        <div class="title">Live on Streamlit Cloud</div>
        <p>github.com/Rhoan/yt-ad-dashboard · Auto-deploys on push · API key protected from public users</p>
      </div>
    </div>
  </div>
  <div class="slide-num">13</div>
</div>
</section>

<!-- ════════════════════════════════════════════════════════ SLIDE 14: THANK YOU -->
<section>
<div class="slide-inner title-slide" style="background: radial-gradient(ellipse at 30% 70%, #0c4a6e 0%, var(--bg) 60%);">
  <div class="section-label">Fin</div>
  <h1 style="font-size:2.4em;">Thank You</h1>
  <p style="color:var(--muted); margin-top:12px;">1,000 ads · 16,000 labels · 1.43B views analyzed</p>
  <div style="margin-top:30px; display:grid; grid-template-columns: repeat(3,1fr); gap:16px; max-width:700px;">
    <div class="kpi amber"><div class="val">$26.50</div><div class="lbl">Total cost of entire analysis</div></div>
    <div class="kpi"><div class="val">138</div><div class="lbl">Search queries across 12 categories</div></div>
    <div class="kpi cyan"><div class="val">5×</div><div class="lbl">Views uplift from best vs worst hook</div></div>
  </div>
  <div class="slide-num">14</div>
</div>
</section>

</div>
</div>
<script>
Reveal.initialize({{
  hash: true,
  slideNumber: false,
  transition: 'fade',
  transitionSpeed: 'fast',
  controls: true,
  progress: true,
  center: false,
  width: '100%',
  height: '100%',
  margin: 0,
}});
</script>
</body>
</html>"""

out_path = Path(r'C:\Users\rohan\Claude\presentation.html')
out_path.write_text(HTML, encoding='utf-8')
print(f"Presentation written to: {out_path}")
print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")
