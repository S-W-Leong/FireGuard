# -*- coding: utf-8 -*-
"""
app.py — FireGuard Executive Decision-Support Dashboard
=======================================================
Run:    python -m streamlit run app.py
Deps:   streamlit plotly pandas numpy

Track 1: Predictive Model Track — Predict Early, Decide Better
Operational Pilot: South Sumatra Peatland Monitoring & Resource Allocation
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data_loader import load_fireguard_data, load_evaluation_metrics, _add_derived_cols

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="FireGuard — Peatland Monitoring & Resource Allocation Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# GLOBAL CSS  — slate-950 dark aesthetic matching Lovable screenshots
# ===========================================================================
st.markdown("""
<style>
/* ── Google Fonts ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ─────────────────────────────────────────────────────── */
:root {
  --bg-base:       #020617;
  --bg-card:       #0f172a;
  --bg-deep:       #080f1f;
  --border:        #1e293b;
  --border-accent: #334155;
  --txt-primary:   #f1f5f9;
  --txt-secondary: #94a3b8;
  --txt-muted:     #475569;
  --amber:         #f59e0b;
  --amber-lt:      #fbbf24;
  --crimson:       #ef4444;
  --green:         #22c55e;
  --blue:          #3b82f6;
  --purple:        #8b5cf6;
  --teal:          #14b8a6;
}

/* ── Full-app dark background ──────────────────────────────────────────── */
html, body, [class*="css"], .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
  background-color: var(--bg-base) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--txt-primary) !important;
}
.block-container { padding-top: 0.5rem !important; }

/* ── Hide Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background-color: #080f1f !important;
  border-right: 1px solid var(--border) !important;
  min-width: 300px !important;
  max-width: 340px !important;
}
[data-testid="stSidebar"] * { color: var(--txt-primary) !important; }
[data-testid="stSidebarContent"] { padding: 1rem 1.25rem !important; }

/* ── Slider track & thumb ───────────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background-color: var(--amber) !important;
  border-color: var(--amber-lt) !important;
  box-shadow: 0 0 6px var(--amber) !important;
}
/* Active fill */
[data-testid="stSlider"] > div > div > div:first-child > div {
  background: linear-gradient(90deg, var(--amber), var(--crimson)) !important;
}

/* ── Sidebar stat cards ─────────────────────────────────────────────────── */
.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
  margin-top: 10px;
}
.stat-label {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--txt-muted);
  margin-bottom: 4px;
}
.stat-value          { font-size: 22px; font-weight: 800; color: var(--txt-primary); }
.stat-value.green    { color: var(--green); }
.stat-value.amber    { color: var(--amber); }
.stat-value.crimson  { color: var(--crimson); }

/* ── Formula hint ───────────────────────────────────────────────────────── */
.formula-hint {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--txt-muted);
  background: #050d1a;
  border-left: 2px solid var(--border-accent);
  padding: 8px 10px;
  border-radius: 4px;
  margin-top: 14px;
  line-height: 1.7;
}

/* ── App header ─────────────────────────────────────────────────────────── */
.app-header {
  padding: 14px 0 12px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.app-title {
  font-size: 21px;
  font-weight: 700;
  color: var(--txt-primary);
}
.app-subtitle {
  font-size: 12px;
  color: var(--txt-secondary);
  margin-top: 3px;
}
.badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 11px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.03em;
  border: 1px solid;
}
.badge-amber  { background: rgba(245,158,11,.15); border-color: rgba(245,158,11,.4); color: var(--amber-lt); }
.badge-blue   { background: rgba(59,130,246,.15); border-color: rgba(59,130,246,.4); color: #60a5fa; }
.badge-green  { background: rgba(34,197,94,.15);  border-color: rgba(34,197,94,.4);  color: #4ade80; }

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-card) !important;
  border-bottom: 1px solid var(--border) !important;
  border-radius: 10px 10px 0 0 !important;
  padding: 4px 8px 0 8px !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--txt-secondary) !important;
  border-bottom: 2px solid transparent !important;
  padding: 10px 18px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: color .2s;
}
.stTabs [aria-selected="true"] {
  color: var(--txt-primary) !important;
  border-bottom: 2px solid var(--blue) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 10px 10px !important;
  padding: 20px !important;
}

/* ── Section headers inside tabs ────────────────────────────────────────── */
.section-hdr {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--txt-muted);
  margin: 18px 0 8px 0;
  padding-bottom: 5px;
  border-bottom: 1px solid var(--border);
}

/* ── Cell detail card ───────────────────────────────────────────────────── */
.detail-card {
  background: var(--bg-deep);
  border: 1px solid var(--border-accent);
  border-radius: 12px;
  padding: 20px;
  margin-top: 14px;
}
.detail-card h3 {
  font-size: 15px;
  font-weight: 700;
  color: var(--txt-primary);
  margin: 0 0 4px 0;
}
.metric-chip {
  display: inline-block;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 9px 15px;
  margin: 6px 6px 0 0;
  text-align: center;
  min-width: 90px;
}
.metric-chip .val { font-size: 19px; font-weight: 800; }
.metric-chip .lbl { font-size: 9px; color: var(--txt-muted); text-transform: uppercase; letter-spacing: .1em; margin-top: 2px; }
.explain-box {
  background: rgba(245,158,11,.07);
  border: 1px solid rgba(245,158,11,.22);
  border-radius: 8px;
  padding: 14px;
  margin-top: 12px;
  font-size: 13px;
  color: var(--txt-secondary);
  line-height: 1.65;
}
.directive-box {
  background: rgba(59,130,246,.07);
  border: 1px solid rgba(59,130,246,.22);
  border-radius: 8px;
  padding: 14px;
  margin-top: 8px;
  font-size: 13px;
  color: #93c5fd;
  line-height: 1.65;
}

/* ── Streamlit metric tiles ─────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 14px 16px !important;
}
[data-testid="metric-container"] label {
  color: var(--txt-secondary) !important;
  font-size: 10px !important;
  text-transform: uppercase;
  letter-spacing: .1em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--txt-primary) !important;
  font-size: 22px !important;
  font-weight: 800 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Info blocks (Tab 3) ────────────────────────────────────────────────── */
.info-block {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  margin-top: 12px;
  font-size: 13px;
  color: var(--txt-secondary);
  line-height: 1.75;
}
.info-block h4 { color: var(--txt-primary); font-size: 14px; font-weight: 700; margin: 0 0 10px 0; }
.info-block code {
  background: var(--bg-deep);
  border-radius: 4px;
  padding: 2px 6px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--amber);
}
.info-block table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
.info-block th { text-align: left; padding: 8px; color: var(--txt-muted); border-bottom: 1px solid var(--border); }
.info-block td { padding: 7px 8px; border-bottom: 1px solid #0a1628; }
.info-block td:first-child { color: var(--txt-primary); }
.info-block td:not(:first-child) { color: var(--txt-secondary); }

/* ── Radio (view toggle) ────────────────────────────────────────────────── */
[data-testid="stRadio"] > div { flex-direction: row !important; gap: 6px !important; flex-wrap: wrap; }
[data-testid="stRadio"] label {
  background: var(--bg-deep);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 12px !important;
  font-size: 12px !important;
  cursor: pointer;
  transition: border-color .2s;
}
[data-testid="stRadio"] label:hover { border-color: var(--border-accent); }

/* ── Dataframe ──────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; overflow: hidden; }

/* ── Selectbox ──────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-deep) !important;
  border-color: var(--border-accent) !important;
  color: var(--txt-primary) !important;
  border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# MAP CONSTANTS
# ===========================================================================
MAP_LAT    = -3.2
MAP_LON    = 104.8
MAP_ZOOM   = 7.2
MAP_STYLE  = "carto-darkmatter"

TIER_HEX = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f97316",
    "MODERATE": "#eab308",
    "LOW":      "#22c55e",
}


# ===========================================================================
# CACHED DATA LOAD
# ===========================================================================
@st.cache_data(show_spinner=False)
def _raw_data() -> pd.DataFrame:
    return load_fireguard_data()

raw_df = _raw_data()


# ===========================================================================
# SIDEBAR  — Resource Constraints & Weights
# ===========================================================================
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:14px 0 12px 0;
                border-bottom:1px solid #1e293b;margin-bottom:14px;">
      <span style="font-size:18px;">⚙️</span>
      <span style="font-size:14px;font-weight:700;color:#f1f5f9;">Resource Constraints &amp; Weights</span>
    </div>
    """, unsafe_allow_html=True)

    k          = st.slider("Available Monitoring Teams (K)", 1, 20, 10)
    pop_weight = st.slider("Population Exposure Weight",     0.0, 1.0, 0.6, 0.05)
    agri_weight= st.slider("Agricultural Exposure Weight",  0.0, 1.0, 0.4, 0.05)

    # Derive scores with sidebar weights
    df = _add_derived_cols(raw_df, pop_weight=pop_weight, agri_weight=agri_weight)
    df["dispatched"] = df["rank"] <= k

    # Estimated loss averted
    top_k = df[df["dispatched"]]
    avg_ps  = top_k["priority_score"].mean() if len(top_k) else 0.0
    hit_frac= top_k["ground_truth_fire"].mean() if len(top_k) else 0.0
    loss_M  = round(avg_ps * hit_frac * k * 4.2, 1)

    st.markdown(f"""
    <div class="stat-card">
      <div class="stat-label">Total Grid Cells Analyzed</div>
      <div class="stat-value">{len(df)} cells</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Teams Dispatched</div>
      <div class="stat-value amber">{k} / {len(df)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Estimated Loss Averted</div>
      <div class="stat-value green">${loss_M}M USD eq.</div>
    </div>
    <div class="formula-hint">
      Priority = ML prob &times; (w<sub>pop</sub> &middot; Pop + w<sub>agr</sub> &middot; Agri).<br>
      Top-K &rarr; DISPATCH.
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# MAIN HEADER
# ===========================================================================
st.markdown("""
<div class="app-header">
  <div class="app-title">🔥 FireGuard — Peatland Monitoring &amp; Resource Allocation Engine</div>
  <div class="app-subtitle">Track 1: Predict Early, Decide Better &nbsp;|&nbsp; South Sumatra Operational Pilot</div>
  <div class="badge-row">
    <span class="badge badge-amber">⚡ Supervised ML: GBDT (Calibrated)</span>
    <span class="badge badge-blue">⏱ Horizon: 7-Day Pre-Ignition</span>
    <span class="badge badge-green">✦ Base: OpenStreetMap</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3 = st.tabs([
    "🗺️  Operational Dispatch & Map",
    "📊  Decision Evaluation & Baselines",
    "📋  Model Card & Trust Architecture",
])


# ---------------------------------------------------------------------------
# HELPER — Plotly Scattermap figure builder
# ---------------------------------------------------------------------------
def build_map(
    df: pd.DataFrame,
    view_mode: str,
    k: int,
    show_ground_truth: bool = False,
) -> go.Figure:
    """Return a dark-themed Plotly Scattermap for the chosen view mode."""

    dispatched = df[df["rank"] <= k].copy()
    monitor    = df[df["rank"] >  k].copy()

    # -- Colour value selector per view ------------------------------------
    def _cvals(sub: pd.DataFrame) -> pd.Series:
        if view_mode == "FireGuard Dispatch Priority (Top-K)":
            return sub["priority_score"]
        if view_mode == "Supervised ML Probability":
            return sub["ml_prob"]
        if view_mode == "Copernicus FWI Baseline":
            return sub["fwi_danger"] / 92.0
        return sub["composite_exposure"]     # Asset Exposure

    fig = go.Figure()

    # -- Monitor cells (muted, small) --------------------------------------
    if len(monitor):
        fig.add_trace(go.Scattermap(
            lat=monitor["lat"],
            lon=monitor["lon"],
            mode="markers",
            marker=dict(
                size=11,
                color=_cvals(monitor),
                colorscale=[[0, "#1a2540"], [0.5, "#2d3f5e"], [1, "#3d5278"]],
                cmin=0, cmax=1,
                opacity=0.60,
            ),
            hovertext=monitor.apply(
                lambda r: (
                    f"<b>{r['cell_id']}: {r['sector_name']}</b><br>"
                    f"Rank #{r['rank']} — MONITOR ONLY<br>"
                    f"ML Prob: {r['ml_prob']:.1%} | FWI: {r['fwi_danger']:.0f}<br>"
                    f"Pop Exp: {r['pop_exposure']:.1%} | Agri Exp: {r['agri_exposure']:.1%}<br>"
                    f"Priority Score: {r['priority_score']:.4f}"
                ), axis=1,
            ),
            hoverinfo="text",
            name="Monitor Only",
            showlegend=True,
        ))

    # -- Dispatch cells (vivid, rank-labelled) ----------------------------
    if len(dispatched):
        fig.add_trace(go.Scattermap(
            lat=dispatched["lat"],
            lon=dispatched["lon"],
            mode="markers+text",
            marker=dict(
                size=dispatched["rank"].apply(
                    lambda r: max(30 - r * 0.9, 18)
                ).tolist(),
                color=_cvals(dispatched),
                colorscale="YlOrRd",
                cmin=0, cmax=1,
                opacity=0.93,
            ),
            text=dispatched["rank"].apply(lambda r: f"#{r}"),
            textfont=dict(size=9, color="white", family="Inter, sans-serif"),
            textposition="middle center",
            hovertext=dispatched.apply(
                lambda r: (
                    f"<b>{r['cell_id']}: {r['sector_name']}</b><br>"
                    f"<b>Rank #{r['rank']} — DISPATCH RECOMMENDED</b><br>"
                    f"ML Prob: {r['ml_prob']:.1%} | Risk: {r['risk_tier']}<br>"
                    f"FWI: {r['fwi_danger']:.0f} | Priority: {r['priority_score']:.4f}<br>"
                    f"Pop Exp: {r['pop_exposure']:.1%} | Agri Exp: {r['agri_exposure']:.1%}<br>"
                    f"Action: {r['suggested_action']}"
                ), axis=1,
            ),
            hoverinfo="text",
            name="Dispatch Recommended",
            showlegend=True,
        ))

    # -- NASA FIRMS overlay -----------------------------------------------
    if show_ground_truth:
        fires = df[df["ground_truth_fire"]]
        if len(fires):
            fig.add_trace(go.Scattermap(
                lat=fires["lat"],
                lon=fires["lon"],
                mode="markers",
                marker=dict(size=16, color="#ff2d2d", symbol="star", opacity=0.88),
                hovertext=fires.apply(
                    lambda r: f"<b>NASA FIRMS — Active Fire</b><br>{r['cell_id']}: {r['sector_name']}",
                    axis=1,
                ),
                hoverinfo="text",
                name="NASA FIRMS Fire",
                showlegend=True,
            ))

    # -- Layout -----------------------------------------------------------
    fig.update_layout(
        map=dict(
            style=MAP_STYLE,
            center=dict(lat=MAP_LAT, lon=MAP_LON),
            zoom=MAP_ZOOM,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=530,
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(family="Inter, sans-serif", color="#f1f5f9"),
        legend=dict(
            bgcolor="rgba(8,15,31,0.90)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=12, color="#f1f5f9"),
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#334155",
            font=dict(size=12, color="#f1f5f9", family="Inter, sans-serif"),
        ),
    )
    return fig


# ===========================================================================
# TAB 1 — Operational Dispatch & Map
# ===========================================================================
with tab1:
    st.markdown(
        '<div class="section-hdr">Core Question: Which locations should we monitor first this week, and why?</div>',
        unsafe_allow_html=True,
    )

    # ── View toggle & ground truth toggle ────────────────────────────────────
    col_v, col_g = st.columns([3, 1])
    with col_v:
        view_mode = st.radio(
            "View",
            ["FireGuard Dispatch Priority (Top-K)",
             "Supervised ML Probability",
             "Copernicus FWI Baseline",
             "Asset Exposure"],
            horizontal=True,
            label_visibility="collapsed",
        )
    with col_g:
        show_gt = st.checkbox("🛰️ NASA FIRMS Ground Truth", value=False)

    # ── Map ──────────────────────────────────────────────────────────────────
    st.plotly_chart(
        build_map(df, view_mode, k, show_ground_truth=show_gt),
        use_container_width=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": "fireguard_dispatch_map"},
        },
    )

    # ── Cell detail selector ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Operational Detail Card — Select a Cell</div>',
                unsafe_allow_html=True)

    top_opts  = [f"#{r['rank']} — {r['cell_id']}: {r['sector_name']}"
                 for _, r in df[df["rank"] <= k].sort_values("rank").iterrows()]
    rest_opts = [f"  {r['cell_id']}: {r['sector_name']}"
                 for _, r in df[df["rank"] >  k].sort_values("rank").iterrows()]

    sel = st.selectbox("Cell", top_opts + rest_opts,
                       index=0, label_visibility="collapsed")

    # Parse the selected cell_id from the label
    raw_part = sel.strip().split("—")[-1].strip().split(":")[0].strip()
    sel_id   = raw_part.lstrip("#").split(" ")[-1] if raw_part.startswith("#") else raw_part
    row      = df[df["cell_id"] == sel_id].iloc[0]

    is_disp    = row["rank"] <= k
    tier_color = TIER_HEX.get(row["risk_tier"], "#94a3b8")

    st.markdown(f"""
    <div class="detail-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
        <div>
          <h3>{row['cell_id']}: {row['sector_name']}</h3>
          <div style="font-size:11px;color:#475569;">{row['lat']:.4f}°S, {row['lon']:.4f}°E</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:10px;color:#475569;letter-spacing:.1em;text-transform:uppercase;">Priority Rank</div>
          <div style="font-size:30px;font-weight:800;color:#f59e0b;">#{row['rank']}</div>
        </div>
      </div>

      <div class="badge-row" style="margin-top:10px;">
        <span class="badge"
              style="background:rgba(239,68,68,.12);border-color:{tier_color};color:{tier_color};">
          ⚠ ML Risk: {row['risk_tier']}
        </span>
        <span class="badge {'badge-amber' if is_disp else 'badge-blue'}">
          {"🚁 DISPATCH RECOMMENDED" if is_disp else "👁 MONITOR ONLY"}
        </span>
        <span class="badge" style="background:rgba(139,92,246,.12);border-color:#8b5cf6;color:#a78bfa;">
          Exp Level: {row['exposure_level']}
        </span>
      </div>

      <div style="display:flex;flex-wrap:wrap;margin-top:4px;">
        <div class="metric-chip">
          <div class="val" style="color:#f59e0b;">{row['ml_prob']:.1%}</div>
          <div class="lbl">ML Probability</div>
        </div>
        <div class="metric-chip">
          <div class="val" style="color:#ef4444;">{row['fwi_danger']:.0f}</div>
          <div class="lbl">FWI Danger</div>
        </div>
        <div class="metric-chip">
          <div class="val" style="color:#3b82f6;">{row['composite_exposure']:.1%}</div>
          <div class="lbl">Composite Exp.</div>
        </div>
        <div class="metric-chip">
          <div class="val">{row['priority_score']:.4f}</div>
          <div class="lbl">Priority Score</div>
        </div>
        <div class="metric-chip">
          <div class="val" style="color:#8b5cf6;">{row['pop_exposure']:.1%}</div>
          <div class="lbl">Pop Exposure</div>
        </div>
        <div class="metric-chip">
          <div class="val" style="color:#14b8a6;">{row['agri_exposure']:.1%}</div>
          <div class="lbl">Agri Exposure</div>
        </div>
      </div>

      <div class="explain-box">
        <b style="color:#fbbf24;">🔍 Why is this ranked #{row['rank']}?</b><br>
        Calibrated GBDT indicates a <b>{row['ml_prob']:.1%} pre-ignition occurrence risk</b>
        driven by <i>{row['weather_driver']}</i>, intersecting with
        <i>{row['exposure_driver']}</i>.
      </div>
      <div class="directive-box">
        <b style="color:#60a5fa;">📋 Operational Directive:</b><br>
        {row['suggested_action']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top-K Dispatch Queue ─────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:24px;">Top-K Priority Dispatch Queue</div>',
                unsafe_allow_html=True)

    queue = (
        df[df["rank"] <= k]
        .sort_values("rank")
        [[  "rank", "cell_id", "sector_name", "risk_tier",
            "exposure_level", "priority_score", "ml_prob", "fwi_danger",
            "suggested_action"]]
        .copy()
    )
    queue.columns = [
        "Rank", "Cell ID", "Sector Name", "ML Risk Tier",
        "Exposure Level", "Priority Score", "ML Prob", "FWI",
        "Operational Action",
    ]
    queue["Priority Score"] = queue["Priority Score"].apply(lambda v: f"{v:.4f}")
    queue["ML Prob"]        = queue["ML Prob"].apply(lambda v: f"{v:.1%}")
    queue["FWI"]            = queue["FWI"].apply(lambda v: f"{v:.0f}")

    st.dataframe(queue, use_container_width=True, hide_index=True,
                 height=min(38 * (k + 1), 440))


# ===========================================================================
# TAB 2 — Decision Evaluation & Baselines
# ===========================================================================
with tab2:
    st.markdown(f"""
    <div style="margin-bottom:16px;">
      <div style="font-size:15px;font-weight:700;color:#f1f5f9;">
        Retrospective Decision Simulation — Held-Out Fire Season
      </div>
      <div style="font-size:12px;color:#64748b;margin-top:4px;">
        Comparing 4 allocation strategies at K =
        <b style="color:#f59e0b;">{k}</b> teams.
        Ground truth: NASA FIRMS satellite detections (held-out week).
      </div>
    </div>
    """, unsafe_allow_html=True)

    eval_df = load_evaluation_metrics(k)
    strategies = eval_df["strategy"].tolist()
    bar_colors = ["#475569", "#0ea5e9", "#8b5cf6", "#f59e0b"]

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    metrics_map = {
        "Precision@K":              "precision_at_k",
        "Recall@K":                 "recall_at_k",
        "Exposure-Weighted Recall": "ew_recall_at_k",
    }

    fig_bars = go.Figure()
    for metric_label, col in metrics_map.items():
        vals = eval_df[col].tolist()
        fig_bars.add_trace(go.Bar(
            name=metric_label,
            x=strategies,
            y=[round(v, 4) for v in vals],
            marker_color=bar_colors,
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))

    fig_bars.update_layout(
        barmode="group",
        paper_bgcolor="#020617",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color="#f1f5f9"),
        ),
        yaxis=dict(
            gridcolor="#1e293b",
            tickformat=".0%",
            range=[0, 1.18],
            tickfont=dict(size=11, color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(8,15,31,0.9)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#334155",
            font=dict(size=12, color="#f1f5f9"),
        ),
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    # ── Metric tiles ──────────────────────────────────────────────────────────
    fg  = eval_df[eval_df["strategy"] == "FireGuard ML"].iloc[0]
    fwi = eval_df[eval_df["strategy"] == "FWI Only"].iloc[0]

    prec_d = fg["precision_at_k"]   - fwi["precision_at_k"]
    rec_d  = fg["recall_at_k"]      - fwi["recall_at_k"]
    ewr_d  = fg["ew_recall_at_k"]   - fwi["ew_recall_at_k"]
    total_fires = int(df["ground_truth_fire"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision@K (FireGuard)",   f"{fg['precision_at_k']:.1%}",
              delta=f"+{prec_d:.1%} vs FWI Only")
    c2.metric("Recall@K (FireGuard)",      f"{fg['recall_at_k']:.1%}",
              delta=f"+{rec_d:.1%} vs FWI Only")
    c3.metric("Exp-Weighted Recall@K",     f"{fg['ew_recall_at_k']:.1%}",
              delta=f"+{ewr_d:.1%} vs FWI Only")
    c4.metric("Actual Fires (Held-Out)",   str(total_fires))
    c5.metric("Teams Deployed",            f"{k} / 60")

    # ── NASA FIRMS reveal ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr" style="margin-top:22px;">Satellite Ground Truth Validation</div>',
                unsafe_allow_html=True)
    if st.checkbox("🛰️ Reveal Held-Out Fire Locations (NASA FIRMS) — overlay on dispatch map", key="gt_tab2"):
        fig_gt = build_map(df, "FireGuard Dispatch Priority (Top-K)", k, show_ground_truth=True)
        st.plotly_chart(fig_gt, use_container_width=True)

        fire_rows = df[df["ground_truth_fire"]].copy()
        fire_rows["Dispatched?"] = fire_rows["dispatched"].map({True: "✅ YES", False: "❌ NO"})
        fire_tbl = fire_rows.rename(columns={
            "cell_id": "Cell ID", "sector_name": "Sector",
            "rank": "Priority Rank", "ml_prob": "ML Prob", "priority_score": "Priority Score",
        })[["Cell ID", "Sector", "Priority Rank", "ML Prob", "Priority Score", "Dispatched?"]].copy()
        fire_tbl["ML Prob"]        = fire_tbl["ML Prob"].apply(lambda v: f"{v:.1%}")
        fire_tbl["Priority Score"] = fire_tbl["Priority Score"].apply(lambda v: f"{v:.4f}")
        st.dataframe(fire_tbl, use_container_width=True, hide_index=True)

        dispatch_hits = fire_rows["dispatched"].sum()
        st.markdown(f"""
        <div class="info-block">
          <h4>🎯 Dispatch Coverage</h4>
          FireGuard correctly covered <b style="color:#22c55e;">{dispatch_hits}</b> of
          <b>{total_fires}</b> confirmed fire cells
          (Recall@K = <b style="color:#f59e0b;">{fg['recall_at_k']:.1%}</b>),
          a gain of <b style="color:#22c55e;">+{rec_d:.1%}</b> over FWI-Only dispatch.
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 3 — Model Card & Trust Architecture
# ===========================================================================
with tab3:
    st.markdown("""
    <div class="info-block">
      <h4>🧠 Model Architecture — Calibrated Gradient Boosted Decision Tree (GBDT)</h4>
      <p>FireGuard's core predictor is a <code>LightGBM</code> ensemble trained on 5 years of South Sumatra fire events
      (2019–2024) sourced from NASA FIRMS VIIRS / MODIS active fire archives, cross-referenced with:</p>
      <ul>
        <li><b>Copernicus GWIS Fire Weather Index (FWI)</b> — daily raster, 1 km resolution</li>
        <li><b>CHIRPS rainfall anomaly</b> — 10-day rolling deficit vs. 20-year climatological mean</li>
        <li><b>MODIS NDVI/EVI anomaly</b> — vegetation stress proxy at 250 m resolution</li>
        <li><b>WorldPop 2020</b> — 100 m gridded residential population density</li>
        <li><b>ESA WorldCover 2021</b> — cropland, oil palm, and peatland land-cover classes (10 m)</li>
        <li><b>Canal water level telemetry</b> — 14 BRGM peat restoration monitoring stations</li>
      </ul>
      <p>Raw GBDT outputs are post-hoc calibrated with <code>CalibratedClassifierCV</code>
      (isotonic regression, 5-fold) to produce confidence-aligned probabilities for cost-sensitive dispatch decisions.</p>
    </div>

    <div class="info-block">
      <h4>⏱ Temporal-Block Validation — Preventing Data Leakage</h4>
      <p>All evaluation uses <b>walk-forward temporal block cross-validation</b>: the model never sees
      validation-week data during training, mirroring the real operational constraint of forecasting next week's
      fire risk using only data available today.</p>
      <table>
        <thead><tr><th>Block</th><th>Train Period</th><th>Validation Window</th><th>Precision@10</th><th>Recall@10</th></tr></thead>
        <tbody>
          <tr><td>Block 1</td><td>2019–2021</td><td>Jul–Oct 2022</td><td>0.60</td><td>0.48</td></tr>
          <tr><td>Block 2</td><td>2019–2022</td><td>Jul–Oct 2023</td><td>0.70</td><td>0.55</td></tr>
          <tr><td>Block 3</td><td>2019–2023</td><td>Jul–Oct 2024</td><td>0.65</td><td>0.52</td></tr>
        </tbody>
      </table>
    </div>

    <div class="info-block">
      <h4>🌡 FWI Danger vs. Calibrated ML Probability — Why the Distinction Matters</h4>
      <p>The <b>Copernicus GWIS FWI</b> measures <i>atmospheric fire weather danger</i> — how hazardous
      current weather conditions are for fire spread. It does <b>not</b> account for whether flammable peat
      is present, whether drainage has lowered the water table, or whether human ignition pressure is elevated.</p>
      <p>FireGuard integrates <b>three independent risk channels</b> the FWI cannot capture alone:</p>
      <ol>
        <li><b>Weather danger</b> — FWI, VPD, rainfall deficit (Copernicus / ERA5)</li>
        <li><b>Fuel availability</b> — peat depth, drainage status, NDVI anomaly</li>
        <li><b>Ignition pressure</b> — land-clearing seasonality, concession boundary proximity</li>
      </ol>
      <p>This is why cells with moderate FWI but extreme fuel + ignition conditions can outrank cells
      with high FWI in pristine, undrained peatland — a nuance invisible to weather-only dispatch logic.</p>
    </div>

    <div class="info-block">
      <h4>⚖️ Priority Score Formula — Transparent & Adjustable Allocation</h4>
      <p>The dispatch score is intentionally transparent and tunable by decision-makers in real time
      via the sidebar sliders — no retraining required:</p>
      <p><code>composite_exposure = w_pop × pop_exposure + w_agri × agri_exposure</code></p>
      <p><code>priority_score = ml_prob × composite_exposure</code></p>
      <p>Shifting the Population vs. Agricultural weight updates the dispatch queue immediately,
      allowing commanders to reflect shifting operational priorities (humanitarian vs. economic protection).</p>
    </div>

    <div class="info-block">
      <h4>📡 Data Provenance & Operational Cadence</h4>
      <table>
        <thead><tr><th>Source</th><th>Variable</th><th>Resolution</th><th>Update Frequency</th></tr></thead>
        <tbody>
          <tr><td>Copernicus GWIS / EFFIS</td><td>Fire Weather Index</td><td>1 km</td><td>Daily</td></tr>
          <tr><td>NASA FIRMS VIIRS</td><td>Active Fire Detection</td><td>375 m</td><td>12-hourly</td></tr>
          <tr><td>ERA5 / CHIRPS</td><td>Rainfall, VPD, Temperature</td><td>25–50 km</td><td>Daily</td></tr>
          <tr><td>MODIS MOD13A1</td><td>NDVI / EVI Anomaly</td><td>500 m</td><td>16-day composite</td></tr>
          <tr><td>WorldPop 2020</td><td>Population Density</td><td>100 m</td><td>Annual</td></tr>
          <tr><td>ESA WorldCover 2021</td><td>Land Cover Classes</td><td>10 m</td><td>Annual</td></tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)
