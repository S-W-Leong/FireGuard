"""FireGuard historical dashboard, adapted from origin/Dashboard-1.0.

The teammate's dark styling and three-tab workflow are retained. All scores,
exposures, rankings and evaluations now come from verified project artifacts.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_fireguard_data, load_grid, load_manifest
from fireguard.dashboard_data import EXPOSURE_MODES, evaluate, rank_week

st.set_page_config(page_title="FireGuard — Preventive Monitoring", page_icon="🔥",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
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
#MainMenu, footer { visibility: hidden !important; }
header { background: transparent !important; }
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
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def assets(bundle_version):
    return load_fireguard_data(), load_grid(), load_manifest()


def pct(value):
    return "Undefined" if pd.isna(value) else f"{value:.1%}"


def map_figure(frame, grid, view, exposure_column, k, reveal=False):
    """Clipped decision cells, with markers at representative cell coordinates."""
    column, color_label = {
        "Priority": ("priority_score", "Risk × exposure"),
        "Model risk": ("risk_score", "Risk score"),
        "FWI": ("fwi_lag1", "Previous-day FWI"),
        "Exposure": (exposure_column, "Exposure"),
    }[view]
    fig = go.Figure()
    excluded = [f for f in grid["features"] if f["properties"]["cell_id"] not in set(frame.cell_id)]
    if excluded:
        fig.add_trace(go.Choroplethmap(
            geojson={"type": "FeatureCollection", "features": excluded},
            featureidkey="properties.cell_id", locations=[f["properties"]["cell_id"] for f in excluded],
            z=[0] * len(excluded), colorscale=[[0, "#334155"], [1, "#334155"]], showscale=False,
            marker_line_width=.4, marker_line_color="#64748b", marker_opacity=.7,
            hovertemplate="%{location}<br>Excluded: population or landscape quality<extra></extra>"))
    fig.add_trace(go.Choroplethmap(
        geojson=grid, featureidkey="properties.cell_id", locations=frame.cell_id,
        z=frame[column], zmin=0, zmax=max(float(frame[column].max()), 1e-9),
        colorscale=[[0, "#122c42"], [.35, "#26627a"], [.65, "#c99932"], [1, "#f59e0b"]],
        marker_line_width=.25, marker_line_color="#64748b", marker_opacity=.75,
        customdata=frame[["rank", "risk_score", "fwi_lag1", exposure_column, "priority_score"]].to_numpy(),
        colorbar={"title": color_label, "thickness": 12, "len": .7, "tickfont": {"size": 10}},
        hovertemplate="%{location}<br>Priority rank: %{customdata[0]:.0f}<br>Risk score: %{customdata[1]:.4f}"
            "<br>Previous-day FWI: %{customdata[2]:.2f}<br>Exposure: %{customdata[3]:,.1f}"
            "<br>Risk × exposure: %{customdata[4]:,.2f}<extra></extra>"))
    selected = frame.loc[frame.selected]
    fig.add_trace(go.Scattermap(lat=selected.latitude, lon=selected.longitude,
        mode="markers", marker={"size": 9, "color": "#fbbf24"}, name=f"Selected {min(k, len(frame))}",
        customdata=selected[["cell_id", "rank"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>Selected · rank %{customdata[1]}<extra></extra>"))
    if reveal:
        positives = frame.loc[frame.y == 1]
        fig.add_trace(go.Scattermap(lat=positives.latitude, lon=positives.longitude,
            mode="markers", marker={"size": 13, "color": "#ef4444", "opacity": .85},
            name="Cells with subsequent detections", text=positives.cell_id,
            hovertemplate="%{text}<br>At least one qualifying detection in the outcome week<extra></extra>"))
    fig.update_layout(map={"style": "carto-darkmatter", "center": {"lat": -3.2, "lon": 104.3}, "zoom": 6.6},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=500,
        paper_bgcolor="#020617", font={"color": "#f1f5f9", "family": "Inter"},
        legend={"orientation": "h", "y": -.02}, uirevision="south-sumatra")
    return fig


try:
    from data_loader import BUNDLE
    bundle_version = (BUNDLE / "manifest.json").stat().st_mtime_ns
    data, grid, manifest = assets(bundle_version)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load the verified dashboard bundle: {exc}")
    st.stop()


@st.cache_data(show_spinner=False)
def year_evaluation(year, exposure_mode, budget, bundle_version):
    frame, _, _ = assets(bundle_version)
    return evaluate(frame.loc[frame.prediction_date.str.startswith(str(year))], exposure_mode, budget)


with st.sidebar:
    st.markdown("## 🔥 FireGuard")
    st.caption("Predict Early. Prioritize Prevention.")
    st.markdown("### Monitoring scenario")
    year = st.selectbox("Historical year", manifest["years"], index=len(manifest["years"]) - 1, key="year")
    dates = sorted(data.loc[data.prediction_date.str.startswith(str(year)), "prediction_date"].unique())
    default = f"{year}-09-02" if f"{year}-09-02" in dates else dates[-1]
    if "week" in st.session_state and st.session_state.week not in dates:
        st.session_state.week = default
    week = st.selectbox("Week starting (Asia/Jakarta)", dates, index=dates.index(default), key="week")
    exposure_mode = st.selectbox("Exposure to prioritize", list(EXPOSURE_MODES), key="exposure")
    k = st.slider("Available monitoring teams (K)", 1, 200, 50, key="budget",
                  help="One equal-cost cell inspection per team during the selected week. Travel routing is not modeled.")
    exposure_column, unit, exposure_year = EXPOSURE_MODES[exposure_mode]
    st.markdown(f"""<div class="stat-card"><div class="stat-label">Eligible decision cells</div>
        <div class="stat-value">{manifest['cohort_cells']:,} / {manifest['grid_cells']:,}</div></div>
        <div class="stat-card"><div class="stat-label">This week's monitoring budget</div>
        <div class="stat-value amber">{k} cells · {k / manifest['cohort_cells']:.1%}</div></div>""", unsafe_allow_html=True)
    st.markdown("**Priority = risk score × exposure**")
    st.caption(f"{exposure_mode}: {unit}, map year {exposure_year}. Units remain separate. Ties use cell ID.")
    st.info("Historical demonstration. The model uses archived inputs; live data availability has not been verified.")

ranked = rank_week(data.loc[data.prediction_date == week], exposure_mode, k)
selected = ranked.loc[ranked.selected]
outcome_end = ranked.outcome_end_date.iloc[0]
st.markdown("# FireGuard")
st.markdown("Preventive monitoring priorities · **South Sumatra, Indonesia**")
st.caption(f"HISTORICAL · {week} to {outcome_end} · {manifest['model_name']} · "
           f"{'Exploratory temporal evaluation' if year == 2024 else 'Development and calibration period'}")
dispatch_tab, evaluation_tab, trust_tab = st.tabs([
    "📍 Monitoring priorities & map", "📊 Evaluation & baselines", "🛡 Model & data"])

with dispatch_tab:
    st.markdown("### Where should the teams monitor?")
    left, right = st.columns([3, 1])
    with left:
        view = st.radio("Map layer", ["Priority", "Model risk", "FWI", "Exposure"], horizontal=True, key="map_layer")
        st.plotly_chart(map_figure(ranked, grid, view, exposure_column, k), width="stretch", key="dispatch_map")
        st.caption("Clipped ~10 km decision cells. Gold markers show the selected cells; grey cells are excluded. "
                   "Colors show the active layer in its own units. Markers indicate cell coordinates.")
    with right:
        st.markdown("#### Selected portfolio")
        st.metric("Cells selected", f"{len(selected):,}")
        st.metric(f"{exposure_mode} in selected cells ({unit})", f"{selected[exposure_column].sum():,.0f}")
        st.caption("Exposure within cell boundaries, not estimated damage or people affected.")
        st.markdown("**Recommended next step**")
        st.write("Prioritize preventive inspections in these cells and verify local conditions before allocating field resources.")
        st.caption("Equal inspection costs are assumed. Access, travel time and intervention effectiveness are not modeled.")

    st.markdown(f"### This week's top {k} priorities")
    columns = {"rank": "Rank", "cell_id": "Cell", "risk_score": "Risk score",
               "priority_score": "Risk × exposure", "fwi_lag1": "Previous-day FWI",
               "population_2020": "Population (2020)", "annual_cropland_ha_2020": "Annual cropland ha (2020)",
               "closed_canopy_oil_palm_ha_2019": "Oil palm ha (2019)"}
    queue = selected[list(columns)].rename(columns=columns)
    st.dataframe(queue, hide_index=True, width="stretch", height=330,
        column_config={"Risk score": st.column_config.NumberColumn(format="%.4f"),
                       "Risk × exposure": st.column_config.NumberColumn(format="%.2f"),
                       "Previous-day FWI": st.column_config.NumberColumn(format="%.2f")})
    export = selected[["prediction_date", "outcome_end_date", *columns,
                       "latitude", "longitude", "feature_end_date", "fwi_source_date"]]
    st.download_button("Download this monitoring plan (CSV)", export.to_csv(index=False),
                       file_name=f"fireguard_{week}_{exposure_column}_K{k}.csv", mime="text/csv")

    st.markdown("### Why this cell?")
    options = ranked.cell_id.tolist()
    if "cell" in st.session_state and st.session_state.cell not in options:
        st.session_state.cell = options[0]
    cell_id = st.selectbox("Inspect a decision cell", options, key="cell",
        format_func=lambda cell: f"#{int(ranked.loc[ranked.cell_id == cell, 'rank'].iloc[0])} · {cell}")
    cell = ranked.loc[ranked.cell_id == cell_id].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Model risk score", f"{cell.risk_score:.4f}")
    b.metric(f"{exposure_mode} ({unit})", f"{cell[exposure_column]:,.1f}")
    c.metric("Priority rank", f"{int(cell['rank'])} / {len(ranked)}")
    status = "Selected" if cell.selected else "Outside this week's budget"
    st.write(f"**{status}.** Score {cell.risk_score:.4f} × {cell[exposure_column]:,.1f} {unit} "
             f"= **{cell.priority_score:,.2f}** priority units. Selection follows descending priority, then cell ID.")
    st.write(f"Previous-day FWI **{cell.fwi_lag1:.2f}**, prior 7-day detections **{int(cell.past_fire_detections_7d)}**, "
             f"prior 30-day detections **{int(cell.past_fire_detections_30d)}**. "
             "These are recorded inputs, not a causal explanation of an individual fire.")
    st.table(pd.DataFrame({"Recorded input": ["FWI (previous day)", "FWI (7-day mean)", "FWI (7-day maximum)",
                        "FWI (30-day mean)", "Population (2020)", "Annual cropland ha (2020)", "Oil palm ha (2019)"],
                          "Value": [f"{cell[name]:,.2f}" for name in ["fwi_lag1", "fwi_mean_7d", "fwi_max_7d",
                          "fwi_mean_30d", "population_2020", "annual_cropland_ha_2020", "closed_canopy_oil_palm_ha_2019"]]}))
    st.caption(f"Cell representative coordinate: {abs(cell.latitude):.4f}° S, {cell.longitude:.4f}° E. "
               f"Feature cutoff: {cell.feature_end_date}; FWI valid date: {cell.fwi_source_date}. "
               "Weather timestamps do not establish historical publication availability.")

with evaluation_tab:
    st.markdown("### Did the priority list capture subsequent detections?")
    st.write(f"Every strategy receives **{k} cells per week**, the same **949-cell cohort**, and the same "
             f"**{exposure_mode.lower()}** exposure measurements. Observed outcomes are used only for evaluation.")
    period = st.radio("Comparison period", ["Selected week", "All weeks in selected year"],
                      index=1, horizontal=True, key="comparison_period")
    metrics = (year_evaluation(year, exposure_mode, k, bundle_version) if period.startswith("All")
               else evaluate(ranked, exposure_mode, k))
    results = metrics.set_index("method")
    ours = results.loc["FireGuard × exposure"]
    fwi = results.loc["FWI × exposure"]
    a, b, c = st.columns(3)
    a.metric(f"Precision@{k}", pct(ours.precision))
    b.metric(f"Recall@{k}", pct(ours.recall))
    difference = None if pd.isna(ours.exposure_weighted_recall) or pd.isna(fwi.exposure_weighted_recall) else (
        ours.exposure_weighted_recall - fwi.exposure_weighted_recall) * 100
    c.metric("Exposure-weighted recall", pct(ours.exposure_weighted_recall),
             delta=None if difference is None else f"{difference:+.1f} pp vs FWI × exposure")
    chart = go.Figure()
    for column, name, color in [("precision", "Precision", "#3b82f6"), ("recall", "Recall", "#14b8a6"),
                                 ("exposure_weighted_recall", "Exposure-weighted recall", "#f59e0b")]:
        chart.add_trace(go.Bar(x=metrics["method"], y=metrics[column], name=name, marker_color=color,
                              hovertemplate="%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>"))
    chart.update_layout(barmode="group", height=390, margin={"l": 0, "r": 0, "t": 10, "b": 0},
                        paper_bgcolor="#020617", plot_bgcolor="#0f172a", font_color="#f1f5f9",
                        yaxis={"range": [0, 1], "tickformat": ".0%"}, legend={"orientation": "h", "y": 1.12})
    st.plotly_chart(chart, width="stretch", key="baseline_chart")
    display = metrics[["method", "weeks", "selected_cells", "true_positives", "precision", "recall", "exposure_weighted_recall"]].copy()
    for column in ["precision", "recall", "exposure_weighted_recall"]:
        display[column] = display[column].map(pct)
    st.dataframe(display.rename(columns={"method": "Strategy", "weeks": "Weeks", "selected_cells": "Selections",
        "true_positives": "Captured positive cell-weeks", "precision": "Precision", "recall": "Recall",
        "exposure_weighted_recall": "Exposure-weighted recall"}), hide_index=True, width="stretch")
    st.caption(f"{int(ours.positive_cells):,} positive cell-weeks in this comparison. Precision = positives selected / selections. "
        "Recall = positives selected / all positives. Exposure-weighted recall = exposure in selected positive cells / "
        "exposure in all positive cells. Yearly results pool numerators and denominators; undefined ratios stay undefined.")
    st.info("Exposure capture measures where detections occurred relative to mapped assets. It does not establish "
            "burned area, losses or effectiveness of preventive inspections. FireGuard may lose to some baselines, particularly for oil palm.")
    st.checkbox("Reveal subsequent detections for the selected week", key="reveal_outcomes")
    if st.session_state.reveal_outcomes:
        st.plotly_chart(map_figure(ranked, grid, "Priority", exposure_column, k, reveal=True), width="stretch", key="outcome_map")
        positive = ranked.loc[ranked.y == 1]
        st.caption(f"Red markers: {len(positive)} cells with a qualifying detection during {week}–{outcome_end}. "
                   "Markers use cell coordinates, not individual fire locations. Continuing fire and agricultural burning are included.")
        st.dataframe(positive[["cell_id", "rank", "selected", "risk_score", exposure_column]].rename(
            columns={"cell_id": "Cell", "rank": "Priority rank", "selected": "Selected", "risk_score": "Risk score"}),
            hide_index=True, width="stretch")

with trust_tab:
    st.markdown("### The actual model and its evidence")
    st.write(f"**{manifest['model_name']}**, trained on 2022. Selection and sigmoid calibration use 2023. "
             "The base model is not refitted on later years. The 2024 comparison is exploratory because prior project decisions examined it.")
    dev = manifest["model_metrics"]["2023"]
    temporal = manifest["model_metrics"]["2024"]
    a, b = st.columns(2)
    a.metric("2023 development average precision", f"{dev['average_precision']:.3f}")
    b.metric("2024 exploratory average precision", f"{temporal['average_precision']:.3f}")
    st.caption("Average precision summarizes the precision-recall ranking; it is not classification accuracy. "
               "Scores should be treated as relative risk because calibration shifts across years.")
    st.markdown("#### Prediction target")
    st.write("At least one qualifying VIIRS S-NPP vegetation detection in the next seven local calendar days. "
             "This is not a verified new ignition, damaging wildfire or burned-area target. No detection does not prove no fire.")
    st.markdown("#### Retained prediction features")
    st.table(pd.DataFrame({"Feature": manifest["model"]["feature_names"]}))
    st.write("FWI and detection counts are log-transformed; annual sine/cosine encode seasonality.")
    if "logistic" in manifest["model"]:
        st.write("The retained logistic model standardizes inputs using training data only. "
                 "Latitude and longitude position cells on the map and sample weather. Population and agricultural hectares "
                 "multiply the risk score for allocation. **They do not enter this occurrence model.** "
                 "Peat is retained as geographic context.")
    else:
        st.write("The retained tree model uses the named weather, history, calendar and landscape inputs listed above. "
                 "Coordinates are used for mapping and weather sampling. Population and agricultural hectares are allocation multipliers.")
    st.markdown("#### Data sources and dates")
    st.table(pd.DataFrame(manifest["sources"]).rename(columns={"dataset": "Dataset", "date": "Date", "measurement": "Measurement"}))
    st.markdown("#### Limits to carry into the pitch")
    for limit in manifest["limits"]:
        st.write(f"• {limit}")
    with st.expander("Reproducible handoff and prototype provenance"):
        st.write(f"Runtime package: `{manifest['model_package']}`. Bundle: {manifest['rows']:,} cell-weeks, "
                 f"{manifest['weeks']} weeks. Bundle file hashes are checked when loaded.")
        st.write("The teammate's Streamlit layout, visual style and three-tab flow came from "
                 f"`{manifest['prototype_branch']}` at `{manifest['prototype_commit']}`. "
                 "The mock loader, loss-averted estimates and placeholder model claims were replaced.")
        st.json({"model_sha256": manifest["selection"]["model_sha256"], "bundle_hashes": manifest["bundle_hashes"]})
