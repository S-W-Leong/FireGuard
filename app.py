"""FireGuard historical dashboard, adapted from origin/Dashboard-1.0.

The teammate's dark styling and three-tab workflow are retained. All scores,
exposures, rankings and evaluations now come from verified project artifacts.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_fireguard_data, load_grid, load_manifest
from fireguard.dashboard_data import (COMBINED_MODE, EXPOSURE_MODES, SHARE_COLUMNS,
    evaluate, policy_comparison, rank_week, reference_totals)

st.set_page_config(page_title="FireGuard — Preventive Monitoring", page_icon="🔥",
                   layout="wide", initial_sidebar_state="auto")
# Native theming handles appearance; this small layout override keeps the two
# built-in sidebar controls at the same viewport anchor. Streamlit has no
# public setting for their position. Keep the toolbar mounted for reopening.
st.html("""<style>
[data-testid="stSidebarCollapseButton"] {
    position: absolute;
    top: 1rem;
    left: 1rem;
    margin: 0;
    visibility: visible;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 4rem;
}
</style>""")


@st.cache_data(show_spinner=False, max_entries=2)
def assets(bundle_version):
    frame = load_fireguard_data()
    return frame, load_grid(), load_manifest(), reference_totals(frame)


def pct(value):
    return "Undefined" if pd.isna(value) else f"{value:.1%}"


def map_figure(frame, grid, view, exposure_column, k, reveal=False):
    """Clipped decision cells, with markers at representative cell coordinates."""
    combined = exposure_column == "exposure_index"
    exposure_label = "Combined exposure index" if combined else "Exposure"
    exposure_format = ".6f" if combined else ",.1f"
    priority_format = ".6f" if combined else ",.2f"
    column, color_label = {
        "Priority": ("priority_score", "Risk × exposure"),
        "Model risk": ("risk_score", "Risk score"),
        "FWI": ("fwi_lag1", "Previous-day FWI"),
        "Exposure": (exposure_column, exposure_label),
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
            "<br>Previous-day FWI: %{customdata[2]:.2f}<br>" + exposure_label + ": %{customdata[3]:" + exposure_format + "}"
            "<br>Risk × exposure: %{customdata[4]:" + priority_format + "}<extra></extra>"))
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
    data, grid, manifest, totals = assets(bundle_version)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load the verified dashboard bundle: {exc}")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=64)
def year_evaluation(year, exposure_mode, budget, bundle_version, weights):
    frame, _, _, reference = assets(bundle_version)
    return evaluate(frame.loc[frame.prediction_date.str.startswith(str(year))], exposure_mode, budget,
                    weights=weights, totals=reference)


with st.sidebar:
    st.markdown("## 🔥 FireGuard")
    st.caption("Historical monitoring planner")
    st.markdown("### Monitoring scenario")
    year = st.selectbox("Historical year", manifest["years"], index=len(manifest["years"]) - 1, key="year")
    dates = sorted(data.loc[data.prediction_date.str.startswith(str(year)), "prediction_date"].unique())
    default = f"{year}-09-02" if f"{year}-09-02" in dates else dates[-1]
    if "week" in st.session_state and st.session_state.week not in dates:
        st.session_state.week = default
    week = st.selectbox("Week starting (Asia/Jakarta)", dates, index=dates.index(default), key="week",
                        format_func=lambda date: pd.Timestamp(date).strftime("%d %b %Y"))
    exposure_mode = st.selectbox("Exposure to prioritize", [COMBINED_MODE, *EXPOSURE_MODES], key="exposure")
    combined = exposure_mode == COMBINED_MODE
    weights = None
    if combined:
        population_weight = st.slider("Population emphasis (%)", 0, 100, 50, 5, key="population_weight",
            help="The remaining emphasis goes to agriculture. This weights the index, not the share of teams.") / 100
        cropland_weight = st.slider("Cropland share of agriculture (%)", 0, 100, 50, 5, key="cropland_weight",
            disabled=population_weight == 1, help="The rest of the agricultural emphasis goes to closed-canopy oil palm.") / 100
        weights = {"Population": population_weight,
                   "Annual cropland": (1 - population_weight) * cropland_weight,
                   "Closed-canopy oil palm": (1 - population_weight) * (1 - cropland_weight)}
        st.caption(f"Population {weights['Population']:.1%} · Cropland {weights['Annual cropland']:.1%} · "
                   f"Oil palm {weights['Closed-canopy oil palm']:.1%}")
        st.caption("These are policy assumptions. The balanced starting point is 50% population and 50% agriculture.")
    k = st.slider("Available monitoring teams (K)", 1, 200, 50, key="budget",
                  help="One equal-cost cell inspection per team during the selected week. Travel routing is not modeled.")
    exposure_column, unit, exposure_year = (("exposure_index", "index", "2019/2020") if combined
                                          else EXPOSURE_MODES[exposure_mode])
    st.metric("Eligible decision cells", f"{manifest['cohort_cells']:,} / {manifest['grid_cells']:,}", border=True)
    st.caption(f"{k} teams cover {k / manifest['cohort_cells']:.1%} of eligible cells this week.")
    st.markdown("**Priority = risk score × exposure**")
    if combined:
        st.caption("Each criterion is its fraction of the fixed eligible-cell total, then multiplied by its weight. "
                   "People and hectares retain their own units. Ties use cell ID.")
    else:
        st.caption(f"{exposure_mode}: {unit}, map year {exposure_year}. Units remain separate. Ties use cell ID.")
    st.caption("Historical demonstration using archived inputs. Live data availability has not been verified.")

ranked = rank_week(data.loc[data.prediction_date == week], exposure_mode, k, weights=weights, totals=totals)
selected = ranked.loc[ranked.selected]
outcome_end = ranked.outcome_end_date.iloc[0]
st.title("FireGuard", anchor=False)
st.markdown("Preventive monitoring priorities · **South Sumatra, Indonesia**")
st.caption(f":orange-badge[Historical demonstration] · {pd.Timestamp(week):%d %b %Y} – {pd.Timestamp(outcome_end):%d %b %Y} · {manifest['model_name']} · "
           f"{'Exploratory temporal evaluation' if year == 2024 else 'Development and calibration period'}")
dispatch_tab, evaluation_tab, trust_tab = st.tabs([
    "Monitoring plan", "Evaluation", "Model & data"])

with dispatch_tab:
    st.subheader("Where should the teams monitor?", anchor=False)
    a, b = st.columns(2)
    a.metric("Cells selected", f"{len(selected):,}", border=True)
    if combined:
        b.metric("Selected share of combined index", f"{selected.exposure_index.sum():.1%}", border=True,
                 help="Sum of the selected cells' weighted exposure shares. All 949 cells sum to 100% before risk multiplication.")
        with st.container(horizontal=True):
            for name, label in [("Population", "People in selected cells"), ("Annual cropland", "Annual cropland ha"),
                                ("Closed-canopy oil palm", "Oil palm ha")]:
                st.metric(label, f"{selected[EXPOSURE_MODES[name][0]].sum():,.0f}", border=True, width="content")
        st.caption("Recorded exposure within selected cell boundaries. The two agricultural maps can overlap; "
                   "their hectares are separate measurements. Exposure does not estimate damage.")
    else:
        b.metric(f"Selected exposure ({unit})", f"{selected[exposure_column].sum():,.0f}", border=True,
                 help=f"{exposure_mode} within selected cell boundaries; map year {exposure_year}.")
        st.caption(f"{exposure_mode} within selected cell boundaries, not estimated damage or people affected.")
    view = st.segmented_control("Map layer", ["Priority", "Model risk", "FWI", "Exposure"],
                                default="Priority", required=True, key="map_layer", wrap=True)
    st.plotly_chart(map_figure(ranked, grid, view, exposure_column, k), width="stretch", key="dispatch_map",
                    alt=f"South Sumatra {view.lower()} map with {k} selected monitoring cells.")
    st.caption("Clipped ~10 km decision cells. Gold markers show the selected cells; grey cells are excluded. "
               "Colors show the active layer in its own units. Markers indicate cell coordinates.")
    with st.expander("Using this monitoring plan", icon=":material/info:"):
        st.markdown("**Recommended next step**")
        st.write("Prioritize preventive inspections in these cells and verify local conditions before allocating field resources.")
        st.caption("Equal inspection costs are assumed. Access, travel time and intervention effectiveness are not modeled.")

    columns = {"rank": "Rank", "cell_id": "Cell", "risk_score": "Risk score",
               "priority_score": "Risk × exposure", "fwi_lag1": "Previous-day FWI",
               "population_2020": "Population (2020)", "annual_cropland_ha_2020": "Annual cropland ha (2020)",
               "closed_canopy_oil_palm_ha_2019": "Oil palm ha (2019)"}
    queue = selected[list(columns)].rename(columns=columns)
    export = selected[["prediction_date", "outcome_end_date", *columns,
                       "latitude", "longitude", "feature_end_date", "fwi_source_date"]].copy()
    export["exposure_mode"] = exposure_mode
    suffix = ""
    if combined:
        export["exposure_index"] = selected.exposure_index
        for name, share_column in SHARE_COLUMNS.items():
            export[share_column] = selected[share_column]
            export[share_column + "_contribution"] = selected[share_column + "_contribution"]
            export[share_column + "_weight"] = weights[name]
            export[share_column + "_reference_total"] = totals[EXPOSURE_MODES[name][0]]
        suffix = "_weights_" + "_".join(f"{weights[name]:.3f}" for name in EXPOSURE_MODES)
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.subheader(f"This week's top {k} priorities", anchor=False, width="content")
        st.download_button("Download plan", export.to_csv(index=False),
                           file_name=f"fireguard_{week}_{exposure_column}_K{k}{suffix}.csv", mime="text/csv",
                           icon=":material/download:", key="download_plan", on_click="ignore")
    st.dataframe(queue, hide_index=True, width="stretch", height=330,
        column_config={"Risk score": st.column_config.NumberColumn(format="%.4f"),
                       "Risk × exposure": st.column_config.NumberColumn(format="%.6f" if combined else "%.2f"),
                       "Previous-day FWI": st.column_config.NumberColumn(format="%.2f"),
                       "Population (2020)": st.column_config.NumberColumn(format="%,.0f"),
                       "Annual cropland ha (2020)": st.column_config.NumberColumn(format="%,.1f"),
                       "Oil palm ha (2019)": st.column_config.NumberColumn(format="%,.1f")},
        alt=f"Top {k} monitoring priorities for the week starting {week}.")

    st.markdown("### Why this cell?")
    options = ranked.cell_id.tolist()
    if "cell" in st.session_state and st.session_state.cell not in options:
        st.session_state.cell = options[0]
    cell_id = st.selectbox("Inspect a decision cell", options, key="cell",
        format_func=lambda cell: f"#{int(ranked.loc[ranked.cell_id == cell, 'rank'].iloc[0])} · {cell}")
    cell = ranked.loc[ranked.cell_id == cell_id].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Model risk score", f"{cell.risk_score:.4f}", border=True)
    b.metric("Combined exposure index" if combined else f"Exposure ({unit})",
             f"{cell[exposure_column]:.6f}" if combined else f"{cell[exposure_column]:,.1f}", border=True)
    c.metric("Priority rank", f"{int(cell['rank'])} / {len(ranked)}", border=True)
    status = "Selected" if cell.selected else "Outside this week's budget"
    exposure_text = f"{cell[exposure_column]:.6f}" if combined else f"{cell[exposure_column]:,.1f} {unit}"
    priority_text = f"{cell.priority_score:.6f}" if combined else f"{cell.priority_score:,.2f}"
    st.write(f"**{status}.** Score {cell.risk_score:.4f} × {exposure_text} "
             f"= **{priority_text}** priority units. Selection follows descending priority, then cell ID.")
    if combined:
        st.table(pd.DataFrame([{"Criterion": name, "Cell share of eligible total": f"{cell[share]:.3%}",
            "Eligible-cell reference total": f"{totals[EXPOSURE_MODES[name][0]]:,.0f} {EXPOSURE_MODES[name][1]}",
            "Policy weight": f"{weights[name]:.1%}", "Weighted contribution": f"{cell[share + '_contribution']:.6f}"}
            for name, share in SHARE_COLUMNS.items()]), alt="Population, cropland and oil palm contributions to this cell's exposure index.")
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
    period = st.segmented_control("Comparison period", ["Selected week", "All weeks in selected year"],
                      default="All weeks in selected year", required=True, key="comparison_period", wrap=True)
    metrics = (year_evaluation(year, exposure_mode, k, bundle_version, weights) if period.startswith("All")
               else evaluate(ranked, exposure_mode, k, weights=weights, totals=totals))
    results = metrics.set_index("method")
    ours = results.loc["FireGuard × exposure"]
    fwi = results.loc["FWI × exposure"]
    a, b, c = st.columns(3)
    a.metric(f"Precision@{k}", pct(ours.precision), border=True, height="stretch")
    b.metric(f"Recall@{k}", pct(ours.recall), border=True, height="stretch")
    difference = None if pd.isna(ours.exposure_weighted_recall) or pd.isna(fwi.exposure_weighted_recall) else (
        ours.exposure_weighted_recall - fwi.exposure_weighted_recall) * 100
    c.metric("Combined-index recall" if combined else "Exposure-weighted recall", pct(ours.exposure_weighted_recall),
             delta=None if difference is None else f"{difference:+.1f} pp", border=True, height="stretch",
             delta_description="vs FWI × exposure")
    chart = go.Figure()
    chart_metrics = ([("population_capture", "Population capture", "#3b82f6"),
                      ("cropland_capture", "Cropland capture", "#14b8a6"),
                      ("oil_palm_capture", "Oil palm capture", "#f59e0b")] if combined else
                     [("precision", "Precision", "#3b82f6"), ("recall", "Recall", "#14b8a6"),
                      ("exposure_weighted_recall", "Exposure-weighted recall", "#f59e0b")])
    for column, name, color in chart_metrics:
        chart.add_trace(go.Bar(y=metrics["method"], x=metrics[column], orientation="h", name=name, marker_color=color,
                              hovertemplate="%{y}<br>%{x:.1%}<extra>%{fullData.name}</extra>"))
    chart.update_layout(barmode="group", height=520, margin={"l": 0, "r": 10, "t": 10, "b": 70},
                        paper_bgcolor="#020617", plot_bgcolor="#0f172a", font_color="#f1f5f9",
                        xaxis={"range": [0, 1], "tickformat": ".0%", "dtick": .25},
                        yaxis={"autorange": "reversed", "automargin": True, "tickfont": {"size": 10},
                               "tickvals": metrics["method"].tolist(),
                               "ticktext": [method.replace(" × ", "<br>× ").replace(" only", "<br>only")
                                            .replace("Historical frequency", "Historical<br>frequency")
                                            for method in metrics["method"]]},
                        legend={"orientation": "h", "y": -.15, "x": 0, "font": {"size": 10}})
    st.plotly_chart(chart, width="stretch", key="baseline_chart",
                    alt=("Population, cropland and oil palm capture for seven equal-budget strategies." if combined else
                         "Precision, recall, and exposure-weighted recall for seven equal-budget monitoring strategies."))
    capture_columns = ["population_capture", "cropland_capture", "oil_palm_capture"] if combined else []
    display = metrics[["method", "weeks", "selected_cells", "true_positives", "precision", "recall",
                       "exposure_weighted_recall", *capture_columns]].copy()
    for column in ["precision", "recall", "exposure_weighted_recall", *capture_columns]:
        display[column] = display[column].map(pct)
    st.dataframe(display.rename(columns={"method": "Strategy", "weeks": "Weeks", "selected_cells": "Selections",
        "true_positives": "Captured positive cell-weeks", "precision": "Precision", "recall": "Recall",
        "exposure_weighted_recall": "Combined-index recall" if combined else "Exposure-weighted recall",
        "population_capture": "Population capture", "cropland_capture": "Cropland capture", "oil_palm_capture": "Oil palm capture"}),
        hide_index=True, width="stretch", alt="Equal-budget baseline outcomes and separate exposure capture measurements.")
    st.caption(f"{int(ours.positive_cells):,} positive cell-weeks in this comparison. Precision = positives selected / selections. "
        "Recall = positives selected / all positives. Exposure-weighted recall = exposure in selected positive cells / "
        "exposure in all positive cells. Yearly results pool numerators and denominators; undefined ratios stay undefined.")
    st.info("Exposure capture measures where detections occurred relative to mapped assets. It does not establish "
            "burned area, losses or effectiveness of preventive inspections. FireGuard may lose to some baselines, particularly for oil palm.")
    if combined:
        st.caption("Combined-index recall uses the same fixed normalizers and policy weights as allocation. "
                   "Each separate capture ratio uses the original people or hectare measurements on that same selected plan.")
        with st.expander("Weight tradeoffs for the selected week", expanded=True, icon=":material/balance:"):
            sensitivity = policy_comparison(data.loc[data.prediction_date == week], k, weights=weights, totals=totals)
            st.dataframe(sensitivity, hide_index=True, width="stretch",
                column_config={column: st.column_config.NumberColumn(format="percent")
                               for column in ["Population weight", "Cropland weight", "Oil palm weight"]},
                alt="Selected-week portfolios under current, people-focused, balanced and agriculture-focused weighting policies.")
            st.caption(f"Week starting {week}, K={k}. 'Cells changed' counts replacement cells relative to the current plan. "
                       "These are exposure totals within selected cells; subsequent detections are not used to choose weights.")
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
    a.metric("2023 development average precision", f"{dev['average_precision']:.3f}", border=True)
    b.metric("2024 exploratory average precision", f"{temporal['average_precision']:.3f}", border=True)
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
