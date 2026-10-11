"""Dashboard boundaries: real scores, separate exposure units and fair budgets."""
from pathlib import Path

import pandas as pd
import pytest


def sample():
    return pd.DataFrame({
        "cell_id": ["b", "a", "c"] * 2,
        "prediction_date": ["2024-01-01"] * 3 + ["2024-01-08"] * 3,
        "y": [1, 0, 1, 0, 1, 0],
        "risk_score": [1., 1., 0., 0., 2., 1.],
        "population_2020": [10., 5., 20.] * 2,
        "annual_cropland_ha_2020": [0., 5., 100.] * 2,
        "closed_canopy_oil_palm_ha_2019": [100., 0., 1.] * 2,
        "fwi_baseline": [1., 1., 0., 0., 2., 1.],
        "past_fire_7d": [1., 1., 0., 0., 2., 1.],
        "past_fire_30d": [1., 1., 0., 0., 2., 1.],
        "historical_frequency": [1., 1., 0., 0., 2., 1.],
    })


def test_dispatch_uses_one_exposure_and_never_outcomes():
    # Catches mixing people with hectares, outcome-based ties, and global budgets.
    from fireguard.dashboard_data import rank_week
    frame = sample().iloc[:3].copy()
    assert rank_week(frame, "Population", 2).query("selected").cell_id.tolist() == ["b", "a"]
    assert rank_week(frame, "Annual cropland", 1).query("selected").cell_id.tolist() == ["a"]
    assert rank_week(frame, "Closed-canopy oil palm", 1).query("selected").cell_id.tolist() == ["b"]
    frame.y = 1 - frame.y
    frame.risk_score = 1.
    frame.population_2020 = 10.
    assert rank_week(frame, "Population", 1).query("selected").cell_id.tolist() == ["a"]
    assert rank_week(frame, "Population", 10).selected.sum() == 3
    with pytest.raises(ValueError, match="single week"):
        rank_week(sample(), "Population", 1)


def test_dashboard_evaluation_pools_capture_and_keeps_identical_budgets():
    # Catches averaging ratios, normalization of exposure and regenerating mock cells.
    from fireguard.dashboard_data import evaluate
    result = evaluate(sample(), "Population", 1).set_index("method")
    ours = result.loc["FireGuard × exposure"]
    assert ours.selected_cells == 2
    assert ours.true_positives == 1
    assert ours.precision == .5
    assert ours.recall == pytest.approx(1 / 3)
    assert ours.exposure_weighted_recall == pytest.approx(10 / 35)
    assert result.selected_cells.nunique() == 1
    frame = sample()
    frame.y = 0
    empty = evaluate(frame, "Population", 100)
    assert empty.selected_cells.tolist() == [6] * len(empty)
    assert empty.recall.isna().all()
    assert empty.exposure_weighted_recall.isna().all()


def test_portable_bundle_matches_saved_model_and_metrics():
    # Catches using the rejected environmental model or changing the comparison cohort.
    from data_loader import load_fireguard_data, load_manifest
    from fireguard.dashboard_data import evaluate
    frame = load_fireguard_data()
    manifest = load_manifest()
    assert frame.groupby("prediction_date").size().unique().tolist() == [949]
    assert frame.cell_id.nunique() == 949
    assert len(frame) == 98696
    assert manifest["model"]["feature_names"] == [
        "log1p_fwi_lag1", "log1p_fwi_mean_7d", "log1p_fwi_max_7d", "log1p_fwi_mean_30d",
        "log1p_past_fire_detections_7d", "log1p_past_fire_detections_30d", "annual_sin", "annual_cos"]
    assert (pd.to_datetime(frame.fwi_source_date) < pd.to_datetime(frame.prediction_date)).all()
    year = frame.loc[frame.prediction_date.str.startswith("2024")]
    result = evaluate(year, "Population", 50).set_index("method")
    assert result.loc["Model risk only", "true_positives"] == 471
    assert result.loc["FWI only", "true_positives"] == 149
    assert result.loc["FireGuard × exposure", "exposure_weighted_recall"] == pytest.approx(.47435350775590013, abs=1e-7)


def test_bundle_join_rejects_missing_cells_and_future_weather():
    # Catches silent row loss and prediction-time contamination in a new model handoff.
    from fireguard.dashboard_bundle import join_inputs
    scores = pd.DataFrame({"cell_id": ["a", "b"], "prediction_date": ["2024-01-01"] * 2,
                           "risk_score": [.1, .2]})
    predictors = pd.DataFrame({"cell_id": ["a", "b"], "prediction_date": ["2024-01-01"] * 2,
                              "feature_end_date": ["2023-12-31"] * 2,
                              "fwi_source_date": ["2023-12-31"] * 2})
    assert join_inputs(scores, predictors).risk_score.tolist() == [.1, .2]
    with pytest.raises(ValueError, match="missing"):
        join_inputs(scores, predictors.iloc[:1])
    predictors.loc[0, "fwi_source_date"] = "2024-01-01"
    with pytest.raises(ValueError, match="precede"):
        join_inputs(scores, predictors)


def test_streamlit_controls_change_real_dispatch_without_revealing_labels():
    # Catches disconnected UI controls and future labels leaking into the dispatch queue.
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=25).run()
    assert not app.exception
    queue = app.dataframe[0].value
    assert len(queue) == 50
    assert "y" not in queue.columns
    first = queue["Cell"].tolist()
    app.selectbox(key="exposure").select("Annual cropland")
    app.slider(key="budget").set_value(10).run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 10
    assert app.dataframe[0].value["Cell"].tolist() != first[:10]
    app.selectbox(key="week").select("2024-01-01").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 10
    app.checkbox(key="reveal_outcomes").check().run()
    assert not app.exception
    assert app.dataframe[-1].value["Cell"].size >= 1
