"""A declared allocation policy, with fixed units and explicit tradeoffs."""
from pathlib import Path

import pandas as pd
import pytest


def cells():
    return pd.DataFrame({
        "cell_id": ["a", "b", "c"], "prediction_date": ["2024-01-01"] * 3,
        "risk_score": [.2, .3, .4], "y": [1, 0, 1],
        "population_2020": [80., 10., 10.], "annual_cropland_ha_2020": [0., 90., 10.],
        "closed_canopy_oil_palm_ha_2019": [0., 0., 100.],
        "fwi_baseline": [.2, .3, .4], "past_fire_7d": [.2, .3, .4],
        "past_fire_30d": [.2, .3, .4], "historical_frequency": [.2, .3, .4],
    })


def policy(pop=.5, crop=.25, palm=.25):
    return {"Population": pop, "Annual cropland": crop, "Closed-canopy oil palm": palm}


def totals():
    return {"population_2020": 100., "annual_cropland_ha_2020": 100.,
            "closed_canopy_oil_palm_ha_2019": 100.}


def test_combined_ranking_uses_fixed_shares_and_declared_weights():
    # Catches summing raw units, excluding an agricultural layer or changing the denominator after filtering.
    from fireguard.dashboard_data import rank_week
    frame = cells()
    ranked = rank_week(frame, "Combined exposure", 1, weights=policy(), totals=totals())
    assert ranked.cell_id.tolist() == ["c", "b", "a"]
    assert ranked.exposure_index.tolist() == pytest.approx([.325, .275, .4])
    assert ranked.priority_score.tolist() == pytest.approx([.13, .0825, .08])
    subset = rank_week(frame.iloc[:1], "Combined exposure", 1, weights=policy(), totals=totals())
    assert subset.priority_score.iloc[0] == pytest.approx(.08)
    frame.y = 1 - frame.y
    assert rank_week(frame, "Combined exposure", 1, weights=policy(), totals=totals()).cell_id.tolist() == ["c", "b", "a"]


def test_combined_policy_is_invariant_to_population_unit_scale():
    # Catches a numeric-scale advantage for population over hectares.
    from fireguard.dashboard_data import rank_week
    frame = cells()
    frame.population_2020 *= 1000
    reference = totals()
    reference["population_2020"] = 100000.
    result = rank_week(frame, "Combined exposure", 1, weights=policy(), totals=reference)
    assert result.priority_score.tolist() == pytest.approx([.13, .0825, .08])


def test_reference_totals_count_each_cell_once_and_reject_changing_maps():
    # Catches counting every historical week as additional people or farmland.
    from fireguard.dashboard_data import reference_totals
    later = cells().assign(prediction_date="2024-01-08")
    assert reference_totals(pd.concat([cells(), later])) == totals()
    later.loc[0, "population_2020"] = 79.
    with pytest.raises(ValueError, match="static"):
        reference_totals(pd.concat([cells(), later]))


def test_combined_rejects_invalid_policy_and_undefined_active_normalizers():
    # Catches hidden weight renormalization and zero/negative exposure substitutions.
    from fireguard.dashboard_data import rank_week
    for invalid in [policy(.6, .25, .25), policy(-.5, .75, .75), policy(float("nan"), .25, .25)]:
        with pytest.raises(ValueError, match="weights"):
            rank_week(cells(), "Combined exposure", 1, weights=invalid, totals=totals())
    with pytest.raises(ValueError, match="reference totals"):
        rank_week(cells(), "Combined exposure", 1, weights=policy())
    zero = totals()
    zero["closed_canopy_oil_palm_ha_2019"] = 0.
    frame = cells().assign(closed_canopy_oil_palm_ha_2019=0.)
    with pytest.raises(ValueError, match="positive"):
        rank_week(frame, "Combined exposure", 1, weights=policy(), totals=zero)
    assert rank_week(frame, "Combined exposure", 1, weights=policy(1, 0, 0), totals=zero).cell_id.iloc[0] == "a"


def test_combined_evaluation_keeps_separate_capture_and_pools_index_totals():
    # Catches re-ranking separately for each reported exposure and averaging weekly ratios.
    from fireguard.dashboard_data import evaluate
    later = cells().assign(prediction_date="2024-01-08", y=[0, 1, 0], risk_score=[1., 0., 0.])
    result = evaluate(pd.concat([cells(), later]), "Combined exposure", 1, weights=policy(), totals=totals())
    row = result.set_index("method").loc["FireGuard × exposure"]
    assert row.true_positives == 1
    assert row.precision == .5
    assert row.recall == pytest.approx(1 / 3)
    assert row.exposure_weighted_recall == pytest.approx(.325)
    assert row.population_capture == pytest.approx(.1)
    assert row.cropland_capture == pytest.approx(.1)
    assert row.oil_palm_capture == 1.
    assert result.selected_cells.nunique() == 1


def test_policy_comparison_changes_cells_and_reports_original_units():
    # Catches a disconnected sensitivity table or raw exposure sums being normalized.
    from fireguard.dashboard_data import policy_comparison
    result = policy_comparison(cells(), 1, weights=policy(), totals=totals()).set_index("Policy")
    assert result.loc["Current settings", "People in selected cells"] == 10.
    assert result.loc["Current settings", "Oil palm ha"] == 100.
    assert result.loc["People focused", "People in selected cells"] == 80.
    assert result.loc["People focused", "Cells changed"] == 1
    assert result.loc["Balanced", "Cells changed"] == 0


def test_combined_streamlit_sliders_update_queue_and_keep_risk_scores():
    # Catches weights omitted from cache keys, a stale queue, or changes to prediction scores.
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    app.selectbox(key="exposure").select("Combined exposure").run()
    assert not app.exception
    def slider_weights():
        return tuple(app.slider(key=key).value for key in
                     ("population_weight", "cropland_weight", "oil_palm_weight"))

    assert slider_weights() == (50, 25, 25)
    app.slider(key="population_weight").set_value(60).run()
    assert slider_weights() == (60, 20, 20)
    app.slider(key="cropland_weight").set_value(45).run()
    assert slider_weights() == (41, 45, 14)
    app.slider(key="oil_palm_weight").set_value(70).run()
    assert slider_weights() == (14, 16, 70)
    assert not app.exception
    initial = app.dataframe[0].value
    assert len(initial) == 50
    assert (initial["Risk × exposure"] > 0).all()
    old_ids = initial.Cell.tolist()
    old_scores = initial.set_index("Cell")["Risk score"]
    baseline = app.dataframe[1].value
    initial_capture = baseline.loc[baseline.Strategy == "FireGuard × exposure", "Population capture"].iloc[0]
    app.slider(key="population_weight").set_value(100).run()
    assert slider_weights() == (100, 0, 0)
    assert not app.exception
    revised = app.dataframe[0].value
    assert revised.Cell.tolist() != old_ids
    shared = old_scores.index.intersection(revised.Cell)
    assert revised.set_index("Cell").loc[shared, "Risk score"].tolist() == old_scores.loc[shared].tolist()
    new_baseline = app.dataframe[1].value
    assert new_baseline.loc[new_baseline.Strategy == "FireGuard × exposure", "Population capture"].iloc[0] != initial_capture
    app.slider(key="population_weight").set_value(0).run()
    assert slider_weights() == (0, 50, 50)
    app.slider(key="cropland_weight").set_value(100).run()
    assert slider_weights() == (0, 100, 0)
    app.selectbox(key="exposure").select("Population").run()
    app.selectbox(key="exposure").select("Combined exposure").run()
    assert slider_weights() == (0, 100, 0)
    assert not app.exception
    assert len(app.dataframe[0].value) == 50
