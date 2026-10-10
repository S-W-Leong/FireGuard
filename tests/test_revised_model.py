import numpy as np
import pandas as pd
import pytest


def fixture_rows():
    from fireguard.revised_model import LOG_COLUMNS, FRACTION_COLUMNS
    data = {c: [0.0, 3.0] for c in LOG_COLUMNS}
    data.update({c: [0.0, 0.1] for c in FRACTION_COLUMNS})
    data.update(ffmc_lag1=[70.0, 80.0], prediction_date=["2022-01-03", "2022-01-10"])
    return pd.DataFrame(data)


def test_feature_values_ignore_coordinates_population_and_outcomes():
    # Break caught: using outcome or exposure/coordinate columns in fitting.
    from fireguard.revised_model import features
    df = fixture_rows()
    first = features(df)
    df["latitude"] = [100, -100]
    df["longitude"] = [999, -999]
    df["population_2020"] = [0, 1e12]
    df["fire_positive_next_7d"] = [1, 0]
    pd.testing.assert_frame_equal(first, features(df))
    assert first.iloc[1]["log1p_dc_lag1"] == pytest.approx(np.log(4))
    assert first.iloc[0]["ffmc_lag1"] == 70


@pytest.mark.parametrize("column,value", [("dc_lag1", -1), ("ffmc_lag1", np.nan), ("peat_fraction_peatmap2017", 1.1)])
def test_bad_predictors_are_rejected(column, value):
    from fireguard.revised_model import features
    df = fixture_rows()
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        features(df)


def test_split_purges_outcome_crossing_next_year():
    # Break caught: treating a Jan-1 outcome as a completed training label.
    from fireguard.revised_model import split_year
    df = pd.DataFrame({"prediction_date": pd.to_datetime(["2022-12-19", "2022-12-26", "2023-01-02"]),
                       "outcome_end_date": pd.to_datetime(["2022-12-25", "2023-01-01", "2023-01-08"])})
    assert split_year(df, 2022).index.tolist() == [0]
    assert split_year(df, 2023).index.tolist() == [2]


def test_cohort_never_imputes_population_and_is_fixed_each_week():
    from fireguard.revised_model import cohort
    df = pd.DataFrame({"cell_id": ["a", "b", "c"] * 2,
        "prediction_date": ["2022-01-03"] * 3 + ["2022-01-10"] * 3,
        "population_2020": [10, np.nan, 20] * 2,
        "landscape_area_quality_ok": [True, True, False] * 2})
    selected, excluded = cohort(df)
    assert selected.cell_id.tolist() == ["a", "a"]
    assert excluded == ["b", "c"]
    with pytest.raises(ValueError, match="cohort"):
        cohort(df.drop(index=3))


def test_future_weather_and_duplicate_keys_are_rejected():
    from fireguard.revised_model import validate_time
    df = pd.DataFrame({"cell_id": ["a"], "prediction_date": ["2022-01-03"],
        "outcome_end_date": ["2022-01-09"], "feature_end_date": ["2022-01-02"],
        "fwi_source_date": ["2022-01-02"], "components_source_date": ["2022-01-03"],
        "fire_positive_next_7d": [0]})
    with pytest.raises(ValueError, match="precede"):
        validate_time(df)
    df.components_source_date = "2022-01-02"
    with pytest.raises(ValueError, match="duplicate"):
        validate_time(pd.concat([df, df]))


def test_portable_score_matches_fitted_model_on_unseen_rows():
    # Break caught: export loses scaler or calibrator/intercept sign.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from fireguard.revised_model import export_model, portable_score
    train = pd.DataFrame({"x": [-2., -1., 0., 1., 2., 3.], "z": [0., 1., 0., 1., 0., 1.]})
    y = [0, 0, 0, 1, 1, 1]
    scaler = StandardScaler().fit(train)
    fit = LogisticRegression(C=1).fit(scaler.transform(train), y)
    calibration = LogisticRegression(C=1).fit(fit.decision_function(scaler.transform(train)).reshape(-1, 1), y)
    unseen = pd.DataFrame({"x": [-9., 0.2, 8.], "z": [0.3, 0.7, 0.9]})
    exported = export_model(scaler, fit, calibration, list(train.columns))
    actual = portable_score(unseen, exported)
    expected = calibration.predict_proba(fit.decision_function(scaler.transform(unseen)).reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_training_scaler_uses_2022_only_and_rejects_wrong_year():
    # Break caught: fitting scaler/model on validation or a random year split.
    from fireguard.revised_model import fit_model
    train = fixture_rows()
    train["fire_positive_next_7d"] = [0, 1]
    validation = train.copy()
    validation.prediction_date = ["2023-01-02", "2023-01-09"]
    validation.fwi_lag1 = [999, 1999]
    scaler, model, calibrator, scores = fit_model(train, validation)
    assert scaler.mean_[0] == pytest.approx(0.6931471805599453)
    assert len(scores) == 4
    assert model.C == 0.01  # All four perfectly rank this two-row validation.
    with pytest.raises(ValueError, match="year"):
        fit_model(validation, train)
