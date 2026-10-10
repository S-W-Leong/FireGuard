import json

import pandas as pd
import pytest

from test_revised_model import fixture_rows


def input_table():
    pieces = []
    for year, dates in [(2022, ["2022-01-03", "2022-01-10"]),
                        (2023, ["2023-01-02", "2023-01-09"]),
                        (2024, ["2024-01-01", "2024-01-08"])]:
        for date in dates:
            rows = fixture_rows()
            rows["cell_id"] = ["a", "b"]
            rows["prediction_date"] = pd.Timestamp(date)
            rows["outcome_end_date"] = pd.Timestamp(date) + pd.Timedelta(days=6)
            for column in ["feature_end_date", "fwi_source_date", "components_source_date"]:
                rows[column] = pd.Timestamp(date) - pd.Timedelta(days=1)
            rows["population_2020"] = [10., 20.]
            rows["annual_cropland_ha_2020"] = [2., 5.]
            rows["closed_canopy_oil_palm_ha_2019"] = [0., 10.]
            rows["landscape_area_quality_ok"] = True
            rows["fire_positive_next_7d"] = [0, 1]
            pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def test_end_to_end_exports_scores_and_2024_labels_cannot_change_fit(tmp_path):
    # Break caught: selection/refit/calibration sees exploratory-year labels.
    from fireguard.train_environment import run_experiment
    table = input_table()
    path = tmp_path / "input.csv.gz"
    table.to_csv(path, index=False)
    run_experiment(path, tmp_path / "first")
    model = json.loads((tmp_path / "first/model.json").read_text())
    pred = pd.read_csv(tmp_path / "first/predictions_2024.csv.gz")
    assert len(pred) == 4
    assert pred.risk_score.between(0, 1).all()
    assert pred.groupby("prediction_date").cell_id.nunique().tolist() == [2, 2]
    assert json.loads((tmp_path / "first/verification.json").read_text())["portable_max_error"] < 1e-14
    metrics = json.loads((tmp_path / "first/metrics.json").read_text())
    assert metrics["2024"]["no_prior_7d_detection"]["average_precision"]["risk_score"] is None
    from fireguard.select_mvp import build_mvp
    build_mvp(tmp_path / "first", path, tmp_path / "mvp")
    selected = json.loads((tmp_path / "mvp/selection.json").read_text())
    assert selected["selected_method"] == "risk_score"
    selected_pred = pd.read_csv(tmp_path / "mvp/predictions_2024.csv.gz")
    assert selected_pred.risk_score.tolist() == pytest.approx(pred.risk_score.tolist(), abs=1e-12)
    snapshot = pd.read_csv(tmp_path / "mvp/historical_demo_cells.csv")
    assert snapshot.snapshot_type.str.contains("2024-01-08").all()
    table.loc[table.prediction_date.dt.year == 2024, "fire_positive_next_7d"] = [1, 0, 1, 0]
    table.to_csv(path, index=False)
    run_experiment(path, tmp_path / "second")
    changed = json.loads((tmp_path / "second/model.json").read_text())
    for key in ["scaler", "logistic", "calibrator", "selected_C"]:
        assert changed[key] == model[key]
    with pytest.raises(FileExistsError):
        run_experiment(path, tmp_path / "first")
