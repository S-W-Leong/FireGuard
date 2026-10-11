import json

import numpy as np
import pandas as pd
import pytest

from test_train_environment import input_table


def numeric_tree():
    return {"feature_names": ["x"], "baseline_logit": -2.,
        "calibrator": {"coefficient": 1.2, "intercept": 0.2},
        "trees": [{"feature_idx": [0, 0, 0], "threshold": [1., 0., 0.],
                   "left": [1, 0, 0], "right": [2, 0, 0], "is_leaf": [False, True, True],
                   "value": [0., -0.7, 0.4]}]}


def test_exported_tree_uses_inclusive_left_threshold_and_one_calibration():
    # Break caught: threshold equality goes right, or shrinkage/calibration twice.
    from fireguard.boosted_model import portable_tree_score
    matrix = pd.DataFrame({"x": [1., np.nextafter(1., 2.), -999., 999.]})
    expected = [1 / (1 + np.exp(3.04)), 1 / (1 + np.exp(1.72)),
                1 / (1 + np.exp(3.04)), 1 / (1 + np.exp(1.72))]
    np.testing.assert_allclose(portable_tree_score(matrix, numeric_tree()), expected, atol=1e-14)
    matrix.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        portable_tree_score(matrix, numeric_tree())


@pytest.mark.parametrize("tree_ap,tree_recall,expected", [(0.6, 0.19, False), (0.48, 0.25, False), (0.6, 0.2, True)])
def test_adoption_requires_development_ap_and_weekly_budget_improvement(tree_ap, tree_recall, expected):
    # Break caught: pooled ranking improvement hides worse weekly decisions.
    from fireguard.boosted_model import adoption_gate
    def entry(ap, recall):
        return {"average_precision": ap, "allocation": {"population": {"risk_only": {"50": {"recall": recall}}}}}
    assert adoption_gate(entry(tree_ap, tree_recall), entry(0.48, 0.2)) == expected


def large_table():
    pieces = []
    for i in range(150):
        piece = input_table()
        piece.cell_id = piece.cell_id + str(i)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def incumbent():
    return {"feature_names": ["log1p_fwi_lag1"], "scaler": {"mean": [0.], "scale": [1.]},
            "logistic": {"coefficients": [0.], "intercept": -2.},
            "calibrator": {"coefficient": 1., "intercept": 0.}}


def test_fit_and_portable_export_match_on_unseen_rows_and_reject_wrong_year():
    # Break caught: exporting raw node values incorrectly or training future years.
    from fireguard.boosted_model import fit_challenger, export_trees, portable_tree_score
    from fireguard.revised_model import features
    table = large_table()
    train = table.loc[table.prediction_date.dt.year == 2022]
    validation = table.loc[table.prediction_date.dt.year == 2023]
    fitted, calibrator = fit_challenger(train, validation)
    exported = export_trees(fitted, calibrator, list(features(train).columns))
    unseen = features(table.loc[table.prediction_date.dt.year == 2024])
    expected = calibrator.predict_proba(fitted.decision_function(unseen).reshape(-1, 1))[:, 1]
    np.testing.assert_allclose(portable_tree_score(unseen, exported), expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="year"):
        fit_challenger(validation, train)


def test_exploratory_labels_do_not_change_fit_or_adoption_and_incumbent_is_preserved(tmp_path):
    # Break caught: 2024 informs fit/adoption, or challenger overwrites the UI model.
    from fireguard.boosted_model import run_challenger
    table = large_table()
    input_path = tmp_path / "input.csv.gz"
    model_path = tmp_path / "incumbent.json"
    model_path.write_text(json.dumps(incumbent()))
    before = model_path.read_bytes()
    table.to_csv(input_path, index=False)
    run_challenger(input_path, model_path, tmp_path / "first")
    first = json.loads((tmp_path / "first/model.json").read_text())
    recommendation = json.loads((tmp_path / "first/recommendation.json").read_text())
    table.loc[table.prediction_date.dt.year == 2024, "fire_positive_next_7d"] = 1 - table.loc[table.prediction_date.dt.year == 2024, "fire_positive_next_7d"]
    table.to_csv(input_path, index=False)
    run_challenger(input_path, model_path, tmp_path / "second")
    second = json.loads((tmp_path / "second/model.json").read_text())
    changed = json.loads((tmp_path / "second/recommendation.json").read_text())
    for key in ["trees", "baseline_logit", "calibrator"]:
        assert first[key] == second[key]
    assert recommendation["adopt_challenger"] == changed["adopt_challenger"]
    assert model_path.read_bytes() == before
    assert len(pd.read_csv(tmp_path / "first/predictions_2024.csv.gz")) == 600
    assert json.loads((tmp_path / "first/verification.json").read_text())["portable_max_error"] < 1e-12
    with pytest.raises(FileExistsError):
        run_challenger(input_path, model_path, tmp_path / "first")
