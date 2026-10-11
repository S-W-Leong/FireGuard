import pandas as pd
import pytest


def fixture():
    return pd.DataFrame({"cell_id": ["b", "a", "c"] * 2,
        "prediction_date": ["2024-01-01"] * 3 + ["2024-01-08"] * 3,
        "y": [1, 0, 1, 0, 1, 0], "score": [1., 1., 0., 0., 2., 1.],
        "exposure": [10., 5., 20., 10., 5., 20.]})


def test_weekly_budget_and_ties_do_not_use_labels():
    # Break caught: global top-K, or selecting positives to resolve ties.
    from fireguard.monitoring_metrics import weekly_metrics
    df = fixture()
    result = weekly_metrics(df, "score", "exposure", 1)
    assert result.selected_cells.tolist() == [1, 1]
    assert result.true_positives.tolist() == [0, 1]
    assert result.captured_exposure.tolist() == [0, 5]
    assert result.total_positive_exposure.tolist() == [30, 5]
    df.y = 1 - df.y
    assert weekly_metrics(df, "score", "exposure", 1).selected_ids.tolist() == ["a", "a"]


def test_pooled_exposure_recall_uses_sum_not_average_of_weeks():
    from fireguard.monitoring_metrics import weekly_metrics, pool_metrics
    pooled = pool_metrics(weekly_metrics(fixture(), "score", "exposure", 1))
    assert pooled["recall"] == pytest.approx(1 / 3)
    assert pooled["precision"] == 0.5
    assert pooled["exposure_weighted_recall"] == pytest.approx(5 / 35)


def test_exposure_changes_ranking_without_adding_people_and_hectares():
    from fireguard.monitoring_metrics import weekly_metrics
    df = fixture()
    df["weighted"] = df.score * df.exposure
    assert weekly_metrics(df, "weighted", "exposure", 1).selected_ids.tolist() == ["b", "c"]
    with pytest.raises(ValueError):
        weekly_metrics(df, "score", "exposure", 0)
