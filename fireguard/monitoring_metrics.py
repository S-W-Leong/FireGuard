"""Weekly budget and exposure metrics; no outcome-informed tie breaking."""
import numpy as np
import pandas as pd


def weekly_metrics(frame: pd.DataFrame, score: str, exposure: str, k: int) -> pd.DataFrame:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("budget must be a positive integer")
    if frame.duplicated(["prediction_date", "cell_id"]).any():
        raise ValueError("duplicate cell-week keys")
    if not np.isfinite(frame[[score, exposure]].to_numpy()).all() or (frame[exposure] < 0).any():
        raise ValueError("scores and exposure must be finite; exposure nonnegative")
    if not frame.y.isin([0, 1]).all():
        raise ValueError("labels must be binary")
    rows = []
    for date, week in frame.groupby("prediction_date", sort=True):
        selected = week.sort_values([score, "cell_id"], ascending=[False, True], kind="stable").head(k)
        rows.append({"prediction_date": date, "selected_cells": len(selected),
            "positive_cells": int(week.y.sum()), "true_positives": int(selected.y.sum()),
            "total_positive_exposure": float((week.y * week[exposure]).sum()),
            "captured_exposure": float((selected.y * selected[exposure]).sum()),
            "selected_ids": "|".join(selected.cell_id)})
    return pd.DataFrame(rows)


def pool_metrics(weekly: pd.DataFrame) -> dict:
    totals = weekly[["selected_cells", "positive_cells", "true_positives", "total_positive_exposure", "captured_exposure"]].sum()
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else None
    return {"weeks": len(weekly), "selected_cells": int(totals.selected_cells),
        "positive_cells": int(totals.positive_cells), "true_positives": int(totals.true_positives),
        "precision": ratio(totals.true_positives, totals.selected_cells),
        "recall": ratio(totals.true_positives, totals.positive_cells),
        "exposure_weighted_recall": ratio(totals.captured_exposure, totals.total_positive_exposure),
        "total_positive_exposure": float(totals.total_positive_exposure),
        "captured_exposure": float(totals.captured_exposure)}
