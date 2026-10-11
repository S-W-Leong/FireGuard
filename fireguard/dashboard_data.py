"""Data-only dashboard decisions; outcomes never enter the allocation score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .monitoring_metrics import pool_metrics, weekly_metrics

EXPOSURE_MODES = {
    "Population": ("population_2020", "people", "2020"),
    "Annual cropland": ("annual_cropland_ha_2020", "ha", "2020"),
    "Closed-canopy oil palm": ("closed_canopy_oil_palm_ha_2019", "ha", "2019"),
}
COMBINED_MODE = "Combined exposure"
DEFAULT_WEIGHTS = {"Population": .5, "Annual cropland": .25, "Closed-canopy oil palm": .25}
SHARE_COLUMNS = {"Population": "population_share", "Annual cropland": "cropland_share",
                 "Closed-canopy oil palm": "oil_palm_share"}
METHODS = {
    "FireGuard × exposure": ("risk_score", True),
    "Model risk only": ("risk_score", False),
    "FWI × exposure": ("fwi_baseline", True),
    "FWI only": ("fwi_baseline", False),
    "Previous 7 days × exposure": ("past_fire_7d", True),
    "Previous 30 days × exposure": ("past_fire_30d", True),
    "Historical frequency × exposure": ("historical_frequency", True),
}


def reference_totals(frame: pd.DataFrame) -> dict:
    """Static exposure totals across unique eligible cells, never across weeks."""
    columns = [value[0] for value in EXPOSURE_MODES.values()]
    if not np.isfinite(frame[columns].to_numpy()).all() or (frame[columns] < 0).any().any():
        raise ValueError("reference exposure must be finite and nonnegative")
    if (frame.groupby("cell_id")[columns].nunique() > 1).any().any():
        raise ValueError("reference maps must be static for each cell")
    return frame.drop_duplicates("cell_id")[columns].sum().to_dict()


def _with_exposure(frame, exposure_mode, weights, totals):
    result = frame.copy()
    if exposure_mode != COMBINED_MODE:
        return result, EXPOSURE_MODES[exposure_mode][0]
    weights = DEFAULT_WEIGHTS if weights is None else weights
    if set(weights) != set(EXPOSURE_MODES):
        raise ValueError("weights must name population, cropland and oil palm")
    values = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1., rtol=0, atol=1e-9):
        raise ValueError("weights must be finite, nonnegative and sum to one")
    if totals is None:
        raise ValueError("combined exposure requires fixed reference totals")
    result["exposure_index"] = 0.
    for name, (column, _, _) in EXPOSURE_MODES.items():
        total = float(totals[column])
        if not np.isfinite(total) or total < 0 or (total == 0 and weights[name] > 0):
            raise ValueError("active exposure must have a positive finite reference total")
        if not np.isfinite(result[column]).all() or (result[column] < 0).any():
            raise ValueError("exposure must be finite and nonnegative")
        share = SHARE_COLUMNS[name]
        result[share] = result[column] / total if total > 0 else 0.
        result[share + "_contribution"] = result[share] * weights[name]
        result["exposure_index"] += result[share + "_contribution"]
    return result, "exposure_index"


def rank_week(frame: pd.DataFrame, exposure_mode: str, k: int, *, weights=None, totals=None) -> pd.DataFrame:
    """Return one week's stable ranking, respecting the equal-cost team budget."""
    if frame.prediction_date.nunique() != 1:
        raise ValueError("dispatch needs a single week")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("budget must be a positive integer")
    ranked, exposure = _with_exposure(frame, exposure_mode, weights, totals)
    if frame.cell_id.duplicated().any():
        raise ValueError("duplicate cell IDs")
    if not np.isfinite(ranked[["risk_score", exposure]].to_numpy()).all() or (ranked[exposure] < 0).any():
        raise ValueError("risk and exposure must be finite; exposure nonnegative")
    ranked["priority_score"] = ranked.risk_score * ranked[exposure]
    ranked = ranked.sort_values(["priority_score", "cell_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = ranked["rank"] <= k
    return ranked


def evaluate(frame: pd.DataFrame, exposure_mode: str, k: int, *, weights=None, totals=None) -> pd.DataFrame:
    """Fair weekly allocations and pooled capture metrics on identical cells."""
    work, exposure = _with_exposure(frame, exposure_mode, weights, totals)
    rows = []
    for name, (score, weighted) in METHODS.items():
        work["allocation_score"] = work[score] * work[exposure] if weighted else work[score]
        weekly = weekly_metrics(work, "allocation_score", exposure, k)
        row = pool_metrics(weekly)
        keys = weekly.assign(cell_id=weekly.selected_ids.str.split("|")).explode("cell_id")
        chosen = work.merge(keys[["prediction_date", "cell_id"]], on=["prediction_date", "cell_id"], validate="one_to_one")
        for exposure_name, metric in [("Population", "population_capture"), ("Annual cropland", "cropland_capture"),
                             ("Closed-canopy oil palm", "oil_palm_capture")]:
            column = EXPOSURE_MODES[exposure_name][0]
            denominator = float((work.y * work[column]).sum())
            row[metric] = float((chosen.y * chosen[column]).sum() / denominator) if denominator > 0 else None
        rows.append({"method": name, **row})
    return pd.DataFrame(rows)


def policy_comparison(frame: pd.DataFrame, k: int, *, weights=None, totals=None) -> pd.DataFrame:
    """Selected-week portfolio sensitivity, without examining subsequent labels."""
    current = DEFAULT_WEIGHTS if weights is None else weights
    settings = {"Current settings": current,
        "People focused": dict(zip(EXPOSURE_MODES, [.75, .125, .125])),
        "Balanced": DEFAULT_WEIGHTS,
        "Agriculture focused": dict(zip(EXPOSURE_MODES, [.25, .375, .375]))}
    current_ids = set(rank_week(frame, COMBINED_MODE, k, weights=current, totals=totals).query("selected").cell_id)
    rows = []
    for name, values in settings.items():
        selected = rank_week(frame, COMBINED_MODE, k, weights=values, totals=totals).query("selected")
        rows.append({"Policy": name, "Population weight": values["Population"],
            "Cropland weight": values["Annual cropland"], "Oil palm weight": values["Closed-canopy oil palm"],
            "Cells changed": len(current_ids - set(selected.cell_id)),
            "People in selected cells": float(selected.population_2020.sum()),
            "Annual cropland ha": float(selected.annual_cropland_ha_2020.sum()),
            "Oil palm ha": float(selected.closed_canopy_oil_palm_ha_2019.sum())})
    return pd.DataFrame(rows)
