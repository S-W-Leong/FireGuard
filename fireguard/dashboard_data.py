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
METHODS = {
    "FireGuard × exposure": ("risk_score", True),
    "Model risk only": ("risk_score", False),
    "FWI × exposure": ("fwi_baseline", True),
    "FWI only": ("fwi_baseline", False),
    "Previous 7 days × exposure": ("past_fire_7d", True),
    "Previous 30 days × exposure": ("past_fire_30d", True),
    "Historical frequency × exposure": ("historical_frequency", True),
}


def rank_week(frame: pd.DataFrame, exposure_mode: str, k: int) -> pd.DataFrame:
    """Return one week's stable ranking, respecting the equal-cost team budget."""
    if frame.prediction_date.nunique() != 1:
        raise ValueError("dispatch needs a single week")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("budget must be a positive integer")
    exposure, _, _ = EXPOSURE_MODES[exposure_mode]
    if frame.cell_id.duplicated().any():
        raise ValueError("duplicate cell IDs")
    if not np.isfinite(frame[["risk_score", exposure]].to_numpy()).all() or (frame[exposure] < 0).any():
        raise ValueError("risk and exposure must be finite; exposure nonnegative")
    ranked = frame.copy()
    ranked["priority_score"] = ranked.risk_score * ranked[exposure]
    ranked = ranked.sort_values(["priority_score", "cell_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected"] = ranked["rank"] <= k
    return ranked


def evaluate(frame: pd.DataFrame, exposure_mode: str, k: int) -> pd.DataFrame:
    """Fair weekly allocations and pooled capture metrics on identical cells."""
    exposure, _, _ = EXPOSURE_MODES[exposure_mode]
    work = frame.copy()
    rows = []
    for name, (score, weighted) in METHODS.items():
        work["allocation_score"] = work[score] * work[exposure] if weighted else work[score]
        row = pool_metrics(weekly_metrics(work, "allocation_score", exposure, k))
        rows.append({"method": name, **row})
    return pd.DataFrame(rows)
