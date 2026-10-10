"""The fixed environmental logistic model, with portable JSON scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

LOG_COLUMNS = (
    "fwi_lag1", "fwi_mean_7d", "fwi_max_7d", "fwi_mean_30d",
    "past_fire_detections_7d", "past_fire_detections_30d", "dc_lag1",
)
FRACTION_COLUMNS = (
    "peat_fraction_peatmap2017", "shrub_fraction_2020", "grass_fraction_2020",
    "annual_cropland_fraction_2020", "built_fraction_2020", "bare_fraction_2020",
    "water_fraction_2020", "herbaceous_wetland_fraction_2020", "mangrove_fraction_2020",
    "industrial_oil_palm_fraction_2019", "smallholder_oil_palm_fraction_2019",
)
TARGET = "fire_positive_next_7d"


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Only named, past-weather and measured-landscape predictors enter fitting."""
    needed = list(LOG_COLUMNS) + ["ffmc_lag1"] + list(FRACTION_COLUMNS)
    numeric = frame[needed].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("missing or nonfinite predictors; imputation is disabled")
    if (numeric[list(LOG_COLUMNS)] < 0).any().any():
        raise ValueError("negative fire weather or detection counts")
    if not numeric.ffmc_lag1.between(0, 101).all():
        raise ValueError("FFMC outside its supported range")
    fractions = numeric[list(FRACTION_COLUMNS)]
    if ((fractions < 0) | (fractions > 1)).any().any():
        raise ValueError("landscape fraction outside [0, 1]")
    result = pd.DataFrame(index=frame.index)
    for column in LOG_COLUMNS:
        result[f"log1p_{column}"] = np.log1p(numeric[column])
    result["ffmc_lag1"] = numeric.ffmc_lag1
    result[list(FRACTION_COLUMNS)] = fractions
    dates = pd.to_datetime(frame.prediction_date, errors="raise")
    if dates.isna().any():
        raise ValueError("missing prediction date")
    phase = 2 * np.pi * (dates.dt.dayofyear - 1) / 365.2425
    result["annual_sin"] = np.sin(phase)
    result["annual_cos"] = np.cos(phase)
    return result


def validate_time(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    columns = ["prediction_date", "outcome_end_date", "feature_end_date", "fwi_source_date", "components_source_date"]
    for column in columns:
        result[column] = pd.to_datetime(result[column], errors="raise")
        if result[column].isna().any():
            raise ValueError(f"missing {column}")
    if result.duplicated(["cell_id", "prediction_date"]).any():
        raise ValueError("duplicate cell-week keys")
    for column in columns[2:]:
        if not (result[column] == result.prediction_date - pd.Timedelta(days=1)).all():
            raise ValueError(f"{column} must precede prediction by one day")
    if not (result.outcome_end_date == result.prediction_date + pd.Timedelta(days=6)).all():
        raise ValueError("outcome must span seven calendar days")
    if not result[TARGET].isin([0, 1]).all():
        raise ValueError("labels must be binary")
    return result


def cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not frame.landscape_area_quality_ok.isin([True, False]).all():
        raise ValueError("landscape quality must be an explicit boolean")
    mask = (np.isfinite(frame.population_2020) & (frame.population_2020 >= 0)
            & frame.landscape_area_quality_ok)
    result = frame.loc[mask].copy()
    cells = set(result.cell_id)
    weeks = frame.prediction_date.unique()
    counts = result.groupby("prediction_date").cell_id.nunique().reindex(weeks, fill_value=0)
    if not cells or not (counts == len(cells)).all():
        raise ValueError("eligible cohort must be fixed across every week")
    if result.duplicated(["cell_id", "prediction_date"]).any():
        raise ValueError("duplicate cohort keys")
    excluded = sorted(set(frame.cell_id) - cells)
    if len(result) != len(weeks) * len(cells):
        raise ValueError("incomplete cohort")
    return result, excluded


def split_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    prediction = pd.to_datetime(frame.prediction_date)
    outcome = pd.to_datetime(frame.outcome_end_date)
    return frame.loc[(prediction.dt.year == year) & (outcome < pd.Timestamp(year + 1, 1, 1))].copy()


def fit_model(train: pd.DataFrame, validation: pd.DataFrame):
    if not (pd.to_datetime(train.prediction_date).dt.year == 2022).all():
        raise ValueError("training year must be 2022")
    if not (pd.to_datetime(validation.prediction_date).dt.year == 2023).all():
        raise ValueError("validation year must be 2023")
    x_train, x_validation = features(train), features(validation)
    scaler = StandardScaler().fit(x_train)
    z_train, z_validation = scaler.transform(x_train), scaler.transform(x_validation)
    scores, best, best_ap = [], None, -np.inf
    for c in [0.01, 0.1, 1.0, 10.0]:
        fitted = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, tol=1e-6,
                                    class_weight=None, random_state=17).fit(z_train, train[TARGET])
        if fitted.n_iter_.max() >= 1000:
            raise RuntimeError("base model failed to converge")
        ap = float(average_precision_score(validation[TARGET], fitted.predict_proba(z_validation)[:, 1]))
        scores.append({"C": c, "validation_average_precision": ap})
        if ap > best_ap:
            best, best_ap = fitted, ap
    logits = best.decision_function(z_validation).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, tol=1e-6, random_state=17).fit(logits, validation[TARGET])
    if calibrator.n_iter_.max() >= 1000 or calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("sigmoid calibration failed or reversed the risk ranking")
    return scaler, best, calibrator, pd.DataFrame(scores)


def export_model(scaler, model, calibrator, names: list[str]) -> dict:
    return {"format_version": "environment-logistic-v2", "feature_names": names,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "logistic": {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])},
        "calibrator": {"coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
        "selected_C": float(model.C), "train_year": 2022, "calibration_year": 2023,
        "score_interpretation": "relative risk; calibration shift checked in the exploratory evaluation"}


def portable_score(matrix: pd.DataFrame, model: dict) -> np.ndarray:
    values = matrix[model["feature_names"]].to_numpy(dtype=float)
    z = (values - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
    logit = z @ np.asarray(model["logistic"]["coefficients"]) + model["logistic"]["intercept"]
    calibrated = logit * model["calibrator"]["coefficient"] + model["calibrator"]["intercept"]
    return 1 / (1 + np.exp(-np.clip(calibrated, -709, 709)))
