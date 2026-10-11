"""Reproducible train-2022/develop-2023/explore-2024 experiment."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from .monitoring_metrics import pool_metrics, weekly_metrics
from .revised_model import TARGET, cohort, export_model, features, fit_model, portable_score, split_year, validate_time

EXPOSURES = {"population": "population_2020", "annual_cropland": "annual_cropland_ha_2020",
             "closed_canopy_oil_palm": "closed_canopy_oil_palm_ha_2019"}
BUDGETS = [10, 25, 50, 100]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def legacy_logit(frame: pd.DataFrame, model: dict) -> np.ndarray:
    matrix = pd.DataFrame(index=frame.index)
    phase = 2 * np.pi * (pd.to_datetime(frame.prediction_date).dt.dayofyear - 1) / 365.2425
    for name in model["feature_names"]:
        if name.startswith("log1p_"):
            matrix[name] = np.log1p(frame[name.removeprefix("log1p_")])
        elif name == "annual_sin":
            matrix[name] = np.sin(phase)
        elif name == "annual_cos":
            matrix[name] = np.cos(phase)
        else:
            matrix[name] = frame[name]
    z = (matrix.to_numpy() - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
    logistic = model.get("logistic", model)
    return z @ np.asarray(logistic["coefficients"]) + logistic["intercept"]


def calibration_summary(y, scores) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    bins = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (scores >= low) & ((scores < high) if high < 1 else (scores <= high))
        bins.append({"lower": float(low), "upper": float(high), "rows": int(mask.sum()),
            "mean_score": float(scores[mask].mean()) if mask.any() else None,
            "observed_rate": float(y[mask].mean()) if mask.any() else None})
    return {"brier_score": float(brier_score_loss(y, scores)), "mean_score": float(scores.mean()),
            "observed_rate": float(y.mean()), "bins": bins}


def evaluate(frame: pd.DataFrame, methods: list[str]) -> tuple[dict, pd.DataFrame]:
    """All methods see exactly the same rows and exposure in each comparison."""
    result, records = {}, []
    for method in methods:
        entry = {"average_precision": float(average_precision_score(frame.y, frame[method])), "allocation": {}}
        if method in ["risk_score", "original_logistic", "previous_no_coordinates", "historical_frequency"]:
            entry["calibration"] = calibration_summary(frame.y, frame[method])
        for exposure_name, exposure_column in EXPOSURES.items():
            exposed = frame.copy()
            exposed["weighted_score"] = exposed[method] * exposed[exposure_column]
            entry["allocation"][exposure_name] = {}
            for strategy, score in [("risk_only", method), ("risk_times_exposure", "weighted_score")]:
                metrics = {}
                for k in BUDGETS:
                    weekly = weekly_metrics(exposed, score, exposure_column, k)
                    metrics[str(k)] = pool_metrics(weekly)
                    weekly["method"], weekly["exposure"], weekly["strategy"], weekly["K"] = method, exposure_name, strategy, k
                    records.append(weekly)
                entry["allocation"][exposure_name][strategy] = metrics
        result[method] = entry
    return result, pd.concat(records, ignore_index=True)


def run_experiment(input_path: Path, output_path: Path, legacy_models: dict | None = None,
                   grid_path: Path | None = None) -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"preserve existing experiment: {output_path}")
    input_hash = sha256(input_path)
    frame = validate_time(pd.read_csv(input_path))
    frame, excluded = cohort(frame)
    # Validate every predictor before any fit; this checks coverage, not future labels.
    features(frame)
    for column in EXPOSURES.values():
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"invalid exposure {column}")
    if grid_path is not None:
        grid = pd.read_csv(grid_path)[["cell_id", "latitude", "longitude"]]
        frame = frame.merge(grid, on="cell_id", validate="many_to_one")
        if frame[["latitude", "longitude"]].isna().any().any():
            raise ValueError("missing legacy model coordinates")
    train, validation, exploration = [split_year(frame, year) for year in [2022, 2023, 2024]]
    for year, part in zip([2022, 2023, 2024], [train, validation, exploration]):
        if not len(part) or part[TARGET].nunique() != 2:
            raise ValueError(f"year {year} needs observations from both classes")
    output_path.mkdir(parents=True)
    protocol = {"experiment": "environment-logistic-v2", "seed": 17,
        "input": str(input_path), "input_sha256": input_hash, "feature_names": list(features(train).columns),
        "splits": {"train": 2022, "selection_and_calibration": 2023, "exploratory_evaluation": 2024},
        "C_candidates": [0.01, 0.1, 1., 10.], "selection": "2023 AP; smaller C breaks ties",
        "cohort_cells": frame.cell_id.nunique(), "excluded_cells": excluded,
        "no_imputation": True, "no_base_model_refit": True, "budgets": BUDGETS,
        "exposures": EXPOSURES, "ties": "descending score then ascending cell_id",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "counts": {str(year): {"rows": len(part), "weeks": part.prediction_date.nunique(),
                  "positives": int(part[TARGET].sum())} for year, part in zip([2022, 2023, 2024], [train, validation, exploration])}}
    human_protocol = Path("docs/FireGuard_Model_Protocol.md")
    if human_protocol.exists():
        protocol["human_protocol_sha256"] = sha256(human_protocol)
    # Freeze this metadata before model selection and before evaluation.
    write_json(output_path / "protocol.json", protocol)
    scaler, fitted, calibrator, candidates = fit_model(train, validation)
    model = export_model(scaler, fitted, calibrator, list(features(train).columns))
    model["protocol_sha256"] = sha256(output_path / "protocol.json")
    write_json(output_path / "model.json", model)
    candidates.to_csv(output_path / "candidate_scores.csv", index=False)
    frequency = train.groupby("cell_id")[TARGET].agg(["sum", "count"])
    frequency["historical_frequency"] = (frequency["sum"] + 1) / (frequency["count"] + 2)
    frequency.to_csv(output_path / "historical_frequency.csv")
    legacy_models = copy.deepcopy(legacy_models or {})
    # Frozen older model without coordinates had no calibrator. Give it the
    # same 2023 calibration opportunity; its fitted coefficients stay frozen.
    for name, older in legacy_models.items():
        if "calibrator" not in older:
            logits = legacy_logit(validation, older).reshape(-1, 1)
            calibration = LogisticRegression(C=1e6, max_iter=1000, tol=1e-6, random_state=17).fit(logits, validation[TARGET])
            if calibration.coef_[0, 0] <= 0 or calibration.n_iter_.max() >= 1000:
                raise RuntimeError("legacy calibration did not converge or reversed ranking")
            older["calibrator"] = {"coefficient": float(calibration.coef_[0, 0]), "intercept": float(calibration.intercept_[0])}
    write_json(output_path / "legacy_baselines.json", legacy_models)
    all_results, max_error, weekly_records = {}, 0., []
    for year, part in [(2023, validation), (2024, exploration)]:
        predictions = part[["cell_id", "prediction_date", "outcome_end_date", "feature_end_date",
            "fwi_source_date", "components_source_date", TARGET, "past_fire_detections_7d",
            *EXPOSURES.values()]].copy().rename(columns={TARGET: "y"})
        matrix = features(part)
        logits = fitted.decision_function(scaler.transform(matrix))
        predictions["risk_score"] = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        predictions["uncalibrated_score"] = fitted.predict_proba(scaler.transform(matrix))[:, 1]
        max_error = max(max_error, float(np.max(np.abs(predictions.risk_score - portable_score(matrix, model)))))
        predictions["historical_frequency"] = part.cell_id.map(frequency.historical_frequency)
        for name, source in [("fwi_baseline", "fwi_lag1"), ("past_fire_7d", "past_fire_detections_7d"),
                             ("past_fire_30d", "past_fire_detections_30d")]:
            predictions[name] = part[source]
        for name, older in legacy_models.items():
            raw = legacy_logit(part, older)
            calibration = older["calibrator"]
            predictions[name] = 1 / (1 + np.exp(-np.clip(raw * calibration["coefficient"] + calibration["intercept"], -709, 709)))
        methods = ["risk_score", "historical_frequency", "fwi_baseline", "past_fire_7d", "past_fire_30d", *legacy_models]
        metrics, weekly = evaluate(predictions, methods)
        weekly["year"] = year
        weekly_records.append(weekly)
        subset = predictions.loc[predictions.past_fire_detections_7d == 0]
        sensitivity = {name: float(average_precision_score(subset.y, subset[name])) if subset.y.sum() > 0 else None for name in methods}
        all_results[str(year)] = {"rows": len(predictions), "positives": int(predictions.y.sum()),
            "methods": metrics, "no_prior_7d_detection": {"rows": len(subset), "positives": int(subset.y.sum()), "average_precision": sensitivity}}
        predictions.to_csv(output_path / f"predictions_{year}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.concat(weekly_records, ignore_index=True).to_csv(output_path / "weekly_metrics.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    write_json(output_path / "metrics.json", all_results)
    pd.DataFrame({"feature": model["feature_names"], "coefficient_standardized": fitted.coef_[0],
                  "training_mean": scaler.mean_, "training_scale": scaler.scale_}).to_csv(output_path / "coefficients.csv", index=False)
    if max_error > 1e-12 or sha256(input_path) != input_hash:
        raise RuntimeError("portable export mismatch or input changed")
    write_json(output_path / "verification.json", {"status": "passed", "portable_max_error": max_error,
        "input_hash_unchanged": True, "fixed_cohort_cells": frame.cell_id.nunique(),
        "training_rows": len(train), "excluded_cross_year_training_rows": int((pd.to_datetime(frame.prediction_date).dt.year == 2022).sum() - len(train)),
        "no_coordinates_or_population_predictors": not any(n in model["feature_names"] for n in ["latitude", "longitude", "population_2020"]),
        "evaluation_status": "exploratory; 2024 has previously been examined"})
    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/probe/environment/weekly_environment.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/environment-logistic-v2"))
    args = parser.parse_args()
    original_path = Path("data/experiments/logistic-v1/model.json")
    review_path = Path("data/experiments/feature-review/models.json")
    frozen = {"original_logistic": json.loads(original_path.read_text()),
              "previous_no_coordinates": json.loads(review_path.read_text())["without_coordinates"]}
    hashes = {str(p): sha256(p) for p in [original_path, review_path]}
    results = run_experiment(args.input, args.output, frozen, Path("data/probe/grid_exposure.csv"))
    if any(sha256(Path(p)) != before for p, before in hashes.items()):
        raise RuntimeError("frozen comparator changed")
    write_json(args.output / "frozen_input_hashes.json", hashes)
    for year, metrics in results.items():
        print(year, {method: round(entry["average_precision"], 4) for method, entry in metrics["methods"].items()}, flush=True)


if __name__ == "__main__":
    main()
