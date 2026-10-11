"""Package the model selected by development AP, with reproducible inference."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .revised_model import TARGET, cohort, features, split_year, validate_time
from .train_environment import EXPOSURES, legacy_logit, sha256, write_json


def select_method(metrics: dict) -> str:
    development = metrics["2023"]["methods"]
    old = development.get("previous_no_coordinates")
    if old is not None and old["average_precision"] > development["risk_score"]["average_precision"]:
        return "previous_no_coordinates"
    return "risk_score"


def score_model(frame: pd.DataFrame, model: dict) -> np.ndarray:
    """Score named raw features; excludes exposure, coordinates and labels."""
    logit = legacy_logit(frame, model)
    calibration = model["calibrator"]
    return 1 / (1 + np.exp(-np.clip(logit * calibration["coefficient"] + calibration["intercept"], -709, 709)))


def reproduce_previous(frame: pd.DataFrame, frozen: dict) -> tuple[dict, dict]:
    """Rebuild the earlier 951-cell model on its original 2022 cohort."""
    population = frame.population_2020
    original_cohort = frame.loc[np.isfinite(population) & (population >= 0)]
    train = split_year(original_cohort, 2022)
    matrix = features(train)[frozen["feature_names"]]
    scaler = StandardScaler().fit(matrix)
    fitted = LogisticRegression(C=frozen["selected_C"], solver="lbfgs", max_iter=1000,
        tol=1e-6, class_weight=None, random_state=17).fit(scaler.transform(matrix), train[TARGET])
    if fitted.n_iter_.max() >= 1000:
        raise RuntimeError("selected model reproduction failed to converge")
    differences = [np.max(np.abs(scaler.mean_ - np.asarray(frozen["scaler"]["mean"]))),
        np.max(np.abs(scaler.scale_ - np.asarray(frozen["scaler"]["scale"]))),
        np.max(np.abs(fitted.coef_[0] - np.asarray(frozen["coefficients"]))),
        abs(float(fitted.intercept_[0]) - frozen["intercept"])]
    if max(differences) > 1e-8:
        raise RuntimeError(f"selected model differs from its frozen fit: {max(differences)}")
    model = {"format_version": "fireguard-mvp-1", "feature_names": frozen["feature_names"],
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "logistic": {"coefficients": fitted.coef_[0].tolist(), "intercept": float(fitted.intercept_[0])},
        "calibrator": copy.deepcopy(frozen["calibrator"]), "selected_C": float(fitted.C),
        "train_year": 2022, "calibration_year": 2023, "training_cohort_cells": train.cell_id.nunique()}
    return model, {"training_rows": len(train), "training_positives": int(train[TARGET].sum()),
                   "training_cohort_cells": train.cell_id.nunique(), "max_frozen_parameter_difference": float(max(differences))}


def build_mvp(experiment_path: Path, input_path: Path, output_path: Path) -> dict:
    experiment_path, input_path, output_path = map(Path, [experiment_path, input_path, output_path])
    if output_path.exists():
        raise FileExistsError(f"preserve existing package: {output_path}")
    metrics = json.loads((experiment_path / "metrics.json").read_text())
    selected = select_method(metrics)
    protocol = json.loads((experiment_path / "protocol.json").read_text())
    if sha256(input_path) != protocol["input_sha256"]:
        raise ValueError("input differs from the evaluated dataset")
    frame = validate_time(pd.read_csv(input_path))
    if selected == "previous_no_coordinates":
        frozen = json.loads((experiment_path / "legacy_baselines.json").read_text())[selected]
        model, reproduction = reproduce_previous(frame, frozen)
    else:
        model = json.loads((experiment_path / "model.json").read_text())
        reproduction = {"training_rows": protocol["counts"]["2022"]["rows"], "new_environmental_model": True}
    eligible, excluded = cohort(frame)
    model["score_interpretation"] = "relative risk score; retrospective reanalysis; calibration shifts across years"
    model["feature_transform"] = "log1p named FWI/count/DC columns; raw named fraction/FFMC columns; sine/cosine of 2*pi*(dayofyear-1)/365.2425"
    model["evaluation_cohort_cells"] = eligible.cell_id.nunique()
    output_path.mkdir(parents=True)
    write_json(output_path / "model.json", model)
    error = 0.
    for year in [2023, 2024]:
        old = pd.read_csv(experiment_path / f"predictions_{year}.csv.gz", parse_dates=["prediction_date"])
        part = split_year(eligible, year)
        predicted = part[["cell_id", "prediction_date"]].copy()
        predicted["selected_score"] = score_model(part, model)
        check = old.merge(predicted, on=["cell_id", "prediction_date"], validate="one_to_one")
        if len(check) != len(old):
            raise ValueError("selected model coverage differs")
        error = max(error, float(np.max(np.abs(check[selected] - check.selected_score))))
        check["risk_score"] = check.selected_score
        keep = ["cell_id", "prediction_date", "outcome_end_date", "feature_end_date", "fwi_source_date", "components_source_date",
                "y", "past_fire_detections_7d", *EXPOSURES.values(), "risk_score"]
        check[keep].to_csv(output_path / f"predictions_{year}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        if year == 2024:
            snapshot = part.loc[part.prediction_date == part.prediction_date.max()].copy()
            snapshot["risk_score"] = score_model(snapshot, model)
            snapshot["snapshot_type"] = "historical demonstration; " + snapshot.prediction_date.max().strftime("%Y-%m-%d")
            snapshot.to_csv(output_path / "historical_demo_cells.csv", index=False)
    if error > 1e-8:
        raise RuntimeError(f"packaged scores differ from evaluation: {error}")
    summary = {"selected_method": selected, "selection_year": 2023,
        "selection_metric": "average precision", "development_average_precision": metrics["2023"]["methods"][selected]["average_precision"],
        "exploratory_2024_average_precision": metrics["2024"]["methods"][selected]["average_precision"],
        "evaluation_cohort_cells": eligible.cell_id.nunique(), "excluded_cells": excluded,
        "model_sha256": sha256(output_path / "model.json"), "input_sha256": sha256(input_path),
        "max_packaged_score_error": error, "reproduction": reproduction,
        "limits": ["2024 is exploratory after earlier project analyses", "Scores are not verified operational probabilities",
                   "Agriculture covers annual cropland and closed-canopy oil palm only", "Oil-palm allocation does not beat every baseline"]}
    write_json(output_path / "selection.json", summary)
    write_json(output_path / "metrics.json", {year: metrics[year]["methods"][selected] for year in ["2023", "2024"]})
    pd.DataFrame({"feature": model["feature_names"], "coefficient_standardized": model["logistic"]["coefficients"],
        "training_mean": model["scaler"]["mean"], "training_scale": model["scaler"]["scale"]}).to_csv(output_path / "coefficients.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("data/experiments/environment-logistic-v2"))
    parser.add_argument("--input", type=Path, default=Path("data/probe/environment/weekly_environment.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/mvp-model"))
    args = parser.parse_args()
    print(json.dumps(build_mvp(args.experiment, args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
