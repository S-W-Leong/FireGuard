"""Build a portable dashboard bundle from verified historical model exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard_data import EXPOSURE_MODES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join_inputs(scores: pd.DataFrame, predictors: pd.DataFrame) -> pd.DataFrame:
    keys = ["cell_id", "prediction_date"]
    extra = [c for c in predictors if c not in scores or c in keys]
    result = scores.merge(predictors[extra], on=keys, how="left", validate="one_to_one", indicator=True)
    if (result._merge != "both").any():
        raise ValueError("missing predictor cells in the model handoff")
    result = result.drop(columns="_merge")
    prediction = pd.to_datetime(result.prediction_date)
    for column in ["feature_end_date", "fwi_source_date", "components_source_date"]:
        if column in result and not (pd.to_datetime(result[column]) < prediction).all():
            raise ValueError(f"{column} must precede the prediction date")
    if "outcome_end_date" in result and not (pd.to_datetime(result.outcome_end_date) == prediction + pd.Timedelta(days=6)).all():
        raise ValueError("outcome windows must contain the next seven local calendar days")
    return result


def build_bundle(model_package: Path, input_path: Path, baseline_package: Path, grid_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"preserve the current bundle; choose a new --output: {output}")
    verification = json.loads((model_package / "verification.json").read_text())
    if verification.get("status") != "passed":
        raise ValueError("model package must pass export verification")
    selection_path = model_package / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        selection = json.loads((model_package / "recommendation.json").read_text())
        if selection.get("status") != "verified" or not selection.get("adopt_challenger"):
            raise ValueError("challenger package must be verified and recommended before integration")
    if sha256(model_package / "model.json") != selection["model_sha256"]:
        raise ValueError("model differs from its verified handoff")
    protocol = json.loads((baseline_package / "protocol.json").read_text())
    if sha256(input_path) != protocol["input_sha256"]:
        raise ValueError("predictors differ from the evaluated dataset")
    model = json.loads((model_package / "model.json").read_text())
    raw_columns = ["cell_id", "prediction_date", "feature_end_date", "fwi_source_date", "components_source_date",
        "fwi_lag1", "fwi_mean_7d", "fwi_max_7d", "fwi_mean_30d", "past_fire_detections_30d",
        "ffmc_lag1", "dc_lag1", "peat_fraction_peatmap2017"]
    predictors = pd.read_csv(input_path, usecols=raw_columns)
    grid = json.loads(grid_path.read_text())
    cells = pd.DataFrame([f["properties"] for f in grid["features"]])
    frames, sources = [], {}
    for year in [2023, 2024]:
        pred_path = model_package / f"predictions_{year}.csv.gz"
        base_path = baseline_package / f"predictions_{year}.csv.gz"
        scores = pd.read_csv(pred_path)
        baselines = pd.read_csv(base_path, usecols=["cell_id", "prediction_date", "fwi_baseline",
            "past_fire_7d", "past_fire_30d", "historical_frequency"])
        # Model packages can contain the same baselines; preserve their evaluated values.
        extra = [c for c in baselines if c not in scores or c in ["cell_id", "prediction_date"]]
        joined = join_inputs(scores, predictors)
        joined = joined.merge(baselines[extra], on=["cell_id", "prediction_date"], validate="one_to_one")
        joined = joined.merge(cells[["cell_id", "latitude", "longitude", "area_km2"]], on="cell_id", validate="many_to_one")
        if len(joined) != len(scores) or joined.groupby("prediction_date").size().nunique() != 1:
            raise ValueError("dashboard must preserve all scored cells in every week")
        required = ["risk_score", "latitude", "longitude", "fwi_baseline", "historical_frequency",
            *(v[0] for v in EXPOSURE_MODES.values())]
        if not np.isfinite(joined[required].to_numpy()).all():
            raise ValueError("missing or invalid scores/exposure cannot be silently substituted")
        frames.append(joined)
        sources[str(pred_path)] = sha256(pred_path)
        sources[str(base_path)] = sha256(base_path)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["cell_id", "prediction_date"]).any():
        raise ValueError("duplicate cell-week keys")
    sources[str(input_path)] = sha256(input_path)
    sources[str(grid_path)] = sha256(grid_path)
    sources[str(model_package / "model.json")] = sha256(model_package / "model.json")
    output.mkdir(parents=True)
    data.to_csv(output / "cell_weeks.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    shutil.copyfile(grid_path, output / "grid.geojson")
    model_metrics = json.loads((model_package / "metrics.json").read_text())
    if "methods" in model_metrics["2024"]:
        model_metrics = {year: value["methods"]["risk_score"] for year, value in model_metrics.items() if year in ["2023", "2024"]}
    manifest = {
        "format": "fireguard-dashboard-1", "status": "historical demonstration",
        "prototype_branch": "origin/Dashboard-1.0",
        "prototype_commit": "54435e96f0dbea77ddbe3bd4b0b6a255c84a7bb1",
        "model_package": str(model_package), "model": model, "selection": selection,
        "model_name": "Logistic regression" if "logistic" in model else "Histogram gradient boosting",
        "model_metrics": model_metrics, "source_hashes": sources,
        "cohort_cells": int(data.cell_id.nunique()), "grid_cells": len(grid["features"]),
        "rows": len(data), "weeks": int(data.prediction_date.nunique()),
        "years": [2023, 2024], "train_year": 2022, "development_year": 2023,
        "exploratory_year": 2024,
        "bundle_hashes": {name: sha256(output / name) for name in ["cell_weeks.csv.gz", "grid.geojson"]},
        "sources": [
            {"dataset": "Copernicus CEMS historical fire danger", "date": "2022–2024",
             "measurement": "FWI, FFMC and DC; 0.25° output, approximately 31 km meteorological resolution"},
            {"dataset": "NASA FIRMS VIIRS S-NPP", "date": "2021–2024",
             "measurement": "Nominal/high-confidence vegetation detections, deduplicated; Asia/Jakarta dates"},
            {"dataset": "WorldPop", "date": "2020", "measurement": "Estimated persons per cell"},
            {"dataset": "ESA WorldCover v100", "date": "2020", "measurement": "Annual-cropland hectares; does not cover all crops"},
            {"dataset": "Descals global oil-palm map", "date": "2019", "measurement": "Closed-canopy industrial/smallholder oil-palm hectares; excludes young/sparse stands"},
            {"dataset": "PEATMAP", "date": "2017 compilation; Indonesian source 2011",
             "measurement": "Historical peat extent; contextual, not peat moisture or depth"},
        ],
        "limits": ["Historical reanalysis; publication-time availability is not verified for live forecasting.",
            "2023 is development and calibration; 2024 is exploratory after earlier analyses.",
            "Risk scores are relative rankings, not verified operational probabilities.",
            "A positive label includes continuing fire and agricultural burning; no detection does not prove no fire.",
            "Exposure capture does not measure prevented fires, affected people, burned hectares or avoided losses.",
            "Static agriculture covers annual cropland and closed-canopy oil palm only.",
            "11 cells are excluded because population is missing or landscape area-quality checks fail."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"output": str(output), "rows": len(data), "cells": manifest["cohort_cells"], "weeks": manifest["weeks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-package", type=Path, default=Path("data/experiments/mvp-model"))
    parser.add_argument("--input", type=Path, default=Path("data/probe/environment/weekly_environment.csv.gz"))
    parser.add_argument("--baselines", type=Path, default=Path("data/experiments/environment-logistic-v2"))
    parser.add_argument("--grid", type=Path, default=Path("data/probe/grid.geojson"))
    parser.add_argument("--output", type=Path, default=Path("data/dashboard"))
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.model_package, args.input, args.baselines, args.grid, args.output), indent=2))


if __name__ == "__main__":
    main()
