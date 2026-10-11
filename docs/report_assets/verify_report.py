"""Recompute report evidence from cached artifacts without fitting or overwriting models.

Run from the repository root with the project's Python environment.
The dashboard-only checks work from a normal checkout; the raw-score check
also requires the ignored, locally cached environmental table.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss

from data_loader import load_fireguard_data, load_grid, load_manifest
from fireguard.dashboard_data import (
    COMBINED_MODE, DEFAULT_WEIGHTS, EXPOSURE_MODES, evaluate,
    policy_comparison, rank_week, reference_totals,
)


def read_json(relative):
    return json.loads((ROOT / relative).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_allocation(frame, scores, exposure, k):
    work = frame.assign(report_score=np.asarray(scores))
    selected = work.sort_values(
        ["prediction_date", "report_score", "cell_id"],
        ascending=[True, False, True], kind="stable",
    ).groupby("prediction_date", sort=False).head(k)
    true = int(selected.y.sum())
    positives = int(work.y.sum())
    total = float((work.y * work[exposure]).sum())
    captured = float((selected.y * selected[exposure]).sum())
    return {
        "weeks": int(work.prediction_date.nunique()),
        "selected_cells": len(selected), "positive_cells": positives,
        "true_positives": true,
        "precision": true / len(selected),
        "recall": true / positives if positives else None,
        "total_positive_exposure": total, "captured_exposure": captured,
        "exposure_weighted_recall": captured / total if total else None,
    }


def main():
    frame = load_fireguard_data()  # Verifies both saved bundle hashes below.
    grid, manifest = load_grid(), load_manifest()
    model = read_json("data/experiments/mvp-model/model.json")
    assert digest(ROOT / "data/experiments/mvp-model/model.json") == manifest["selection"]["model_sha256"]
    assert len(frame) == 98696 and frame.cell_id.nunique() == 949
    assert not frame.duplicated(["cell_id", "prediction_date"]).any()
    assert frame.groupby("prediction_date").size().eq(949).all()
    for name in ["feature_end_date", "fwi_source_date", "components_source_date"]:
        assert (pd.to_datetime(frame[name]) == pd.to_datetime(frame.prediction_date) - pd.Timedelta(days=1)).all()
    assert (pd.to_datetime(frame.outcome_end_date) == pd.to_datetime(frame.prediction_date) + pd.Timedelta(days=6)).all()
    totals = reference_totals(frame)
    environment = read_json("data/experiments/environment-logistic-v2/metrics.json")
    selected_metrics = read_json("data/experiments/mvp-model/metrics.json")
    boosted = read_json("data/experiments/boosted-tree-v1/metrics.json")
    method_map = {
        "Selected logistic": "previous_no_coordinates", "Coordinate logistic": "original_logistic",
        "Environmental logistic": "risk_score", "Prior detections 7d": "past_fire_7d",
        "Prior detections 30d": "past_fire_30d", "FWI": "fwi_baseline",
        "Historical frequency": "historical_frequency",
    }
    result = {
        "status": "passed", "verified_at": datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "source_code_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sorted(
            [ROOT / "app.py", ROOT / "data_loader.py", ROOT / "pyproject.toml", ROOT / "requirements.txt",
             *ROOT.glob("fireguard/*.py"), *ROOT.glob("tests/*.py")])},
        "bundle_hashes": manifest["bundle_hashes"], "model_sha256": manifest["selection"]["model_sha256"],
        "bundle": {"rows": len(frame), "cells": 949, "grid_cells": len(grid["features"]), "weeks": 104,
                   "columns": list(frame.columns)},
        "reference_exposure_totals": totals, "years": {},
        "feature_review": read_json("data/experiments/feature-review/metrics.json"),
        "environment_candidates": pd.read_csv(ROOT / "data/experiments/environment-logistic-v2/candidate_scores.csv").to_dict("records"),
        "coefficients": pd.read_csv(ROOT / "data/experiments/mvp-model/coefficients.csv").to_dict("records"),
    }
    checks, maximum_difference = 0, 0.
    for year in [2023, 2024]:
        part = frame.loc[frame.prediction_date.str.startswith(str(year))]
        ap = float(average_precision_score(part.y, part.risk_score))
        assert abs(ap - selected_metrics[str(year)]["average_precision"]) < 1e-12
        for mode, (column, _, _) in EXPOSURE_MODES.items():
            artifact_name = {"Population": "population", "Annual cropland": "annual_cropland",
                             "Closed-canopy oil palm": "closed_canopy_oil_palm"}[mode]
            for k in [10, 25, 50, 100]:
                for strategy in ["risk_only", "risk_times_exposure"]:
                    scores = part.risk_score if strategy == "risk_only" else part.risk_score * part[column]
                    recomputed = independent_allocation(part, scores, column, k)
                    saved = selected_metrics[str(year)]["allocation"][artifact_name][strategy][str(k)]
                    for name, value in recomputed.items():
                        delta = abs(value - saved[name]) if value is not None else 0.
                        assert (value is None) == (saved[name] is None)
                        assert delta <= max(1e-7, abs(value or 0) * 1e-12), (year, mode, k, strategy, name)
                        maximum_difference = max(maximum_difference, delta)
                    checks += 1
        methods = {label: environment[str(year)]["methods"][key] for label, key in method_map.items()}
        methods["Boosted tree"] = boosted[str(year)]["methods"]["risk_score"]
        combined = {}
        for k in [10, 25, 50, 100]:
            computed = evaluate(part, COMBINED_MODE, k, weights=DEFAULT_WEIGHTS, totals=totals)
            index = sum(DEFAULT_WEIGHTS[mode] * part[column] / totals[column]
                        for mode, (column, _, _) in EXPOSURE_MODES.items())
            work = part.assign(report_exposure=index)
            independent = independent_allocation(work, part.risk_score * index, "report_exposure", k)
            current = computed.set_index("method").loc["FireGuard × exposure"]
            for name in ["true_positives", "precision", "recall", "exposure_weighted_recall"]:
                assert abs(independent[name] - current[name]) < 1e-12
            combined[str(k)] = json.loads(computed.to_json(orient="records"))
        result["years"][str(year)] = {
            "rows": len(part), "positives": int(part.y.sum()), "positive_rate": float(part.y.mean()),
            "start": part.prediction_date.min(), "last_anchor": part.prediction_date.max(),
            "methods": methods, "selected_brier_recomputed": float(brier_score_loss(part.y, part.risk_score)),
            "no_prior_7d_detection": environment[str(year)]["no_prior_7d_detection"],
            "boosted_no_prior_7d_detection": boosted[str(year)]["no_prior_7d_detection"],
            "combined_balanced": combined,
        }
    week = frame.loc[frame.prediction_date == "2024-09-02"]
    plan = rank_week(week, COMBINED_MODE, 50, weights=DEFAULT_WEIGHTS, totals=totals)
    changed_outcomes = week.assign(y=1 - week.y)
    assert plan.cell_id.tolist() == rank_week(changed_outcomes, COMBINED_MODE, 50, weights=DEFAULT_WEIGHTS, totals=totals).cell_id.tolist()
    result["example_week"] = {"date": "2024-09-02", "K": 50,
        "top_five": json.loads(plan.head(5)[["cell_id", "risk_score", "exposure_index", "priority_score",
                          "population_2020", "annual_cropland_ha_2020", "closed_canopy_oil_palm_ha_2019"]].to_json(orient="records")),
        "policy_sensitivity": json.loads(policy_comparison(week, 50, totals=totals).to_json(orient="records"))}
    raw_path = ROOT / "data/probe/environment/weekly_environment.csv.gz"
    if raw_path.exists():
        assert digest(raw_path) == manifest["selection"]["input_sha256"]
        raw = pd.read_csv(raw_path)
        valid = raw.loc[np.isfinite(raw.population_2020) & raw.population_2020.ge(0) & raw.landscape_area_quality_ok]
        matrix = []
        dates = pd.to_datetime(valid.prediction_date)
        phase = 2 * np.pi * (dates.dt.dayofyear.to_numpy() - 1) / 365.2425
        for feature in model["feature_names"]:
            matrix.append(np.log1p(valid[feature[6:]].to_numpy()) if feature.startswith("log1p_") else
                          np.sin(phase) if feature == "annual_sin" else np.cos(phase))
        x = np.column_stack(matrix)
        z = (x - np.array(model["scaler"]["mean"])) / np.array(model["scaler"]["scale"])
        raw_score = z @ np.array(model["logistic"]["coefficients"]) + model["logistic"]["intercept"]
        calibrated = raw_score * model["calibrator"]["coefficient"] + model["calibrator"]["intercept"]
        predicted = valid[["cell_id", "prediction_date"]].assign(report_score=1 / (1 + np.exp(-np.clip(calibrated, -709, 709))))
        compare = frame.merge(predicted, on=["cell_id", "prediction_date"], validate="one_to_one")
        error = float(np.max(np.abs(compare.risk_score - compare.report_score)))
        assert len(compare) == len(frame) and error < 1e-12
        result["raw_score_check"] = {"status": "passed", "rows": len(compare), "maximum_absolute_error": error,
            "input_sha256": digest(raw_path), "raw_rows": len(raw), "raw_columns": len(raw.columns),
            "eligible_rows_before_purge": len(valid)}
    else:
        result["raw_score_check"] = {"status": "unavailable", "reason": "Ignored local source cache is required"}
    result["checks"] = {"saved_allocation_comparisons": checks, "maximum_allocation_difference": maximum_difference,
                        "combined_budget_comparisons": 8, "outcome_independent_example_ranking": True,
                        "bundle_hashes_match": True, "dates_and_unique_keys": True}
    target = Path(__file__).with_name("verification.json")
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(target), "checks": result["checks"],
                      "raw_score_check": result["raw_score_check"], "reference_totals": totals}, indent=2))


if __name__ == "__main__":
    main()
