"""One fixed boosted-tree challenger and portable numeric-tree inference."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from .revised_model import TARGET, cohort, features, split_year, validate_time
from .select_mvp import score_model
from .train_environment import EXPOSURES, calibration_summary, evaluate, sha256, write_json

PARAMETERS = {"loss": "log_loss", "learning_rate": 0.05, "max_iter": 200,
    "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 10.,
    "early_stopping": False, "class_weight": None, "categorical_features": None, "random_state": 17}


def fit_challenger(train: pd.DataFrame, validation: pd.DataFrame):
    if not (pd.to_datetime(train.prediction_date).dt.year == 2022).all():
        raise ValueError("training year must be 2022")
    if not (pd.to_datetime(validation.prediction_date).dt.year == 2023).all():
        raise ValueError("validation year must be 2023")
    if (pd.to_datetime(train.outcome_end_date) >= pd.Timestamp("2023-01-01")).any():
        raise ValueError("training outcomes must end before validation year")
    if train[TARGET].nunique() != 2 or validation[TARGET].nunique() != 2:
        raise ValueError("training and calibration need both classes")
    with threadpool_limits(limits=2):
        fitted = HistGradientBoostingClassifier(**PARAMETERS).fit(features(train), train[TARGET])
        logits = fitted.decision_function(features(validation)).reshape(-1, 1)
        calibrator = LogisticRegression(C=1e6, max_iter=1000, tol=1e-6, random_state=17).fit(logits, validation[TARGET])
    if calibrator.n_iter_.max() >= 1000 or calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("calibration failed or reversed the tree ranking")
    return fitted, calibrator


def export_trees(fitted, calibrator, names: list[str]) -> dict:
    if fitted.n_trees_per_iteration_ != 1 or len(names) != fitted.n_features_in_:
        raise ValueError("export supports binary numeric features only")
    trees = []
    for iteration in fitted._predictors:
        nodes = iteration[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits cannot use numeric export")
        trees.append({"feature_idx": nodes["feature_idx"].astype(int).tolist(),
            "threshold": nodes["num_threshold"].tolist(), "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(), "is_leaf": nodes["is_leaf"].astype(bool).tolist(),
            "value": nodes["value"].tolist()})
    # sklearn already includes learning_rate in the stored leaf values.
    return {"format_version": "fireguard-numeric-boosted-trees-1", "feature_names": names,
        "baseline_logit": float(fitted._baseline_prediction[0, 0]), "trees": trees,
        "calibrator": {"coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])},
        "parameters": fitted.get_params(), "tree_count": len(trees),
        "feature_transform": "fireguard.revised_model.features: log1p named FWI/count/DC inputs; raw fractions/FFMC; annual sine/cosine",
        "score_interpretation": "relative risk; retrospective reanalysis; calibration may shift between years"}


def portable_tree_score(matrix: pd.DataFrame, model: dict) -> np.ndarray:
    values = matrix[model["feature_names"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("tree inputs must be finite; imputation is disabled")
    n = len(values)
    raw = np.full(n, model["baseline_logit"], dtype=float)
    for tree in model["trees"]:
        feature = np.asarray(tree["feature_idx"], dtype=int)
        threshold = np.asarray(tree["threshold"], dtype=float)
        left, right = np.asarray(tree["left"], dtype=int), np.asarray(tree["right"], dtype=int)
        leaf, value = np.asarray(tree["is_leaf"], dtype=bool), np.asarray(tree["value"], dtype=float)
        node_index, active = np.zeros(n, dtype=int), np.arange(n)
        for _ in range(len(leaf) + 1):
            if not len(active):
                break
            current = node_index[active]
            finished = leaf[current]
            raw[active[finished]] += value[current[finished]]
            active, current = active[~finished], current[~finished]
            goes_left = values[active, feature[current]] <= threshold[current]
            node_index[active] = np.where(goes_left, left[current], right[current])
        else:
            raise ValueError("tree contains a traversal cycle")
    calibration = model["calibrator"]
    logit = raw * calibration["coefficient"] + calibration["intercept"]
    return 1 / (1 + np.exp(-np.clip(logit, -709, 709)))


def adoption_gate(challenger: dict, incumbent: dict) -> bool:
    def recall(entry):
        return entry["allocation"]["population"]["risk_only"]["50"]["recall"]
    return bool(challenger["average_precision"] > incumbent["average_precision"]
                and recall(challenger) >= recall(incumbent))


def run_challenger(input_path: Path, incumbent_path: Path, output_path: Path) -> dict:
    input_path, incumbent_path, output_path = map(Path, [input_path, incumbent_path, output_path])
    if output_path.exists():
        raise FileExistsError(f"preserve existing experiment: {output_path}")
    source_hash, incumbent_hash = sha256(input_path), sha256(incumbent_path)
    incumbent = json.loads(incumbent_path.read_text())
    frame, excluded = cohort(validate_time(pd.read_csv(input_path)))
    features(frame)
    for column in EXPOSURES.values():
        if not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
            raise ValueError(f"invalid exposure {column}")
    train, validation = [split_year(frame, year) for year in [2022, 2023]]
    output_path.mkdir(parents=True)
    protocol = {"experiment": "boosted-tree-v1", "input": str(input_path), "input_sha256": source_hash,
        "incumbent": str(incumbent_path), "incumbent_sha256": incumbent_hash,
        "parameters": PARAMETERS, "feature_names": list(features(train).columns), "native_threads": 2,
        "train_year": 2022, "selection_and_calibration_year": 2023, "exploratory_year": 2024,
        "adoption_gate": "higher 2023 AP AND noninferior 2023 risk-only Recall@50",
        "hyperparameter_search": False, "no_base_model_refit": True, "no_imputation": True,
        "cohort_cells": frame.cell_id.nunique(), "excluded_cells": excluded,
        "training_rows": len(train), "training_positives": int(train[TARGET].sum()),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    protocol_doc = Path("docs/FireGuard_Boosted_Tree_Protocol.md")
    if protocol_doc.exists():
        protocol["human_protocol_sha256"] = sha256(protocol_doc)
    write_json(output_path / "protocol.json", protocol)
    print("Fitting one fixed boosted-tree model on 2022...", flush=True)
    fitted, calibrator = fit_challenger(train, validation)
    model = export_trees(fitted, calibrator, list(features(train).columns))
    model["protocol_sha256"] = sha256(output_path / "protocol.json")
    write_json(output_path / "model.json", model)
    frequency = train.groupby("cell_id")[TARGET].agg(["sum", "count"])
    frequency = (frequency["sum"] + 1) / (frequency["count"] + 2)
    methods = ["risk_score", "incumbent_logistic", "fwi_baseline", "past_fire_7d", "past_fire_30d", "historical_frequency"]
    results, all_weekly, max_error, rows_verified = {}, [], 0., 0
    recommendation = None
    for year in [2023, 2024]:
        part = split_year(frame, year)
        matrix = features(part)
        with threadpool_limits(limits=2):
            logits = fitted.decision_function(matrix)
            raw_probability = fitted.predict_proba(matrix)[:, 1]
        predicted = part[["cell_id", "prediction_date", "outcome_end_date", "feature_end_date",
            "fwi_source_date", "components_source_date", TARGET, "past_fire_detections_7d",
            *EXPOSURES.values()]].copy().rename(columns={TARGET: "y"})
        predicted["risk_score"] = calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        predicted["raw_tree_score"] = raw_probability
        predicted["incumbent_logistic"] = score_model(part, incumbent)
        predicted["fwi_baseline"] = part.fwi_lag1
        predicted["past_fire_7d"] = part.past_fire_detections_7d
        predicted["past_fire_30d"] = part.past_fire_detections_30d
        predicted["historical_frequency"] = part.cell_id.map(frequency)
        exported_scores = portable_tree_score(matrix, model)
        max_error = max(max_error, float(np.max(np.abs(exported_scores - predicted.risk_score))))
        rows_verified += len(part)
        metrics, weekly = evaluate(predicted, methods)
        metrics["incumbent_logistic"]["calibration"] = calibration_summary(predicted.y, predicted.incumbent_logistic)
        weekly["year"] = year
        all_weekly.append(weekly)
        subset = predicted.loc[predicted.past_fire_detections_7d == 0]
        results[str(year)] = {"rows": len(predicted), "positives": int(predicted.y.sum()), "methods": metrics,
            "no_prior_7d_detection": {"rows": len(subset), "positives": int(subset.y.sum()),
            "average_precision": {name: float(average_precision_score(subset.y, subset[name])) if subset.y.sum() > 0 else None for name in methods}}}
        if year == 2023:
            recommendation = {"adopt_challenger": adoption_gate(metrics["risk_score"], metrics["incumbent_logistic"]),
                "selection_year": 2023, "selection_rule": protocol["adoption_gate"],
                "challenger_development_AP": metrics["risk_score"]["average_precision"],
                "incumbent_development_AP": metrics["incumbent_logistic"]["average_precision"],
                "challenger_development_Recall50": metrics["risk_score"]["allocation"]["population"]["risk_only"]["50"]["recall"],
                "incumbent_development_Recall50": metrics["incumbent_logistic"]["allocation"]["population"]["risk_only"]["50"]["recall"],
                "incumbent_artifact_preserved": str(incumbent_path), "challenger_model": str(output_path / "model.json"),
                "status": "development_choice_pending_export_verification"}
            write_json(output_path / "recommendation.json", recommendation)
        predicted.to_csv(output_path / f"predictions_{year}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        print(f"{year} AP: tree={metrics['risk_score']['average_precision']:.4f}; logistic={metrics['incumbent_logistic']['average_precision']:.4f}", flush=True)
        if year == 2024:
            snapshot = part.loc[part.prediction_date == part.prediction_date.max()].copy()
            snapshot["risk_score"] = portable_tree_score(features(snapshot), model)
            snapshot["snapshot_type"] = "historical demonstration; " + snapshot.prediction_date.max().strftime("%Y-%m-%d")
            snapshot.to_csv(output_path / "historical_demo_cells.csv", index=False)
    pd.concat(all_weekly, ignore_index=True).to_csv(output_path / "weekly_metrics.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    write_json(output_path / "metrics.json", results)
    if max_error > 1e-12 or sha256(input_path) != source_hash or sha256(incumbent_path) != incumbent_hash:
        raise RuntimeError("export mismatch or inputs/ incumbent changed")
    write_json(output_path / "verification.json", {"status": "passed", "portable_max_error": max_error,
        "portable_rows_verified": rows_verified, "input_hash_unchanged": True, "incumbent_hash_unchanged": True,
        "cohort_cells": frame.cell_id.nunique(), "tree_count": len(model["trees"]),
        "adoption_uses_2023_only": True, "2024_status": "exploratory; previously examined"})
    recommendation["status"] = "verified"
    recommendation["model_sha256"] = sha256(output_path / "model.json")
    recommendation["recommended_model"] = str(output_path / "model.json") if recommendation["adopt_challenger"] else str(incumbent_path)
    write_json(output_path / "recommendation.json", recommendation)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/probe/environment/weekly_environment.csv.gz"))
    parser.add_argument("--incumbent", type=Path, default=Path("data/experiments/mvp-model/model.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/boosted-tree-v1"))
    args = parser.parse_args()
    run_challenger(args.input, args.incumbent, args.output)


if __name__ == "__main__":
    main()
