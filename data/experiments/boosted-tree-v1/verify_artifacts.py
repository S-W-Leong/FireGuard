import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path('data/experiments/boosted-tree-v1')
read = lambda name: json.loads((folder / name).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
model, protocol, choice, results = [read(name) for name in ['model.json', 'protocol.json', 'recommendation.json', 'metrics.json']]
for path, expected in [(protocol['input'], protocol['input_sha256']), (protocol['incumbent'], protocol['incumbent_sha256']), (folder / 'model.json', choice['model_sha256']), (folder / 'protocol.json', model['protocol_sha256']), ('docs/FireGuard_Boosted_Tree_Protocol.md', protocol['human_protocol_sha256'])]:
    assert sha(path) == expected, f'hash mismatch: {path}'
assert len(model['trees']) == 200 and len(model['feature_names']) == 21, 'tree/feature counts'
raw = pd.read_csv(protocol['input'])
mask = np.isfinite(raw.population_2020) & (raw.population_2020 >= 0) & raw.landscape_area_quality_ok
eligible = raw.loc[mask].copy()
assert eligible.cell_id.nunique() == 949, 'cohort size'
weekly = pd.read_csv(folder / 'weekly_metrics.csv.gz', float_precision='round_trip')
exposures = {'population': 'population_2020', 'annual_cropland': 'annual_cropland_ha_2020', 'closed_canopy_oil_palm': 'closed_canopy_oil_palm_ha_2019'}
counts = {'scalar_tree_scores': 0, 'average_precision': 0, 'brier_scores': 0, 'weekly_selections': 0, 'pooled_metrics': 0, 'incumbent_scores': 0}
max_scalar_error = 0.0
for year in [2023, 2024]:
    pred = pd.read_csv(folder / f'predictions_{year}.csv.gz', float_precision='round_trip')
    part = eligible.loc[(pd.to_datetime(eligible.prediction_date).dt.year == year) & (pd.to_datetime(eligible.outcome_end_date) < pd.Timestamp(year + 1, 1, 1))]
    keys = ['cell_id', 'prediction_date']
    assert set(map(tuple, part[keys].to_numpy())) == set(map(tuple, pred[keys].to_numpy())), f'{year} row coverage'
    assert len(pred) == 49348 and pred.cell_id.nunique() == 949 and pred.prediction_date.nunique() == 52, f'{year} cohort counts'
    assert (pred.groupby('prediction_date').cell_id.nunique() == 949).all(), f'{year} weekly coverage'
    previous = pd.read_csv(f'data/experiments/environment-logistic-v2/predictions_{year}.csv.gz', float_precision='round_trip')
    joined = pred.merge(previous[keys + ['previous_no_coordinates']], on=keys, validate='one_to_one')
    assert np.max(np.abs(joined.incumbent_logistic - joined.previous_no_coordinates)) <= 1e-12, f'{year} incumbent reproduction'
    counts['incumbent_scores'] += len(joined)
    samples = pred.iloc[np.linspace(0, len(pred) - 1, 34).astype(int)].merge(raw, on=keys, validate='one_to_one', suffixes=('_saved', ''))
    for _, row in samples.iterrows():
        phase = 2 * np.pi * (pd.Timestamp(row.prediction_date).dayofyear - 1) / 365.2425
        values = [np.log1p(row[name[6:]]) if name.startswith('log1p_') else np.sin(phase) if name == 'annual_sin' else np.cos(phase) if name == 'annual_cos' else row[name] for name in model['feature_names']]
        logit = model['baseline_logit']
        for tree in model['trees']:
            node = 0
            while not tree['is_leaf'][node]:
                node = tree['left'][node] if values[tree['feature_idx'][node]] <= tree['threshold'][node] else tree['right'][node]
            logit += tree['value'][node]
        score = 1 / (1 + math.exp(-(logit * model['calibrator']['coefficient'] + model['calibrator']['intercept'])))
        error = abs(score - row.risk_score)
        assert error <= 1e-12, f'{year} scalar traversal: {row.cell_id} {row.prediction_date} error={error}'
        max_scalar_error = max(max_scalar_error, error)
        counts['scalar_tree_scores'] += 1
    saved_weekly = weekly.loc[weekly.year == year].set_index(['method', 'exposure', 'strategy', 'K', 'prediction_date'])
    assert saved_weekly.index.is_unique, f'{year} duplicate weekly metric keys'
    weeks = list(pred.groupby('prediction_date', sort=True))
    for method, entry in results[str(year)]['methods'].items():
        scores, labels = pred[method].to_numpy(), pred.y.to_numpy()
        order = np.argsort(-scores, kind='stable')
        sorted_scores, sorted_labels = scores[order], labels[order]
        ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
        true_positive = np.cumsum(sorted_labels)[ends]
        ap = float(np.sum(np.diff(np.r_[0, true_positive]) / labels.sum() * true_positive / (ends + 1)))
        assert math.isclose(ap, entry['average_precision'], abs_tol=1e-12), f'{year} {method} AP'
        counts['average_precision'] += 1
        if 'calibration' in entry:
            brier = float(np.mean((scores - labels) ** 2))
            assert math.isclose(brier, entry['calibration']['brier_score'], abs_tol=1e-12), f'{year} {method} Brier'
            counts['brier_scores'] += 1
        for exposure_name, column in exposures.items():
            for strategy in ['risk_only', 'risk_times_exposure']:
                for k in [10, 25, 50, 100]:
                    totals = np.zeros(5)
                    for date, week in weeks:
                        exposure, y = week[column].to_numpy(), week.y.to_numpy()
                        scores = week[method].to_numpy()
                        if strategy == 'risk_times_exposure': scores = scores * exposure
                        selected = np.lexsort((week.cell_id.to_numpy(), -scores))[:k]
                        ids = '|'.join(week.cell_id.to_numpy()[selected])
                        values = np.array([len(selected), y.sum(), y[selected].sum(), np.sum(y * exposure), np.sum(y[selected] * exposure[selected])])
                        saved = saved_weekly.loc[(method, exposure_name, strategy, k, date)]
                        assert ids == saved.selected_ids, f'{year} {method} {exposure_name} {strategy} K{k} {date}: ids'
                        assert np.allclose(values, saved[['selected_cells', 'positive_cells', 'true_positives', 'total_positive_exposure', 'captured_exposure']].to_numpy(dtype=float), rtol=1e-12, atol=1e-8), f'{year} {method} {date}: weekly metrics'
                        totals += values
                        counts['weekly_selections'] += 1
                    pooled = entry['allocation'][exposure_name][strategy][str(k)]
                    for name, value in zip(['selected_cells', 'positive_cells', 'true_positives', 'total_positive_exposure', 'captured_exposure'], totals):
                        assert math.isclose(value, pooled[name], rel_tol=1e-12, abs_tol=1e-8), f'{year} {method} {exposure_name} {strategy} K{k}: {name}'
                    for name, value in [('precision', totals[2] / totals[0]), ('recall', totals[2] / totals[1]), ('exposure_weighted_recall', totals[4] / totals[3])]:
                        assert math.isclose(value, pooled[name], abs_tol=1e-12), f'{year} {method} {exposure_name} {strategy} K{k}: {name}'
                    counts['pooled_metrics'] += 1
    print(f'{year}: independent coverage, scalar scoring, AP/calibration and all budget/exposure checks passed', flush=True)
methods = results['2023']['methods']
tree, logistic = methods['risk_score'], methods['incumbent_logistic']
adopt = tree['average_precision'] > logistic['average_precision'] and tree['allocation']['population']['risk_only']['50']['recall'] >= logistic['allocation']['population']['risk_only']['50']['recall']
assert adopt is False and choice['adopt_challenger'] is False, '2023 adoption gate'
assert choice['recommended_model'] == protocol['incumbent'], 'recommendation path'
assert counts['weekly_selections'] == len(weekly) == 14976, 'weekly row count'
verification = {'status': 'passed', 'checks': counts, 'max_scalar_score_error': max_scalar_error, 'input_and_incumbent_hashes_preserved': True, 'model_protocol_and_human_protocol_hashes_match': True, 'adoption_recomputed_from_2023_only': True, 'recommended_model': choice['recommended_model'], 'checker_source_sha256': sha(__file__)}
(folder / 'independent_verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False) + '\n')
print(json.dumps(verification, indent=2))
