# FireGuard boosted tree comparison results

11 October 2026. Keep the existing logistic model for the MVP. The one fixed
boosted tree challenger did not pass the adoption rule agreed before fitting:
higher 2023 average precision (AP) and at least the logistic model's risk-only
Recall@50. The challenger has been trained, evaluated and exported separately.
The dashboard can continue using the current model package.

The tree improved several exploratory 2024 results, especially agricultural
prioritization. Those gains are useful evidence for later work, but choosing it
because of 2024 would use a year that earlier project decisions already examined.
This completes the authorized comparison; no further tuning is needed for the
demo.

## Experiment and inputs

One HistGradientBoostingClassifier configuration was fitted: 200 iterations,
learning rate 0.05, at most 15 leaves per tree, at least 100 observations per leaf
and L2 regularization 10. Automatic early stopping and class weighting were
disabled; seed 17 and two native threads were used. There was no parameter grid.
The [protocol](FireGuard_Boosted_Tree_Protocol.md) was saved before fitting.

The tree uses the existing 21 environmental inputs: preceding FWI summaries,
prior detection counts, FFMC and DC, measured peat and land-cover/oil-palm
fractions, and annual sine/cosine. Coordinates, cell identity, population and
exposure hectares are excluded from fitting. The incumbent logistic model has
eight inputs: four FWI summaries, two prior detection counts and two seasonal
terms. This compares two complete model packages; it does not isolate the effect
of individual inputs or the algorithm alone.

The tree was trained on 48,399 observations from 949 cells in 2022, with 1,020
positive labels. The final cross-year outcome week was purged. A sigmoid
calibrator was fitted on 2023, with no base-model refit. Both models were
evaluated on exactly the same 949 cells, 52 weeks and 49,348 observations in
each evaluation year. The incumbent's original 2022 training cohort had 951
cells; its frozen model and calibration were preserved for this comparison.

The adoption decision was recorded using 2023 before evaluating 2024. The
selection year also supplies calibration data, so its calibration results are
development results. **2024 remains exploratory, not a fresh independent test.**

## Occurrence ranking and weekly monitoring

AP describes precision across recall levels; it is not classification accuracy.
Recall@50 is the share of positive cell-weeks identified when selecting the 50
highest-risk cells every week. Higher values are better for both metrics.

| Model | 2023 development AP | 2023 Recall@50 | 2024 exploratory AP | 2024 Recall@50 |
|---|---:|---:|---:|---:|
| Current logistic | 0.482 | 19.2% | 0.221 | 21.5% |
| Boosted tree challenger | 0.388 | 16.7% | 0.225 | 22.7% |
| Prior 30-day detection counts | 0.375 | 19.0% | 0.171 | 21.3% |
| FWI alone | 0.262 | 6.3% | 0.114 | 6.8% |

The tree fails both 2023 adoption conditions: it selects 644 positive cell-weeks
at this budget versus 743 for logistic regression. In 2024 it selects 496 versus
471, an additional 25 across the year. There are 2,600 selections per year and
2,187 positive cell-weeks in 2024; these counts describe cell-weeks, not distinct
fires. The 2024 gain is modest, and this experiment does not establish its
statistical reliability.

On observations with no preceding seven-day detection, 2024 AP is 0.119 for the
tree, 0.111 for logistic regression and 0.080 for FWI. This checks the influence
of ongoing detections; it does not establish prediction of new ignitions.

## Population and agricultural priorities

Each policy below ranks model score multiplied by one exposure category and
selects 50 cells per week. The metric is the share of exposure associated with
positive cell-weeks captured by those selections. People and hectares remain
separate; cropland and oil palm are not added into a combined agricultural area.

| Exposure | 2023 logistic | 2023 tree | 2024 logistic | 2024 tree |
|---|---:|---:|---:|---:|
| Population | 48.1% | 49.5% | 47.4% | 48.8% |
| Annual cropland | 47.5% | 47.6% | 39.2% | 42.7% |
| Closed-canopy oil palm | 33.3% | 38.2% | 30.8% | 36.6% |

The tree has better exposure-weighted allocation in both years despite its
worse 2023 occurrence ranking. The strongest exploratory gain is oil palm,
about 5.8 percentage points. These are real tradeoffs in the saved results, but
the adoption rule was based on occurrence AP and risk-only Recall@50.

Exposure refers to assets inside cells with qualifying satellite detections.
These metrics do not measure burned hectares, people harmed or losses avoided.
Annual cropland is mapped for 2020 and closed-canopy oil palm for 2019; young or
sparse palms and other plantation commodities remain outside that coverage.

## Calibration and interpretation

In 2024, the tree's mean score is 6.26%, versus 5.31% for logistic regression and
an observed positive rate of 4.43%. Its Brier score is also worse: 0.03928 versus
0.03847, where lower is better. Use relative risk language in the dashboard.
Neither exported model is a verified operational probability or a verified
live seven-day forecast; the weather inputs are retrospective reanalysis.

Only one base-model training year is available. A later reserved period and
verified issued weather forecasts would be needed to assess operational use.
The target remains a qualifying vegetation-fire satellite detection during
the next seven days, which includes agricultural and continuing burning.

## Verification and handoff

All 21 repository tests passed. They include threshold-boundary traversal,
export equivalence on unseen rows, chronology checks, preservation of the
incumbent and a label-flipping check showing that 2024 labels cannot change the
tree fit or adoption decision. Portable JSON inference matched sklearn on all
98,696 evaluation observations, with maximum absolute error 0.0.

An independent artifact check also passed: 68 scalar tree traversals, 12 AP
calculations, six Brier calculations, all 14,976 weekly selections and 288
pooled budget/exposure comparisons. It reproduced the incumbent's scores on
all evaluation rows and checked input, model and protocol hashes.

| Use | Artifact |
|---|---|
| Dashboard default model | [Current logistic model](../data/experiments/mvp-model/model.json) |
| Dashboard historical snapshot | [Current model snapshot](../data/experiments/mvp-model/historical_demo_cells.csv) |
| Challenger decision | [Recommendation](../data/experiments/boosted-tree-v1/recommendation.json) |
| Full challenger metrics | [Metrics](../data/experiments/boosted-tree-v1/metrics.json) |
| Challenger numeric trees | [Model](../data/experiments/boosted-tree-v1/model.json) |
| Training/export check | [Verification](../data/experiments/boosted-tree-v1/verification.json) |
| Independent artifact check | [Independent verification](../data/experiments/boosted-tree-v1/independent_verification.json) |

The input dataset and current model hashes are unchanged. The recorded
[recommendation](../data/experiments/boosted-tree-v1/recommendation.json) retains
the logistic model used by the historical monitoring dashboard.

To reproduce the fit, run the following from the repository root, supplying a
new output directory:

```sh
LOKY_MAX_CPU_COUNT=2 .venv/bin/python -m fireguard.boosted_model --output <new-directory>
```

An existing experiment directory is protected from overwrite. To recheck the
saved artifacts without fitting, run
`.venv/bin/python data/experiments/boosted-tree-v1/verify_artifacts.py`
from the repository root.
Model parameters and software versions are saved in the experiment protocol.
The implementation follows the installed version's
[official classifier documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
