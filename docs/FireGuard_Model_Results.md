# FireGuard model choice and results

Use the logistic model without coordinates for the MVP. The model with peat,
land cover, oil-palm fractions, FFMC and DC performed worse on the 2023
development comparison. The simpler model has been retrained, reproduced
exactly and exported. Population, cropland and oil-palm exposure remain available
for monitoring priorities.

The next project step is the dashboard: a historical risk map, weekly monitoring
budget, separate exposure priorities and explanations. Further model experiments
can wait until the demo works.

## Selected predictors and training

The selected model has eight inputs:

- FWI on the preceding day, plus its preceding 7-day mean, 7-day maximum and
  30-day mean.
- Qualifying satellite detection counts during the preceding 7 and 30 days.
- Annual sine and cosine terms to represent season.

FWI and counts use log1p transforms. Standardization is fitted on 2022 only.
Coordinates, cell identity and population do not enter the classifier. The
environmental candidate added 13 measured predictors; those additions were
tested together and were not retained. This comparison does not establish that
each added variable is individually useless.

The selected base model was reproduced on its original 951-cell 2022 cohort:
48,501 rows, 51 weeks and 1,020 positive observations, with C=10. The final 2022
week was excluded because its outcome reached 2023. Its scaler and fitted
parameters match the frozen earlier model exactly. Sigmoid calibration uses
2023. All comparisons below use the same 949 cells with valid population and
landscape coverage, or 49,348 observations per evaluation year. The new
environmental model was also trained on those 949 cells. Exclusion IDs are
saved in the experiment protocol.

Weather, prior fire activity and landscape variables are motivated by regional
fire research, including [ProbFire](https://nhess.copernicus.org/articles/22/303/2022/).
Our weekly target, calendar encoding and rolling summaries are project choices;
this experiment does not reproduce ProbFire's monthly forecasting model.

## Occurrence ranking comparison

Average precision summarizes the precision-recall ranking; higher is better.
2023 selected the model. **2024 is exploratory**, because earlier project
decisions had already examined that year. It is not fresh independent evidence.

| Method | 2023 development AP | 2024 exploratory AP |
|---|---:|---:|
| Selected logistic model without coordinates | 0.482 | 0.221 |
| Earlier logistic model with coordinates | 0.437 | 0.210 |
| New environmental logistic model | 0.403 | 0.200 |
| Prior 30-day detections | 0.375 | 0.171 |
| FWI alone | 0.262 | 0.114 |
| 2022 historical positive frequency | 0.130 | 0.098 |

Selecting the 50 highest-risk cells each week identifies 21.5% of the 2024
fire-positive cell-weeks, compared with 21.3% for prior 30-day detections and
6.8% for FWI. The advantage over recent fire history at this budget is small:
471 positives selected versus 466 across the year. A higher overall AP does not
guarantee a large improvement at every monitoring budget.

Among observations with no preceding 7-day detection, selected-model AP is
0.111 versus 0.080 for FWI. This sensitivity check reduces the contribution of
ongoing detections, but it does not establish new-ignition prediction.

## Monitoring priorities and agricultural exposure

For each exposure category, rank risk score multiplied by that category's
exposure, then select 50 cells per week. Every baseline receives the same
exposure data. The table reports the share of exposure associated with
fire-positive cells covered by the selections across 2024.

| Exposure category | Selected ML | FWI | Historical frequency | Prior 30-day detections |
|---|---:|---:|---:|---:|
| Population | 47.4% | 39.4% | 43.7% | 34.6% |
| Annual cropland | 39.2% | 31.5% | 38.1% | 32.3% |
| Closed-canopy oil palm | 30.8% | 16.6% | 34.1% | 33.3% |

The ML policy improves these population and annual-cropland comparisons. For
oil palm, historical frequency and recent detections perform better. The demo
must preserve that result rather than claim that ML wins for every asset type.

Population is measured in estimated people; annual cropland and oil palm are
measured in hectares. They are separate priority modes. We have not invented
weights to add them together. Cropland and oil palm also come from different
mapping dates and may overlap, so their hectares are not a unique combined
agricultural area. Young or sparse oil palm and other plantation commodities
remain outside this agricultural coverage.

These metrics concern assets located in cells with a qualifying detection. They
do not measure hectares burned, people harmed, smoke exposure or losses avoided.

## Scores and forecast limits

The selected model's mean 2024 score is 5.31%, while the observed positive rate
is 4.43%. Use relative risk language in the UI, not precise probability claims.
2023 selection and calibration share data. A new reserved period is needed for
independent evaluation.

The inputs are historical reanalysis and static maps. Publication delays and
live issued-forecast access remain unverified. The app should demonstrate a
historical planning workflow, with explicit dates, until that access is resolved.
The target is any qualifying VIIRS S-NPP detection in the next seven days;
it includes agricultural burning and persistent fires. An observed negative
means no qualifying detection, rather than proven absence of fire.

## Saved model and verification

The selected package is [mvp-model](../data/experiments/mvp-model/selection.json).
It contains model parameters, calibration, coefficients, weekly predictions,
metrics, a dated historical snapshot and verification results. The full
environmental comparison is retained in `data/experiments/environment-logistic-v2`.
The [protocol](FireGuard_Model_Protocol.md) and
[selection rule](FireGuard_Model_Selection.md) record the modelling decisions.

All 15 current tests pass. Independent calculations checked 14 AP summaries,
17,472 weekly selections and 98,696 exported predictions. The greatest exported
score discrepancy is below 5e-16. Original data and model hashes remain
unchanged; all 149,760 original weekly rows and columns are preserved. An
additional 36 direct NetCDF component checks passed.

Reproduce the experiment from the cached joined inputs using fresh output
directories:

```sh
.venv/bin/python -m fireguard.train_environment --output data/experiments/environment-reproduction
.venv/bin/python -m fireguard.select_mvp --experiment data/experiments/environment-reproduction --output data/experiments/mvp-reproduction
.venv/bin/python -m pytest -q
```

Score a table containing the named raw inputs and a prediction date:

```python
import json
from pathlib import Path
import pandas as pd
from fireguard.select_mvp import score_model

model = json.loads(Path("data/experiments/mvp-model/model.json").read_text())
cells = pd.read_csv("data/experiments/mvp-model/historical_demo_cells.csv")
cells["risk_score"] = score_model(cells, model)
```
