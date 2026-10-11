# Environmental logistic model: protocol frozen before fitting

11 October 2026. The user requested one promising model to save time. The earlier
2023 comparison favoured a logistic model without coordinates (average precision
0.482 versus 0.437 with coordinates). We will use that family, extend its inputs,
and run no algorithm or feature search after viewing 2024 results.

## Inputs and target

- Input: `data/probe/environment/weekly_environment.csv.gz`.
- Target: at least one qualifying VIIRS S-NPP detection in a cell during the
  following seven local calendar days. This includes persistent fires and
  agricultural burning; it is not a new-ignition or damaging-wildfire label.
- Weather/history: log1p FWI lag 1, mean 7, max 7, mean 30; log1p prior 7/30-day
  detection counts; raw FFMC lag 1; log1p DC lag 1.
- Landscape: peat fraction; WorldCover shrub, grass, annual cropland, built,
  bare, water, herbaceous wetland and mangrove fractions. Tree cover is the
  reference class; snow/moss are absent. Industrial/smallholder closed-canopy
  oil-palm fractions add plantation information.
- Calendar: annual sine/cosine, using day of year and period 365.2425 days,
  retained from the previously strongest development model.
- Latitude, longitude, cell identity and population are excluded from fitting.
- Cropland fraction describes land cover for occurrence prediction; cropland
  hectares describe agricultural exposure for allocation.

The predictor categories are informed by the [regional FDRS study](https://doi.org/10.1007/s11027-006-9043-8)
and [ProbFire](https://nhess.copernicus.org/articles/22/303/2022/), which combines
weather, fire history, peat and forest information. This is an adaptation to our
weekly target and available measurements, not a reproduction of those models.
Calendar coding and rolling summaries are project modelling choices.

## Fixed cohort and chronology

Use 949 cells with finite nonnegative population and a passing landscape area
quality flag. Retain the exclusion list. Every model and baseline uses this same
cohort. Reject missing predictors, invalid labels, duplicate keys, incomplete
weeks and weather/history dates that do not precede prediction dates. No
imputation. Validate static fractions and coverage before fitting.

Train on 2022; remove any training outcome window reaching 2023. Select C from
0.01, 0.1, 1, 10 by 2023 average precision, with smaller C breaking exact ties.
StandardScaler is fitted only on 2022. Use L2 logistic regression, lbfgs,
max_iter=1000, tol=1e-6, no class weighting, seed 17. A sigmoid calibrator with
C=1e6 is fitted on selected model logits and 2023 labels; do not refit the base
model. Selection and calibration share 2023, so its scores are development
results. Treat 2024 as exploratory temporal evaluation because previous project
decisions have already seen 2024 results. Reserve a new period for future
independent validation.

## Evaluation and export

Compare FWI lag 1, prior detections 7/30 days, smoothed 2022 positive frequency
(Beta(1,1)), the frozen coordinate-based original model and the frozen earlier
model without coordinates. Evaluate average precision (AP), Brier score and
calibration bins where scores are probability-like. AP is the implementation's
precision-recall summary, not trapezoidal PR area.

Select top K independently each week for K=10,25,50,100. Sort score descending
then cell_id ascending; labels cannot break ties. Report pooled precision,
recall and exposure-weighted recall. For each of population, annual cropland
and closed-canopy oil palm, compare risk-only and risk-times-exposure strategies
with every baseline receiving the same exposure. Do not add hectares and people
or combine crop/palm into a claimed unique agricultural area. No policy weights
are invented. Report the no-prior-7-day-detection subset separately as a
persistence sensitivity check, not as verified new ignitions.

Save source hashes, protocol, software versions, candidate scores, scaler,
coefficients, calibrator, predictions and weekly metrics. Verify exported
predictions independently against sklearn, and preserve earlier model artifacts.
Run tests for chronology, feature leakage, invalid inputs, deterministic ties,
weekly budgets, exposure units and export equivalence.

Historical reanalysis publication timing remains unverified; display risk scores,
not verified operational forecasts or guaranteed probabilities. Static maps have
different dates. WorldCover tree cover includes plantations, oil palm excludes
young/sparse plantations, and these layers do not cover all agricultural assets.
