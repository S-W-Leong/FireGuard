# FireGuard

Historical decision support for preventive fire monitoring in South Sumatra.
Choose a week, an exposure mode and a team budget. The dashboard ranks cells
by model risk × exposure and compares the selections with subsequent satellite
detections and the same-budget baselines.

Deployed streamlit app: https://fireguard-dormathon2026.streamlit.app

Slide deck: https://docs.google.com/presentation/d/1Ql-1rDRBUJ0d9HOS4g9CZxMMpah97w7f/edit?usp=sharing&ouid=113779798155208176962&rtpof=true&sd=true 

Full documentation: https://drive.google.com/file/d/1Jm4-6QpYkuTZGPY7-C1otvjOFN34Ef_M/view?usp=sharing 

## Run the Streamlit dashboard

Python 3.12 or newer:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Or with the project's existing uv environment:

```sh
uv sync --group dashboard --group dev
uv run --group dashboard streamlit run app.py
```

Open the local URL printed by Streamlit. The committed `data/dashboard/`
bundle includes both 2023 and 2024 and the clipped grid. Source downloads,
API credentials and model training are unnecessary at runtime. A public
deployment has not been created. For Streamlit hosting, the entrypoint is
`app.py` and the runtime dependencies are in `requirements.txt`.

## What the prototype does

- Combine population, annual cropland and closed-canopy oil palm with adjustable
  policy weights, or inspect each exposure separately.
- Inspect actual FWI, model risk, exposure and top-K priorities on the map.
- Review and download the weekly monitoring plan; inspect a cell's inputs.
- Compare selected-week or pooled yearly capture against FWI and fire-history
  baselines. Reveal subsequent detection cells explicitly in evaluation.
- Read the actual model features, source dates and evaluation limits.

Combined exposure starts at **50% population, 25% annual cropland and 25% oil
palm**. Each criterion is divided by its fixed total across the 949 eligible
cells before weighting. Three direct sliders control population, cropland and
oil palm. Moving one redistributes the remaining weight proportionally between
the other two, rounded to whole percentages; if both are zero, the remainder
is split equally. Weights always sum to 100%. These are declared
policy assumptions, not learned model parameters or percentages of teams.

The evaluation tab reports capture for each exposure on the same selected
plan and compares selected-week portfolios under people-focused, balanced and
agriculture-focused policies. The CSV records weights, normalization totals,
cell shares and weighted contributions. Cropland and oil palm remain distinct
measurements because their map dates differ and their areas can overlap.


## Model and data handoff

The dashboard currently uses the verified eight-feature logistic package in
`data/experiments/mvp-model/`. It reads historical predictions; it does not
fit models inside Streamlit. The bundle manifest records the model, source
hashes and prototype provenance. Runtime loading checks the bundled CSV and
geometry hashes.

To build a new bundle from locally cached, verified artifacts:

```sh
uv run python -m fireguard.dashboard_bundle --output /private/tmp/fireguard-dashboard-new
```

The builder refuses to overwrite an existing bundle. `--model-package` accepts
a verified selected model package; a boosted challenger must also be verified
and recommended by its development gate before it can be bundled. Review the
new bundle before replacing the deployed files and restart the app afterward.


## Verify

```sh
uv run --group dashboard --group dev pytest
```

Tests check separate exposure units, deterministic budgets, outcome-independent
selection, pooled capture, handoff dates, recorded metrics and real Streamlit
control interactions. Browser verification additionally checks map rendering.

This is a **historical risk-ranking demonstration**. Base training uses 2022;
2023 supplies development and calibration; 2024 is exploratory. Static map
years differ. Risk scores are not verified operational probabilities and
exposure capture does not establish prevented fires, affected people or losses.
See the public model methodology and results under `docs/`.
