# FireGuard

Historical decision support for preventive fire monitoring in South Sumatra.
Choose a week, one exposure type and a team budget. The dashboard ranks cells
by model risk × exposure and compares the selections with subsequent satellite
detections and the same-budget baselines.

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

- Select population, annual cropland or closed-canopy oil palm separately.
- Inspect actual FWI, model risk, exposure and top-K priorities on the map.
- Review and download the weekly monitoring plan; inspect a cell's inputs.
- Compare selected-week or pooled yearly capture against FWI and fire-history
  baselines. Reveal subsequent detection cells explicitly in evaluation.
- Read the actual model features, source dates and evaluation limits.


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
