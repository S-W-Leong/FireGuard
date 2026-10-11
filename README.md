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
API credentials and model training are unnecessary at runtime. For Streamlit hosting, the entrypoint is
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

## Satellite image comparison and experimentation

FireGuard is a local satellite comparison prototype for investigating landscape changes around reported fires in South Sumatra. It retrieves Sentinel-2 imagery through Sentinel Hub and displays the same area before and after a selected period.

The app includes Before, After, Changes, and Split View. Actual acquisition dates and estimated clear coverage are shown for each image. An optional U-Net overlay highlights experimental change candidates; it is hidden by default and has **not been validated for wildfire detection or damage assessment**.

### Run locally

Requirements: Git, Python 3.12, an internet connection, and Sentinel Hub client credentials with access to Sentinel-2 L2A and the Statistical and Process APIs. API calls use your Sentinel Hub account's allowance. The first dependency installation downloads TensorFlow and may take several minutes.

```sh
git clone https://github.com/S-W-Leong/FireGuard.git
cd FireGuard/satellite_app
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set both values:

```dotenv
SENTINEL_CLIENT_ID=your-client-id
SENTINEL_CLIENT_SECRET=your-client-secret
```

Then start the app:

```sh
./run-local.sh
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/). To use another port, run `./run-local.sh --port 8766` and open the corresponding address. Stop the server with Ctrl+C.

The server binds to the local machine. Credentials stay on the server, and `.env`, generated imagery, and the virtual environment are ignored by Git. Environment variables named `SENTINEL_CLIENT_ID` and `SENTINEL_CLIENT_SECRET` override `.env` values. Credentials are not bundled; each collaborator supplies their own configuration.

### Compare imagery

1. Select an area or enter a bounding box in `[west, south, east, north]` order using WGS 84 longitude and latitude. Each side must span at most 0.25 degrees.
2. Choose non-overlapping before and after windows, each at most 63 days. The app selects the day with the highest estimated clear coverage in each window. The displayed acquisition date may differ from the window endpoints.
3. Click **Load satellite pair**, then inspect **Before**, **After**, or **Split View**. **Recent imagery** fills recent date windows; click Load to retrieve them.
4. To inspect the model output, choose **Changes** and enable **Show experimental U-Net output**.

Clouds remain visible in the natural-color images. Model results are limited to pixels classified as clear in both observations. This filtering does not establish that a prediction is correct: cloud edges, seasonal differences, registration, and unfamiliar terrain can still affect results. The displayed flagged percentage refers to jointly clear model pixels, not confirmed burned area.

### Historical 2026 fire reference

An optional comparison covers an area near a reported OKI fire. Indonesia's environment ministry reports fire activity on **4–8 August 2026** in the [PT Bintang Harapan Palma concession](https://www.kemenlh.go.id/news/detail/dampak-kebakaran-lahan-klhbplh-lakukan-penyegelan-di-konsesi-pt-bintang-harapan-palma).

Use these settings to request the observations previously inspected in the prototype:

| Setting | Value |
| --- | --- |
| Bounding box | `[105.2876675, -3.130584, 105.3376675, -3.080584]` |
| Before From / To | `2026-07-12` / `2026-07-12` |
| After From / To | `2026-08-19` / `2026-08-19` |

These are historical reference dates, not 11 October imagery. A report about a fire in the area does not provide pixel-level labels for the U-Net or prove that every visible difference was caused by that fire. Availability still depends on the provider and account access.

### Included model weights

The full pretrained artifact is bundled at [`satellite_app/models/unet_model.h5`](satellite_app/models/unet_model.h5), so a normal clone includes the weights. No retraining or separate model download is needed to run inference. See the [model notes](satellite_app/models/README.md) for the checksum, input contract, and limitations.

The app independently stretches each raw RGB image to its 2nd–98th percentile range, resizes it to 128 × 128, and stacks before RGB followed by after RGB. The model emits one change probability per pixel. Predictions above 0.5 are shown only where both dates pass the clear-pixel mask. A probability is an uncalibrated model output, not a measured probability of fire.

### Root experimentation notebook

[`Satellite Experimentation.ipynb`](Satellite%20Experimentation.ipynb) consolidates the satellite work into one research entry point at the repository root:

- Original LEVIR-CD dataset loading/visualization, baseline CNN, Siamese U-Net-inspired model, and pretrained ChangeFormer code, saved outputs, and paper credits.
- Experiment summaries, metric interpretation, and known reproducibility issues.
- Sentinel-2 acquisition/clear-coverage workflow and the historical OKI comparison settings above.
- Local feasibility-report inspection, bundled artifact verification, cached image comparison, preprocessing, and opt-in U-Net inference.

Use JupyterLab with the satellite environment to run the new local workflow cells:

```sh
satellite_app/.venv/bin/python -m pip install jupyterlab matplotlib
satellite_app/.venv/bin/python -m jupyter lab
```

Open `Satellite Experimentation.ipynb` from the repository root. The new workflow cells make no provider requests; inference defaults to disabled. Retrieve a pair through the viewer first to populate its ignored local cache. The preserved PyTorch research section needs a separate compatible environment, dataset paths, and checkpoint repairs; **Run All is not a supported end-to-end workflow**. Existing saved research outputs have not been freshly reproduced.

The notebook does not train/export the viewer's separate TensorFlow/Keras artifact or validate wildfire performance. Its saved ChangeFormer result has 94.99% pixel accuracy but only 2.45% recall, illustrating why overall accuracy is insufficient for sparse changes. The viewer has no FIRMS integration or validated fire alerts. The historical risk-ranking model and team-budget dashboard described earlier are a separate pipeline; see their model and integration documentation for scope and evaluation.

### Checks and troubleshooting

From `satellite_app`, run the local tests without making Sentinel API requests:

```sh
.venv/bin/python -m unittest discover -p 'test_*.py'
```

These checks verify software behavior, including request handling and model preprocessing; they do not measure real-world detection accuracy.

- **Credentials not configured:** set both `.env` values or environment variables, then restart the server.
- **Authentication or permission error:** confirm the client is active and your account can use the requested Sentinel Hub APIs.
- **No usable observations:** try wider historical windows or a different area. A requested date can have no satellite pass or usable coverage.
- **No jointly clear pixels:** choose different dates. Visual comparison may still be possible, but an experimental change result requires overlapping clear coverage.
- **Model unavailable:** confirm the bundled file's size and checksum against the model notes and install the dependencies in the app's `.venv`.

This server is intended for local use. The initial request can be slower while imagery is downloaded and TensorFlow loads the model; subsequent matching requests use a local cache.
