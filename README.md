# FireGuard

FireGuard is a local satellite comparison prototype for investigating landscape changes around reported fires in South Sumatra. It retrieves Sentinel-2 imagery through Sentinel Hub and displays the same area before and after a selected period.

The app includes Before, After, Changes, and Split View. Actual acquisition dates and estimated clear coverage are shown for each image. An optional U-Net overlay highlights experimental change candidates; it is hidden by default and has **not been validated for wildfire detection or damage assessment**.

## Run locally

Requirements: Git, Python 3.12, an internet connection, and Sentinel Hub client credentials with access to Sentinel-2 L2A and the Statistical and Process APIs. API calls use your Sentinel Hub account's allowance. The first dependency installation downloads TensorFlow and may take several minutes.

```sh
git clone --branch upgrades https://github.com/S-W-Leong/FireGuard.git
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

## Compare imagery

1. Select an area or enter a bounding box in `[west, south, east, north]` order using WGS 84 longitude and latitude. Each side must span at most 0.25 degrees.
2. Choose non-overlapping before and after windows, each at most 63 days. The app selects the day with the highest estimated clear coverage in each window. The displayed acquisition date may differ from the window endpoints.
3. Click **Load satellite pair**, then inspect **Before**, **After**, or **Split View**. **Recent imagery** fills recent date windows; click Load to retrieve them.
4. To inspect the model output, choose **Changes** and enable **Show experimental U-Net output**.

Clouds remain visible in the natural-color images. Model results are limited to pixels classified as clear in both observations. This filtering does not establish that a prediction is correct: cloud edges, seasonal differences, registration, and unfamiliar terrain can still affect results. The displayed flagged percentage refers to jointly clear model pixels, not confirmed burned area.

## Historical 2026 fire reference

An optional comparison covers an area near a reported OKI fire. Indonesia's environment ministry reports fire activity on **4–8 August 2026** in the [PT Bintang Harapan Palma concession](https://www.kemenlh.go.id/news/detail/dampak-kebakaran-lahan-klhbplh-lakukan-penyegelan-di-konsesi-pt-bintang-harapan-palma).

Use these settings to request the observations previously inspected in the prototype:

| Setting | Value |
| --- | --- |
| Bounding box | `[105.2876675, -3.130584, 105.3376675, -3.080584]` |
| Before From / To | `2026-07-12` / `2026-07-12` |
| After From / To | `2026-08-19` / `2026-08-19` |

These are historical reference dates, not 11 October imagery. A report about a fire in the area does not provide pixel-level labels for the U-Net or prove that every visible difference was caused by that fire. Availability still depends on the provider and account access.

## Included model weights

The full pretrained artifact is bundled at [`satellite_app/models/unet_model.h5`](satellite_app/models/unet_model.h5), so a normal clone includes the weights. No retraining or separate model download is needed to run inference. See the [model notes](satellite_app/models/README.md) for the checksum, input contract, and limitations.

The app independently stretches each raw RGB image to its 2nd–98th percentile range, resizes it to 128 × 128, and stacks before RGB followed by after RGB. The model emits one change probability per pixel. Predictions above 0.5 are shown only where both dates pass the clear-pixel mask. A probability is an uncalibrated model output, not a measured probability of fire.

## Research notebook and project status

[`notebooks/FireGuard Model.ipynb`](notebooks/FireGuard%20Model.ipynb) preserves LEVIR-CD/PyTorch change-detection experiments and a ChangeFormer research section, including existing saved outputs and paper credits. It was not rerun for this package and does not train or export the TensorFlow/Keras artifact used by the app. Its results do not establish wildfire performance. Its Colab paths, dependencies, and checkpoint references require review before execution.

The current package provides satellite retrieval, visual comparison, and experimental inference. NASA FIRMS integration is not implemented, and no FIRMS key is configured. The package does not include a trained fire-risk classifier, validated fire alerts, or a completed team-allocation system. The broader objective is described in [`docs/FireGuard.md`](docs/FireGuard.md); the JSON files under `data/` are feasibility reports, not a bundled training dataset.

## Checks and troubleshooting

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
