# FireGuard model artifact

`unet_model.h5` is the full supplied pretrained TensorFlow/Keras U-Net artifact. It includes architecture and weights in HDF5 format, rather than a Git LFS pointer. The app loads it for inference without retraining or modifying it.

| Property | Value |
| --- | --- |
| File size | 5,726,504 bytes |
| SHA-256 | `a7276e3b22945fb2f0169db5c8679adee2334463579d6ff1ca117be439b1ee55` |
| Input | `(batch, 128, 128, 6)` |
| Channel order | Before RGB, then after RGB |
| Output | `(batch, 128, 128, 1)` change probability |
| Display threshold | Strictly greater than 0.5 |
| Validation status | Unvalidated for wildfire detection and the target region |

The app checks the exact file size and SHA-256, inspects the architecture, and verifies input/output shapes before inference. The checksum establishes artifact identity; it is not evidence of predictive accuracy.

Preprocessing in `model_runtime.py` independently stretches each raw 8-bit RGB observation using its 2nd and 98th percentiles, resizes to 128 × 128 with bicubic interpolation, converts to floats in the 0–1 range, and concatenates the two observations. Both images must use the same area and pixel grid. The app excludes pixels that do not pass both clear-coverage masks.

The yellow overlay is an experimental change candidate mask. It does not identify a change's cause, confirm burned land, detect an active fire, or estimate future fire risk. The original training domain and fire-specific accuracy have not been established in this project.

The root research notebook [`Satellite Experimentation.ipynb`](../../Satellite%20Experimentation.ipynb) contains separate LEVIR-CD/PyTorch and ChangeFormer experiments. It does not produce this HDF5 artifact, and its saved outputs must not be presented as validation of the bundled model.

To check the downloaded file on macOS:

```sh
shasum -a 256 unet_model.h5
wc -c < unet_model.h5
```

On Linux, `sha256sum unet_model.h5` provides the same checksum check. Run these commands from `satellite_app/models`.
