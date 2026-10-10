"""Inference for the verified local U-Net artifact; never trains or modifies weights."""
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageFilter

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'unet_model.h5'
MODEL_SHA256 = 'a7276e3b22945fb2f0169db5c8679adee2334463579d6ff1ca117be439b1ee55'
MODEL_BYTES = 5726504
ANALYSIS_VERSION = MODEL_SHA256 + ':raw-rgb-percentile-128-clear-v2'
SIZE = (128, 128)
_model = None
_lock = threading.Lock()


class ModelError(Exception):
    pass


def model_status(path=MODEL_PATH):
    if not path.is_file():
        return {'available': False, 'reason': 'weights_missing', 'expected_bytes': MODEL_BYTES}
    data = path.read_bytes()
    if data.startswith(b'version https://git-lfs'):
        return {'available': False, 'reason': 'weights_missing', 'expected_bytes': MODEL_BYTES}
    valid = len(data) == MODEL_BYTES and hashlib.sha256(data).hexdigest() == MODEL_SHA256
    return {'available': valid, 'reason': 'ready' if valid else 'weights_not_verified',
            'expected_bytes': MODEL_BYTES, 'method': 'U-Net', 'trained_here': False,
            'validation_status': 'unvalidated',
            'input_shape': [128, 128, 6], 'threshold': 0.5}


def load_verified_model():
    global _model
    if _model is not None:
        return _model
    if not model_status()['available']:
        raise ModelError('The verified U-Net weights are unavailable.')
    import h5py
    # Inspect the model config before deserializing this older HDF5 format.
    with h5py.File(MODEL_PATH, 'r') as file:
        config = json.loads(file.attrs['model_config'])
    allowed = {'InputLayer', 'Conv2D', 'Dropout', 'MaxPooling2D', 'UpSampling2D', 'Concatenate'}
    layers = config.get('config', {}).get('layers', [])
    if config.get('class_name') != 'Functional' or not layers or any(layer.get('class_name') not in allowed for layer in layers):
        raise ModelError('The model architecture does not match the verified U-Net.')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    from tensorflow.keras.models import load_model
    candidate = load_model(MODEL_PATH, compile=False, safe_mode=True)
    if candidate.input_shape != (None, 128, 128, 6) or candidate.output_shape != (None, 128, 128, 1):
        raise ModelError('The model input or output dimensions are incompatible.')
    _model = candidate
    return _model


def prepare_pair(before, after):
    if before.size != after.size:
        raise ModelError('Before and after images must use the same pixel grid.')
    # The supplied detector stretches raw AUTO RGB to its 2nd/98th percentiles
    # before resizing and stacking RGB-before/RGB-after in the 0..1 range.
    arrays = []
    for image in (before, after):
        pixels = np.asarray(image.convert('RGB'))
        low, high = np.percentile(pixels, [2, 98])
        normalized = (np.clip((pixels - low) / (high - low), 0, 1) * 255).astype(np.uint8) if high > low else np.zeros_like(pixels)
        resized = Image.fromarray(normalized).resize(SIZE, Image.Resampling.BICUBIC)
        arrays.append(np.asarray(resized, dtype=np.float32) / 255.0)
    return np.concatenate(arrays, axis=-1)[None, ...]


def common_clear_mask(before, after):
    if before.size != after.size:
        raise ModelError('Quality masks must use the same pixel grid.')
    # BOX downsamples coverage; require every contributing pixel to be clear.
    masks = [np.asarray(image.convert('L').resize(SIZE, Image.Resampling.BOX)) == 255
             for image in (before, after)]
    return masks[0] & masks[1]


def threshold_prediction(prediction, valid):
    if prediction.shape != (128, 128) or valid.shape != (128, 128) or not np.isfinite(prediction).all():
        raise ModelError('The model returned an invalid prediction.')
    if not ((prediction >= 0).all() and (prediction <= 1).all()):
        raise ModelError('The model returned invalid prediction probabilities.')
    count = int(valid.sum())
    if not count:
        raise ModelError('No jointly clear pixels are available for change detection. Choose different dates.')
    changed = (prediction > 0.5) & valid
    return changed, round(float(changed.sum()) / count * 100, 2)


def make_overlay(changed, size):
    mask = Image.fromarray(changed.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    selected = np.asarray(mask) > 0
    eroded = np.asarray(mask.filter(ImageFilter.MinFilter(3))) > 0
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[selected] = [249, 168, 37, 72]
    rgba[selected & ~eroded] = [255, 223, 64, 255]
    return Image.fromarray(rgba)


def analyze_pair(before_path, after_path, before_clear_path, after_clear_path, output_path):
    with Image.open(before_path) as source:
        before = source.convert('RGB')
    with Image.open(after_path) as source:
        after = source.convert('RGB')
    with Image.open(before_clear_path) as a, Image.open(after_clear_path) as b:
        if a.size != before.size or b.size != after.size:
            raise ModelError('Quality masks and imagery must use the same pixel grid.')
        valid = common_clear_mask(a, b)
    if not valid.any():
        raise ModelError('No jointly clear pixels are available for change detection. Choose different dates.')
    with _lock:
        try:
            model = load_verified_model()
            prediction = np.asarray(model([prepare_pair(before, after)], training=False))[0, :, :, 0]
        except ModelError:
            raise
        except (ImportError, OSError, ValueError, RuntimeError, TypeError):
            raise ModelError('The U-Net runtime could not complete inference. Imagery remains available.') from None
    changed, percentage = threshold_prediction(prediction, valid)
    make_overlay(changed, after.size).save(output_path)
    return {'available': True, 'method': 'U-Net', 'threshold': 0.5,
            'validation_status': 'unvalidated',
            'changed_percent': percentage, 'valid_percent': round(float(valid.mean()) * 100, 2),
            'changed_pixels': int(changed.sum()), 'valid_pixels': int(valid.sum()),
            'model_grid': [128, 128], 'model_sha256': MODEL_SHA256,
            'note': 'Yellow outlines are U-Net predictions on jointly clear pixels. This model has not been validated for fire detection or this region.'}
