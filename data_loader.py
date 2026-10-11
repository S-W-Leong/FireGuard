"""Real-data replacement for the Dashboard-1.0 mock loader.

Runtime reads only the portable bundle, never downloads data or trains a model.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parent / "data" / "dashboard"


def load_manifest() -> dict:
    return json.loads((BUNDLE / "manifest.json").read_text())


def _verified_path(name: str) -> Path:
    path = BUNDLE / name
    expected = load_manifest()["bundle_hashes"][name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Dashboard bundle differs from its verified manifest: {name}")
    return path


def load_fireguard_data() -> pd.DataFrame:
    return pd.read_csv(_verified_path("cell_weeks.csv.gz"))


def load_grid() -> dict:
    return json.loads(_verified_path("grid.geojson").read_text())
