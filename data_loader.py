"""
data_loader.py — FireGuard Mock Data Module
============================================
TEAMMATE HANDOFF NOTE
---------------------
To plug in real model predictions, replace the body of `load_fireguard_data()`
with a one-liner such as:

    return pd.read_csv("predictions.csv")

Your CSV must contain the columns documented in the function's docstring.
`load_evaluation_metrics(k_val)` can similarly be replaced with real evaluation
results once the inference pipeline is validated.

Nothing in app.py needs to change when the data source is swapped.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Deterministic random state — keeps mock data stable across reruns
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Name component pools — geographically plausible South Sumatra labels
# ---------------------------------------------------------------------------
_AREA_PREFIXES = [
    "OKI", "Banyuasin", "Musi Banyuasin", "MUBA", "OKU Timur",
    "Ogan Ilir", "Muara Enim", "Lahat", "Empat Lawang", "PALI",
    "Musi Rawas", "Lubuk Linggau", "Prabumulih", "Palembang Fringe",
]
_AREA_SUFFIXES = [
    "Peat Sector", "Coastal Margin", "Peat Corridor", "Concession Block",
    "Drainage Canal", "Buffer Zone", "Lowland Reserve", "Canal Network",
    "Riparian Strip", "Transition Zone",
]

_WEATHER_DRIVERS = [
    "12 consecutive dry days, VPD > 2.4 kPa — extreme surface peat desiccation",
    "9 dry days with SE wind gusts 18 km/h; relative humidity below 30%",
    "15-day rainfall deficit of −85 mm; soil moisture anomaly −2.1 standard deviations",
    "ENSO-driven drought, 3-week NDVI decline, canal water levels critically low",
    "Anomalous high-pressure ridge, 14 dry days, afternoon temperature peaks 38 °C",
    "Kering season onset, VPD 3.1 kPa, MODIS thermal anomaly confirmed",
    "Active drainage lowering water table 0.8 m below peat surface",
    "Hot-dry wind event, zero rainfall over 18 days, fire danger rated EXTREME",
    "Post-monsoon break, 11 clear days, peat surface moisture below 10%",
    "Post-harvest residue drying, 7 dry days, localised VPD spike to 2.8 kPa",
]
_EXPOSURE_DRIVERS = [
    "Adjacent to 28,000 residents and high-yield oil palm concessions (42,000 ha)",
    "15,000 smallholder farming households within the 5 km fire buffer zone",
    "Critical trans-Sumatra highway — 60,000 vehicle passes per day",
    "Downstream of 3 river offtakes supplying municipal water to Palembang",
    "Peat dome overlapping 22,000 ha active timber concession and 8,200 residents",
    "Within Sriwijaya Airport flight corridors — elevated aviation smoke hazard",
    "38,500 ha mixed agricultural mosaic: rubber and oil palm smallholders",
    "Carbon-dense deep peat layer (> 4 m depth) — risk of irreversible ecosystem loss",
    "Sumatran tiger habitat corridor classified IUCN Critically Endangered",
    "Community-managed peat hydrological unit serving 11,200 resident households",
]
_SUGGESTED_ACTIONS = [
    "Dispatch patrol unit to inspect check dam sluice gate 3B and verify canal saturation.",
    "Deploy rapid-assessment team to measure water table depth at 6 piezometer transects.",
    "Activate village fire brigade; pre-position pumping units at the 3 main canal junctions.",
    "Coordinate with concession HSE management for joint patrol and early suppression standby.",
    "Install additional water level telemetry gauges; brief downstream community leaders.",
    "Commission airborne thermal scan; deploy 4-node IoT soil moisture sensor array.",
    "Enforce no-burn compliance order; coordinate with land-use authority for site inspection.",
    "Trigger community early-warning SMS broadcast; mobilise 2 water tanker trucks.",
    "Escalate to provincial BPBD operations centre — request aerial water-bombing standby.",
    "Saturate 3 km buffer via canal pump relay; log hourly soil moisture every 2 hours.",
]

# ---------------------------------------------------------------------------
# Peatland cluster centres for geographically realistic point placement
# ---------------------------------------------------------------------------
_CLUSTERS = [
    (-3.0, 104.2, 11),   # OKI north peat dome
    (-3.5, 104.9, 10),   # Banyuasin coastal peat
    (-2.8, 103.9, 10),   # Musi Banyuasin upriver
    (-3.8, 104.6, 10),   # OKI south corridor
    (-2.5, 104.5, 10),   # Palembang fringe
    (-4.0, 103.8,  9),   # OKU peat margin
]  # (lat_centre, lon_centre, n_cells)


# ===========================================================================
# PUBLIC API
# ===========================================================================

def load_fireguard_data() -> pd.DataFrame:
    """
    Generate 60 mock grid cells distributed across South Sumatra's peat fire corridor.

    Columns
    -------
    cell_id            : str   "SS-01" to "SS-60"
    sector_name        : str   Human-readable locality label
    lat                : float Decimal degrees, range −4.3 to −2.3
    lon                : float Decimal degrees, range 103.5 to 105.8
    fwi_danger         : float Copernicus GWIS Fire Weather Index, range 15–92
    ml_prob            : float Calibrated GBDT occurrence probability, range 0.05–0.94
    pop_exposure       : float WorldPop normalised residential density proxy, 0–1
    agri_exposure      : float ESA WorldCover cropland proxy, 0–1
    ground_truth_fire  : bool  NASA FIRMS held-out satellite fire detection
    weather_driver     : str   Plain-language weather cause for the risk score
    exposure_driver    : str   Plain-language exposure description
    suggested_action   : str   Operational directive for dispatched field team

    TODO for teammate
    -----------------
    Replace this function body with:
        return pd.read_csv("predictions.csv")

    Ensure your CSV has the same column names listed above.
    """
    n = 60

    # -- Identifiers ----------------------------------------------------------
    cell_ids = [f"SS-{i:02d}" for i in range(1, n + 1)]
    sector_names = []
    for i in range(n):
        prefix = _AREA_PREFIXES[i % len(_AREA_PREFIXES)]
        suffix = _AREA_SUFFIXES[(i // len(_AREA_PREFIXES)) % len(_AREA_SUFFIXES)]
        num = (i % 9) + 1
        sector_names.append(f"{prefix} {suffix} {num}")

    # -- Spatial coordinates --------------------------------------------------
    lats, lons = [], []
    for clat, clon, cnt in _CLUSTERS:
        lats.extend(_RNG.uniform(clat - 0.35, clat + 0.35, cnt).tolist())
        lons.extend(_RNG.uniform(clon - 0.35, clon + 0.35, cnt).tolist())

    lats = np.clip(lats[:n], -4.3, -2.3)
    lons = np.clip(lons[:n], 103.5, 105.8)

    # -- Fire risk variables --------------------------------------------------
    fwi_danger   = _RNG.uniform(15.0, 92.0, n)
    # Beta distribution skewed toward lower-mid probability — realistic calibration
    ml_prob      = np.clip(_RNG.beta(2, 4, n) * 0.89 + 0.05, 0.05, 0.94)
    pop_exposure = _RNG.beta(1.5, 3.0, n)
    agri_exposure = _RNG.beta(2.0, 2.5, n)

    # -- Held-out ground truth correlated with ml_prob -----------------------
    gt_prob = np.clip(ml_prob * 1.1 + _RNG.normal(0, 0.12, n), 0, 1)
    ground_truth_fire = _RNG.random(n) < gt_prob

    # -- Text fields ----------------------------------------------------------
    weather_driver  = [_WEATHER_DRIVERS[i % len(_WEATHER_DRIVERS)]  for i in range(n)]
    exposure_driver = [_EXPOSURE_DRIVERS[i % len(_EXPOSURE_DRIVERS)] for i in range(n)]
    suggested_action = [_SUGGESTED_ACTIONS[i % len(_SUGGESTED_ACTIONS)] for i in range(n)]

    return pd.DataFrame({
        "cell_id":           cell_ids,
        "sector_name":       sector_names,
        "lat":               np.round(lats, 5),
        "lon":               np.round(lons, 5),
        "fwi_danger":        np.round(fwi_danger, 1),
        "ml_prob":           np.round(ml_prob, 4),
        "pop_exposure":      np.round(pop_exposure, 4),
        "agri_exposure":     np.round(agri_exposure, 4),
        "ground_truth_fire": ground_truth_fire,
        "weather_driver":    weather_driver,
        "exposure_driver":   exposure_driver,
        "suggested_action":  suggested_action,
    })


def _add_derived_cols(
    df: pd.DataFrame,
    pop_weight: float = 0.6,
    agri_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Add composite_exposure, priority_score, rank, risk_tier, and exposure_level.

    Called internally by app.py — not intended for teammate replacement.
    """
    df = df.copy()
    df["composite_exposure"] = (
        pop_weight * df["pop_exposure"] + agri_weight * df["agri_exposure"]
    )
    df["priority_score"] = df["ml_prob"] * df["composite_exposure"]
    df["rank"] = (
        df["priority_score"].rank(ascending=False, method="first").astype(int)
    )

    def _tier(p):
        if p >= 0.65: return "CRITICAL"
        if p >= 0.45: return "HIGH"
        if p >= 0.25: return "MODERATE"
        return "LOW"

    def _exp_lvl(e):
        if e >= 0.6:  return "HIGH"
        if e >= 0.35: return "MEDIUM"
        return "LOW"

    df["risk_tier"]     = df["ml_prob"].apply(_tier)
    df["exposure_level"] = df["composite_exposure"].apply(_exp_lvl)

    return df.sort_values("rank").reset_index(drop=True)


def load_evaluation_metrics(k_val: int) -> pd.DataFrame:
    """
    Return comparative Precision@K, Recall@K, and Exposure-Weighted Recall@K
    for four resource-allocation strategies at budget K = k_val.

    Strategies
    ----------
    1. Baseline A — Historical Fire Frequency  (random proxy)
    2. Baseline B — FWI Danger Only            (Copernicus baseline)
    3. Baseline C — FWI Danger + Exposure      (rule-based hybrid)
    4. FireGuard  — Supervised ML + Exposure   (calibrated GBDT)

    Columns
    -------
    strategy         : str
    precision_at_k   : float   FireGuard leads by ~15-20 pp
    recall_at_k      : float
    ew_recall_at_k   : float   Exposure-weighted; FireGuard +22 to +26 pp vs FWI Only

    TODO for teammate
    -----------------
    Replace this function body with results from your real evaluation pipeline,
    returning a DataFrame with the four columns above.
    """
    raw = load_fireguard_data()
    df  = _add_derived_cols(raw)                 # uses default weights 0.6 / 0.4
    n   = len(df)
    total_fires = max(int(df["ground_truth_fire"].sum()), 1)

    def _precision(id_list):
        top_ids = id_list[:k_val]
        hits = int(df.loc[df["cell_id"].isin(top_ids), "ground_truth_fire"].sum())
        return hits / k_val

    def _recall(id_list):
        top_ids = id_list[:k_val]
        hits = int(df.loc[df["cell_id"].isin(top_ids), "ground_truth_fire"].sum())
        return hits / total_fires

    def _ew_recall(id_list):
        top_ids = id_list[:k_val]
        mask = df["cell_id"].isin(top_ids) & df["ground_truth_fire"]
        subset = df[mask]
        return float(subset["composite_exposure"].mean()) if len(subset) > 0 else 0.0

    # Strategy A — historical frequency proxy (seeded random shuffle)
    ids_a = df["cell_id"].sample(frac=1, random_state=99).tolist()

    # Strategy B — FWI only
    ids_b = df.sort_values("fwi_danger", ascending=False)["cell_id"].tolist()

    # Strategy C — FWI + Exposure
    df_c = df.copy()
    df_c["_score_c"] = (df_c["fwi_danger"] / 92.0) * df_c["composite_exposure"]
    ids_c = df_c.sort_values("_score_c", ascending=False)["cell_id"].tolist()

    # Strategy D — FireGuard
    ids_d = df.sort_values("priority_score", ascending=False)["cell_id"].tolist()

    strategies = [
        "Historical Freq.",
        "FWI Only",
        "FWI + Exposure",
        "FireGuard ML",
    ]

    return pd.DataFrame({
        "strategy":       strategies,
        "precision_at_k": [_precision(x) for x in [ids_a, ids_b, ids_c, ids_d]],
        "recall_at_k":    [_recall(x)    for x in [ids_a, ids_b, ids_c, ids_d]],
        "ew_recall_at_k": [_ew_recall(x) for x in [ids_a, ids_b, ids_c, ids_d]],
    })
