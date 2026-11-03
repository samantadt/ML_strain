"""Strict one-material schema. Strain is percent, using tensor shear εxy=γxy/2."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

FEATURES = ["exx", "eyy", "exy"]
REGRESSION = ["bandgap_eV", "strain_energy_meV_atom"]
CLASSIFICATION = ["is_direct", "is_stable"]
TARGETS = REGRESSION + CLASSIFICATION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(path, *, labels=True, min_rows=1):
    df = pd.read_csv(path)
    required = FEATURES + (["sample_id", "material_id", "data_source"] if labels else [])
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if labels:
        for c in ["sample_id", "material_id", "data_source"]:
            if df[c].isna().any() or df[c].astype(str).str.strip().eq("").any():
                raise ValueError(f"Empty {c}")
        if df.sample_id.duplicated().any():
            raise ValueError("sample_id must be unique")
        if df.material_id.nunique() != 1:
            raise ValueError("Train one material/phase/protocol per run; material_id must be constant")
        if not set(df.data_source).issubset({"synthetic", "dft", "experiment"}):
            raise ValueError("data_source must be synthetic, dft or experiment")
        if df.data_source.nunique() != 1:
            raise ValueError("Do not mix synthetic, DFT and experimental labels")
        if not any(t in df for t in TARGETS):
            raise ValueError("No target columns")
    for c in FEATURES + [t for t in TARGETS if labels and t in df]:
        # Invalid strings must fail, rather than silently become missing labels.
        df[c] = pd.to_numeric(df[c], errors="raise")
        a = df[c].to_numpy(float)
        if np.isinf(a).any() or (c in FEATURES and np.isnan(a).any()):
            raise ValueError(f"Non-finite {c}")
    if (df[FEATURES].abs() > 30).any().any():
        raise ValueError("Strains must be percent, within ±30%; this workflow is for modest elastic strains")
    # Unique to 1e-8 percentage points: reject repetitions so no near-identical point crosses splits.
    if labels and df[FEATURES].round(8).duplicated().any():
        raise ValueError(
            "Repeated strain tensors: aggregate replicates with their uncertainty before training"
        )
    if labels:
        for c in CLASSIFICATION:
            if c in df and not df[c].dropna().isin([0, 1]).all():
                raise ValueError(f"{c} must be 0/1 or missing")
        for c in REGRESSION:
            if c in df and c == "bandgap_eV" and (df[c].dropna() < 0).any():
                raise ValueError("Band gaps cannot be negative")
        cols = [c for c in TARGETS if c in df]
        if df[cols].isna().all(axis=1).any():
            raise ValueError("Every training row needs at least one label")
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows; got {len(df)}")
    return df.reset_index(drop=True)


def split_data(df, seed=42):
    """Fixed common split: outer strain-norm 15%, then random train/val/cal/test.

    ~55/10/10/10/15 percent. Calibration is NOT used to select weights or models.
    Radius uses the tensor Frobenius norm, including both symmetric shear entries.
    """
    n = len(df)
    if n < 60:
        raise ValueError("At least 60 unique points are required for the five-way protocol")
    x = df[FEATURES].to_numpy(float)
    radius = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 2 * x[:, 2] ** 2)
    # Break exact radial ties deterministically; exact boundary ties are disclosed in metadata.
    order = np.argsort(radius, kind="stable")
    ne = max(1, int(np.ceil(0.15 * n)))
    pool = np.random.default_rng(seed).permutation(order[:-ne])
    k = max(1, int(np.floor(0.10 * n)))
    return {
        "train": np.sort(pool[3 * k :]),
        "validation": np.sort(pool[:k]),
        "calibration": np.sort(pool[k : 2 * k]),
        "test_interior": np.sort(pool[2 * k : 3 * k]),
        "test_outer": np.sort(order[-ne:]),
    }


def domain_flags(reference, points):
    """Convex-hull flag is a geometric support check, never a stability guarantee."""
    reference, points = np.asarray(reference), np.asarray(points)
    in_box = ((points >= reference.min(0)) & (points <= reference.max(0))).all(1)
    try:
        inside = Delaunay(reference).find_simplex(points, tol=1e-10) >= 0
    except QhullError:
        inside = np.zeros(len(points), dtype=bool)  # cannot certify a full 3D support region
    return ~in_box, ~inside


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
