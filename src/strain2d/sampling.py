"""Reproducible maximin Latin-hypercube designs and explicit finite-strain convention."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import qmc

from .data import FEATURES


def lhs(n=1000, low=-2.0, high=10.0, seed=42, trials=10):
    if n < 2 or low >= high or not low <= 0 <= high or trials < 1:
        raise ValueError("Need n>=2, trials>=1, low<high and a range containing zero")
    rng = np.random.default_rng(seed)
    best, score = None, -np.inf
    for _ in range(trials):
        candidate = qmc.scale(
            qmc.LatinHypercube(3, seed=int(rng.integers(2**31))).random(n - 1), [low] * 3, [high] * 3
        )
        candidate = np.vstack([np.zeros(3), candidate])
        separation = pdist(candidate).min()
        if separation > score:
            best, score = candidate, separation
    df = pd.DataFrame(best, columns=FEATURES)
    df.insert(0, "sample_id", [f"s{i:05d}" for i in range(n)])
    return df


def deformation(strain):
    """F=I+ε; ε is symmetric infinitesimal strain in percent, not Green strain."""
    xx, yy, xy = np.asarray(strain, float) / 100
    f = np.array([[1 + xx, xy, 0.0], [xy, 1 + yy, 0.0], [0.0, 0.0, 1.0]])
    if not np.isfinite(f).all() or np.linalg.eigvalsh(f).min() <= 0:
        raise ValueError("Non-positive or invalid deformation")
    return f


def synthetic(n=400, seed=42):
    """Invented smooth functions, not fitted to the paper or valid for h-BN."""
    df = lhs(n, seed=seed)
    x, y, s = df[FEATURES].to_numpy().T
    rng = np.random.default_rng(seed)
    df["material_id"] = "DEMO_ONLY_not_a_material"
    df["data_source"] = "synthetic"
    df["bandgap_eV"] = 4.7 - 0.025 * (x + y) - 0.0015 * (x * y) - 0.002 * s * s + rng.normal(0, 0.003, n)
    df["strain_energy_meV_atom"] = 1.2 * (x * x + y * y) + 0.5 * x * y + 1.8 * s * s
    df["is_direct"] = (s < 3 + 0.1 * x).astype(int)
    df["is_stable"] = ((x * x + y * y + 2 * s * s < 125) & (x + y > -2)).astype(int)
    return df
