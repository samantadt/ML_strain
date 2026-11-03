"""Held-out permutation diagnostics; analysis never alters model selection."""

import numpy as np
import pandas as pd

from .data import CLASSIFICATION, FEATURES
from .workflow import load_bundle


def interpret(run, repeats=10):
    from pathlib import Path

    if repeats < 2:
        raise ValueError("Use at least two permutation repeats")
    bundle = load_bundle(run)
    df = pd.read_csv(Path(run) / "dataset_with_split.csv")
    frame = df.loc[df.partition == "test_interior"].copy()
    x = frame[FEATURES].to_numpy(float)
    model = bundle["model"]
    original = model.predict(x)
    rng = np.random.default_rng(42)
    rows = []
    for j, target in enumerate(bundle["targets"]):
        valid = frame[target].notna().to_numpy()
        y = frame.loc[valid, target].to_numpy(float)
        if not len(y):
            continue
        cls = target in CLASSIFICATION

        def error(p):
            return np.mean((y - p[valid]) ** 2) if cls else np.mean(np.abs(y - p[valid]))

        baseline = error(original[:, j])
        for k, feature in enumerate(FEATURES):
            increases = []
            for _ in range(repeats):
                perturbed = x.copy()
                perturbed[:, k] = rng.permutation(perturbed[:, k])
                increases.append(error(model.predict(perturbed)[:, j]) - baseline)
            rows.append(
                dict(
                    target=target,
                    feature=feature,
                    error="brier" if cls else "mae",
                    error_increase_mean=np.mean(increases),
                    error_increase_std=np.std(increases, ddof=1),
                    repeats=repeats,
                    data_source=bundle["data_source"],
                )
            )
    pd.DataFrame(rows).to_csv(Path(run) / "permutation_importance.csv", index=False)
    return rows
