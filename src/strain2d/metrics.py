"""Metrics retain units and denominators; undefined AUC is explicitly null."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .data import CLASSIFICATION


def score(y, p, classification=False):
    mask = np.isfinite(y)
    y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    if not len(y):
        return {"n": 0}
    if not np.isfinite(p).all():
        raise ValueError("Non-finite predictions")
    if classification:
        label = p >= 0.5
        both = len(np.unique(y)) == 2
        return dict(
            n=len(y),
            accuracy=float(accuracy_score(y, label)),
            balanced_accuracy=float(balanced_accuracy_score(y, label)) if both else None,
            roc_auc=float(roc_auc_score(y, p)) if both else None,
            brier=float(brier_score_loss(y, p)),
            log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
            precision=float(precision_score(y, label, zero_division=0)),
            recall=float(recall_score(y, label, zero_division=0)),
            f1=float(f1_score(y, label, zero_division=0)),
            confusion_matrix=confusion_matrix(y, label, labels=[0, 1]).tolist(),
        )
    return dict(
        n=len(y),
        mae=float(mean_absolute_error(y, p)),
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        r2=float(r2_score(y, p)) if len(y) >= 2 and np.std(y) > 0 else None,
    )


def evaluate(frame, prediction, targets):
    return {
        t: score(frame[t].to_numpy(float), prediction[:, j], t in CLASSIFICATION)
        for j, t in enumerate(targets)
    }


def conformal_radius(y, p, alpha):
    residual = np.abs(np.asarray(y) - np.asarray(p))
    residual = residual[np.isfinite(residual)]
    # Finite-sample split conformal quantile; refuse unsupported finite radius.
    rank = int(np.ceil((len(residual) + 1) * (1 - alpha)))
    if not 1 <= rank <= len(residual):
        raise ValueError(f"Too few calibration labels ({len(residual)}) for alpha={alpha}")
    return float(np.sort(residual)[rank - 1])


def bootstrap_mae(y, p, seed=42, repeats=1000):
    y, p = np.asarray(y), np.asarray(p)
    keep = np.isfinite(y)
    err = np.abs(y[keep] - p[keep])
    if len(err) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = [rng.choice(err, len(err), replace=True).mean() for _ in range(repeats)]
    return np.quantile(vals, [0.025, 0.975]).tolist()
