"""Leakage-controlled training, immutable run artifacts and deployable predictions."""

import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from .data import CLASSIFICATION, FEATURES, TARGETS, digest, domain_flags, json_write, read_data, split_data
from .metrics import bootstrap_mae, conformal_radius, evaluate
from .models import Surrogate

DEFAULT = dict(
    seed=42,
    models=["dummy", "polynomial", "random_forest", "svr", "mlp", "transformer"],
    selection_target="bandgap_eV",
    alpha=0.1,
    threads=2,
    neural=dict(
        d_model=128,
        heads=8,
        layers=4,
        feedforward=256,
        dropout=0.1,
        learning_rate=1e-4,
        weight_decay=1e-4,
        epochs=600,
        patience=60,
        batch_size=64,
        members=3,
    ),
)


def configuration(path=None):
    import copy

    cfg = copy.deepcopy(DEFAULT)
    if path:
        values = yaml.safe_load(Path(path).read_text())
        if not isinstance(values, dict) or set(values) - set(cfg):
            raise ValueError("Invalid/unknown configuration keys")
        for k, v in values.items():
            if k == "neural":
                if not isinstance(v, dict) or set(v) - set(cfg[k]):
                    raise ValueError("Invalid neural configuration")
                cfg[k].update(v)
            else:
                cfg[k] = v
    if not 0 < cfg["alpha"] < 1 or cfg["threads"] < 1:
        raise ValueError("Invalid alpha or threads")
    n = cfg["neural"]
    for k in ["d_model", "heads", "layers", "feedforward", "epochs", "patience", "batch_size", "members"]:
        if not isinstance(n[k], int) or n[k] < 1:
            raise ValueError(f"Invalid neural {k}")
    if n["d_model"] % n["heads"] or not 0 <= n["dropout"] < 1:
        raise ValueError("d_model must divide by heads; dropout must be in [0,1)")
    if n["learning_rate"] <= 0 or n["weight_decay"] < 0:
        raise ValueError("Invalid optimizer settings")
    allowed = {"dummy", "polynomial", "random_forest", "svr", "mlp", "transformer", "xgboost"}
    if not cfg["models"] or len(set(cfg["models"])) != len(cfg["models"]) or set(cfg["models"]) - allowed:
        raise ValueError("Empty, duplicate or unsupported model list")
    return cfg


def train(data, out, config=None):
    cfg = configuration(config)
    df = read_data(data, min_rows=60)
    targets = [t for t in TARGETS if t in df and df[t].notna().any()]
    if cfg["selection_target"] not in targets:
        raise ValueError("selection_target must be an observed target")
    splits = split_data(df, cfg["seed"])
    # Fail before expensive fitting if the common split cannot support any required task.
    for name in ["train", "validation", "calibration"]:
        subset = df.iloc[splits[name]]
        for target in targets:
            count = int(subset[target].notna().sum())
            if count < 5:
                raise ValueError(f"{name}/{target}: only {count} labels; need >=5; add data")
            if name == "train" and target in CLASSIFICATION and subset[target].nunique() < 2:
                raise ValueError(f"train/{target} requires both classes; add representative labels")
    for t in targets:
        if t not in CLASSIFICATION:
            conformal_radius(
                df.iloc[splits["calibration"]][t], np.zeros(len(splits["calibration"])), cfg["alpha"]
            )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never combine artifacts from different runs
    (out / "models").mkdir()
    (out / "figures").mkdir()
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    assignment = np.empty(len(df), dtype=object)
    for name, ids in splits.items():
        assignment[ids] = name
    df.assign(partition=assignment).to_csv(out / "dataset_with_split.csv", index=False)
    x = df[FEATURES].to_numpy(float)
    ti, vi, ci = [splits[s] for s in ["train", "validation", "calibration"]]
    fitted, validation, durations, neural_histories = {}, {}, {}, {}
    selection = cfg["selection_target"]
    metric = "brier" if selection in CLASSIFICATION else "mae"
    # No test labels or test predictions are accessed in this loop.
    for name in cfg["models"]:
        started = time.perf_counter()
        model = Surrogate(name, targets, cfg).fit(x[ti], df.iloc[ti], x[vi], df.iloc[vi])
        durations[name] = time.perf_counter() - started
        validation[name] = evaluate(df.iloc[vi], model.predict(x[vi]), targets)
        fitted[name] = model
        joblib.dump(model, out / "models" / f"{name}.joblib")
        if model.histories:
            neural_histories[name] = model.histories
        print(
            f"{name}: validation {selection} {metric}={validation[name][selection][metric]:.6g}", flush=True
        )
    winner = min(cfg["models"], key=lambda n: validation[n][selection][metric])
    model = fitted[winner]
    calibration = model.predict(x[ci])
    radii = {
        t: conformal_radius(df.iloc[ci][t], calibration[:, j], cfg["alpha"])
        for j, t in enumerate(targets)
        if t not in CLASSIFICATION
    }
    bundle = dict(
        format_version=1,
        model=model,
        reference=x[ti],
        observed_points=x.copy(),
        targets=targets,
        radii=radii,
        alpha=cfg["alpha"],
        material_id=str(df.material_id.iloc[0]),
        data_source=str(df.data_source.iloc[0]),
        data_sha256=digest(data),
    )
    joblib.dump(bundle, out / "model.joblib")
    metrics, predictions = {}, []
    for split in ["test_interior", "test_outer"]:
        ids = splits[split]
        pred = model.predict(x[ids])
        metrics[split] = {}
        for name, fitted_model in fitted.items():
            metrics[split][name] = evaluate(df.iloc[ids], fitted_model.predict(x[ids]), targets)
        table = predict_frame(bundle, df.iloc[ids][FEATURES])
        table.insert(0, "sample_id", df.iloc[ids].sample_id.to_numpy())
        table["partition"] = split
        for j, t in enumerate(targets):
            table[f"true_{t}"] = df.iloc[ids][t].to_numpy()
            if t not in CLASSIFICATION:
                values = df.iloc[ids][t].to_numpy(float)
                valid = np.isfinite(values)
                m = metrics[split][winner][t]
                m["mae_bootstrap_95pct"] = bootstrap_mae(values, pred[:, j], cfg["seed"])
                m["interval_coverage"] = (
                    float(np.mean(np.abs(values[valid] - pred[valid, j]) <= radii[t]))
                    if valid.any()
                    else None
                )
                m["interval_width"] = 2 * radii[t]
        predictions.append(table)
    pd.concat(predictions).to_csv(out / "test_predictions.csv", index=False)
    json_write(out / "metrics.json", metrics)
    json_write(out / "validation_metrics.json", validation)
    json_write(out / "training_history.json", neural_histories)
    versions = {}
    for pkg in ["strain2d", "numpy", "pandas", "scipy", "scikit-learn", "torch", "ase", "phonopy"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    manifest = dict(
        data_sha256=digest(data),
        config_sha256=digest(out / "config.yaml"),
        material_id=bundle["material_id"],
        data_source=bundle["data_source"],
        selected_model=winner,
        selection_target=selection,
        selection_metric=metric,
        selection_partition="validation",
        split_sizes={k: len(v) for k, v in splits.items()},
        train_seconds=durations,
        neural_parameters={k: m.parameter_count() for k, m in fitted.items()},
        source_sha256={p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        versions=versions,
        python=platform.python_version(),
        git_commit=git.stdout.strip() if git.returncode == 0 else None,
        git_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        calibration_alpha=cfg["alpha"],
        conformal_radii=radii,
        refit_all_data=False,
        strain_units="percent; tensor shear",
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
    )
    json_write(out / "manifest.json", manifest)
    if "transformer" in fitted:
        np.save(out / "attention.npy", fitted["transformer"].attention(x[splits["test_interior"]]))
    from .report import make_report

    make_report(out, df, metrics, manifest, neural_histories)
    # Artifact integrity, not an assertion that a pickle is safe from an untrusted sender.
    json_write(
        out / "artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    return manifest


def load_bundle(run):
    path = Path(run) / "model.joblib"
    hashes_path = Path(run) / "artifact_hashes.json"
    if hashes_path.exists():
        hashes = json.loads(hashes_path.read_text())
        if hashes.get("model.joblib") != digest(path):
            raise ValueError("Model artifact checksum mismatch")
    bundle = joblib.load(path)  # Only load models you created or trust (Python pickle).
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model format")
    return bundle


def predict_frame(bundle, frame):
    x = frame[FEATURES].to_numpy(float)
    members = bundle["model"].predict_members(x)
    pred = members.mean(0)
    output = frame[FEATURES].reset_index(drop=True).copy()
    for j, target in enumerate(bundle["targets"]):
        output[f"pred_{target}"] = pred[:, j]
        if target in CLASSIFICATION:
            output[f"label_{target}"] = (pred[:, j] >= 0.5).astype(int)
        else:
            radius = bundle["radii"][target]
            output[f"lower_{target}"] = pred[:, j] - radius
            output[f"upper_{target}"] = pred[:, j] + radius
            if len(members) > 1:
                output[f"ensemble_std_{target}"] = members[:, :, j].std(0, ddof=1)
    box, hull = domain_flags(bundle["reference"], x)
    output["outside_training_box"], output["outside_training_hull"] = box, hull
    output["material_id"], output["data_source"] = bundle["material_id"], bundle["data_source"]
    return output


def screen(run, candidates, out, gap_low, gap_high, stable_probability=0.9, limit=30):
    bundle = load_bundle(run)
    if not {"bandgap_eV", "is_stable"}.issubset(bundle["targets"]):
        raise ValueError("Screening needs bandgap_eV and is_stable labels")
    if gap_low >= gap_high or not 0.5 <= stable_probability <= 1 or limit < 1:
        raise ValueError("Invalid screening constraints")
    points = read_data(candidates, labels=False)
    predictions = predict_frame(bundle, points)
    # Conservative regression interval constraint and support check; stability score is NOT a certification.
    keep = (
        (predictions.lower_bandgap_eV >= gap_low)
        & (predictions.upper_bandgap_eV <= gap_high)
        & (predictions.pred_is_stable >= stable_probability)
        & ~predictions.outside_training_hull
    )
    result = predictions.loc[keep].sort_values("pred_is_stable", ascending=False).head(limit)
    result.to_csv(out, index=False)
    return len(result)


def suggest(run, candidates, out, n=20):
    """Diversity-first exploration mixed with ensemble disagreement. No claim of calibrated OOD risk."""
    from scipy.spatial.distance import cdist

    bundle = load_bundle(run)
    df = read_data(candidates, labels=False).drop_duplicates(FEATURES).reset_index(drop=True)
    if not 1 <= n <= len(df):
        raise ValueError("n must be between 1 and number of unique candidates")
    x = df[FEATURES].to_numpy(float)
    reference = bundle.get("observed_points", bundle["reference"])
    scale = np.maximum(np.ptp(reference, axis=0), 1e-8)
    distance = cdist(x / scale, reference / scale).min(1)
    eligible = distance > 1e-8
    pred = bundle["model"].predict_members(x)
    if len(pred) > 1:
        spread = pred.std(0)
        # Normalize across candidates so eV and meV do not share a raw scale.
        uncertainty = (spread / np.maximum(np.quantile(spread, 0.9, axis=0), 1e-8)).mean(1)
    else:
        uncertainty = np.zeros(len(x))
    chosen = []
    for _ in range(min(n, int(eligible.sum()))):
        acquisition = distance * (1 + uncertainty)
        acquisition[~eligible] = -np.inf
        i = int(np.argmax(acquisition))
        chosen.append(i)
        eligible[i] = False
        distance = np.minimum(distance, np.linalg.norm((x - x[i]) / scale, axis=1))
    result = df.iloc[chosen].copy()
    result["ensemble_disagreement_score"] = uncertainty[chosen]
    result["selection_order"] = np.arange(1, len(chosen) + 1)
    result.to_csv(out, index=False)
    return len(result)


def sensitivity(run, points, out, step=0.05):
    """Finite-difference sensitivity in eV / percentage point; step convergence must be checked."""
    if step <= 0:
        raise ValueError("step must be positive")
    bundle = load_bundle(run)
    if "bandgap_eV" not in bundle["targets"]:
        raise ValueError("No bandgap model")
    frame = read_data(points, labels=False)
    x = frame[FEATURES].to_numpy(float)
    j = bundle["targets"].index("bandgap_eV")
    result = predict_frame(bundle, frame)
    for k, feature in enumerate(FEATURES):
        plus, minus = x.copy(), x.copy()
        plus[:, k] += step
        minus[:, k] -= step
        result[f"dEg_d{feature}_eV_per_percent"] = (
            bundle["model"].predict(plus)[:, j] - bundle["model"].predict(minus)[:, j]
        ) / (2 * step)
        flags = domain_flags(bundle["reference"], np.vstack([plus, minus]))[1]
        result[f"derivative_{feature}_outside_support"] = flags[: len(x)] | flags[len(x) :]
    result["finite_difference_step_percent"] = step
    result.to_csv(out, index=False)
