"""Publication-style figures and a report that separates software from physical validation."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import CLASSIFICATION, FEATURES


def make_report(out, df, metrics, manifest, histories):
    source = manifest["data_source"]
    prefix = "SYNTHETIC — software verification only\n" if source == "synthetic" else ""
    selected = manifest["selected_model"]
    rows = []
    for partition, models in metrics.items():
        for model, targets in models.items():
            for target, values in targets.items():
                for metric, value in values.items():
                    if isinstance(value, (float, int)) or value is None:
                        rows.append(
                            dict(partition=partition, model=model, target=target, metric=metric, value=value)
                        )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    predictions = pd.read_csv(out / "test_predictions.csv")
    targets = list(metrics["test_interior"][selected])
    for target in targets:
        if target in CLASSIFICATION:
            fig, ax = plt.subplots(figsize=(5, 4))
            for split, group in predictions.groupby("partition"):
                bins = np.linspace(0, 1, 6)
                x, y = [], []
                for low, high in zip(bins[:-1], bins[1:]):
                    mask = (
                        group[f"pred_{target}"].between(low, high, inclusive="both" if high == 1 else "left")
                        & group[f"true_{target}"].notna()
                    )
                    if mask.any():
                        x.append(group.loc[mask, f"pred_{target}"].mean())
                        y.append(group.loc[mask, f"true_{target}"].mean())
                ax.plot(x, y, "o-", label=split)
            ax.plot([0, 1], [0, 1], "k--")
            ax.set(
                xlabel="Mean predicted probability",
                ylabel="Observed positive fraction",
                xlim=(0, 1),
                ylim=(0, 1),
                title=prefix + target + " reliability (small-bin counts can be noisy)",
            )
        else:
            fig, ax = plt.subplots(figsize=(5, 4))
            for split, group in predictions.groupby("partition"):
                ax.scatter(group[f"true_{target}"], group[f"pred_{target}"], s=15, alpha=0.7, label=split)
            truth = predictions[f"true_{target}"].dropna()
            if len(truth):
                low, high = truth.min(), truth.max()
                ax.plot([low, high], [low, high], "k--")
            units = "eV" if target == "bandgap_eV" else "meV/atom"
            ax.set(
                xlabel=f"Reference ({units})",
                ylabel=f"Prediction ({units})",
                title=prefix + selected + ": " + target,
            )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{target}.png", dpi=200)
        plt.close(fig)
    if histories:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, members in histories.items():
            for i, h in enumerate(members):
                ax.plot(np.arange(1, len(h) + 1), h, label=f"{name} member {i + 1}")
        ax.set(xlabel="Epoch", ylabel="Masked scaled validation loss", title=prefix + "Convergence")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "convergence.png", dpi=200)
        plt.close(fig)
    attention = out / "attention.npy"
    if attention.exists():
        matrix = np.load(attention).mean(axis=(0, 1, 2))
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix, vmin=0, vmax=1)
        ax.set_xticks(range(3), FEATURES)
        ax.set_yticks(range(3), FEATURES)
        ax.set(xlabel="Key", ylabel="Query", title=prefix + "Mean attention — descriptive, not causal")
        fig.colorbar(image)
        fig.tight_layout()
        fig.savefig(out / "figures" / "attention.png", dpi=200)
        plt.close(fig)
    lines = [
        "# Strain2D calculation report",
        "",
        "> SYNTHETIC DATA: these results validate software only; they are not material predictions."
        if source == "synthetic"
        else "> Material-specific surrogate. Review DFT convergence and held-out error before scientific use.",
        "",
        f"Material/protocol: `{manifest['material_id']}`. Source: `{source}`.",
        f"Selected model: **{selected}**, using **validation** {manifest['selection_target']} {manifest['selection_metric']}.",
        f"Partition sizes: `{manifest['split_sizes']}`.",
        "",
        "## Untouched test results",
        "",
        "| Test | Target | N | MAE or Brier | RMSE or AUC | R² or balanced accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def fmt(value):
        return "undefined" if value is None else f"{value:.5g}"

    for partition in metrics:
        for target, m in metrics[partition][selected].items():
            if target in CLASSIFICATION:
                values = [m.get("brier"), m.get("roc_auc"), m.get("balanced_accuracy")]
            else:
                values = [m.get("mae"), m.get("rmse"), m.get("r2")]
            lines.append(
                f"| {partition} | {target} | {m['n']} | " + " | ".join(fmt(v) for v in values) + " |"
            )
    lines += [
        "",
        "Regression units: bandgap in eV; energy in meV/atom. Classification scores are dimensionless.",
        "All models: `metrics.csv`; raw test rows: `test_predictions.csv`; confusion matrices: `metrics.json`.",
        "",
        "## Uncertainty and scope",
        "",
        f"Split-conformal alpha = {manifest['calibration_alpha']}; constant absolute-residual radii: `{manifest['conformal_radii']}`.",
        "Marginal coverage relies on exchangeability. The outer-strain test is deliberately shifted; coverage is measured, not promised there.",
        "Ensemble spread is disagreement, not a confidence interval. Classification probabilities are not certified stability.",
        "The training hull is a support diagnostic, not a stable region. No refit on calibration or test data was performed.",
        "Bootstrap MAE intervals quantify held-out sampling variability, not systematic DFT error or training-seed variation.",
        "",
        "## Interpretation",
        "",
        "Attention is descriptive and cannot establish causal strain coupling. Constant out-of-plane tokens are omitted.",
        "Use finite-difference sensitivity at several step sizes and confirm proposed mechanisms by new DFT calculations.",
        "",
        "## Reproducibility and physical validation",
        "",
        f"Dataset SHA256: `{manifest['data_sha256']}`.",
        "Configuration, versions, exact split, model files, training histories and artifact hashes are saved in this directory.",
        "The source dataset must use one material/phase and a consistent numerical protocol.",
        "A successful synthetic run does not establish DFT accuracy or reproduce the paper's 0.0103 eV MAE.",
        "Check docs/VALIDATION.md before presenting material results to a supervisor.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
