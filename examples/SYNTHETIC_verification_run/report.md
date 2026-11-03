# Strain2D calculation report

> SYNTHETIC DATA: these results validate software only; they are not material predictions.

Material/protocol: `DEMO_ONLY_not_a_material`. Source: `synthetic`.
Selected model: **polynomial**, using **validation** bandgap_eV mae.
Partition sizes: `{'train': 220, 'validation': 40, 'calibration': 40, 'test_interior': 40, 'test_outer': 60}`.

## Untouched test results

| Test | Target | N | MAE or Brier | RMSE or AUC | R² or balanced accuracy |
|---|---|---:|---:|---:|---:|
| test_interior | bandgap_eV | 40 | 0.0022971 | 0.0029369 | 0.9995 |
| test_interior | strain_energy_meV_atom | 40 | 0.059294 | 0.074078 | 1 |
| test_interior | is_direct | 40 | 0.0068806 | 1 | 1 |
| test_interior | is_stable | 40 | 0.061836 | 0.97619 | 0.875 |
| test_outer | bandgap_eV | 60 | 0.0024197 | 0.0029937 | 0.99948 |
| test_outer | strain_energy_meV_atom | 60 | 0.24409 | 0.2638 | 0.99997 |
| test_outer | is_direct | 60 | 0.0022051 | 1 | 1 |
| test_outer | is_stable | 60 | 5.5557e-05 | undefined | undefined |

Regression units: bandgap in eV; energy in meV/atom. Classification scores are dimensionless.
All models: `metrics.csv`; raw test rows: `test_predictions.csv`; confusion matrices: `metrics.json`.

## Uncertainty and scope

Split-conformal alpha = 0.1; constant absolute-residual radii: `{'bandgap_eV': 0.00483483084469416, 'strain_energy_meV_atom': 0.13404717040526748}`.
Marginal coverage relies on exchangeability. The outer-strain test is deliberately shifted; coverage is measured, not promised there.
Ensemble spread is disagreement, not a confidence interval. Classification probabilities are not certified stability.
The training hull is a support diagnostic, not a stable region. No refit on calibration or test data was performed.
Bootstrap MAE intervals quantify held-out sampling variability, not systematic DFT error or training-seed variation.

## Interpretation

Attention is descriptive and cannot establish causal strain coupling. Constant out-of-plane tokens are omitted.
Use finite-difference sensitivity at several step sizes and confirm proposed mechanisms by new DFT calculations.

## Reproducibility and physical validation

Dataset SHA256: `23463290ad83f7274ff664435353c69261747f280abb1d3b73711f2d823d6698`.
Configuration, versions, exact split, model files, training histories and artifact hashes are saved in this directory.
The source dataset must use one material/phase and a consistent numerical protocol.
A successful synthetic run does not establish DFT accuracy or reproduce the paper's 0.0103 eV MAE.
Check docs/VALIDATION.md before presenting material results to a supervisor.
