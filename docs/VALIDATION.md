# Validation protocol and evidence boundaries

## Software checks

Run `python -m pytest -q`, `ruff check src tests` and the README smoke workflow. Tests cover:

- deterministic sampling, tensor shear and deformation validity;
- strict schemas, provenance/source isolation, duplicates and split disjointness;
- conformal quantiles and support flags;
- actual neural training improvement, missing-label gradients, probability bounds, attention row sums and persistence;
- complete baseline training/report/prediction/screening/suggestion flow and model-checksum rejection;
- XML gap/directness/metal/energy/force-convergence extraction on hand-constructed fixtures;
- fixed-cell POSCAR generation and generated shell syntax;
- energy conversion, rejection logs and missing phonon labels;
- real Phonopy displacement/force-constant APIs driven by known test forces, substituting VASP reading.

The XML fixtures are minimal, and the phonon integration fixture is not a real crystal calculation. No real VASP output was provided. Run the cluster pilot and inspect extraction before treating the collection path as validated for your VASP version.

## Data split and learning

All models share the exact saved split. The largest 15% of strain-tensor Frobenius norms form `test_outer`; exact norm ties are resolved in input order. From the rest, approximately 10% of the full dataset goes to each of validation, calibration and `test_interior`; the remainder (about 55%) is training. This conservative split sacrifices training size to make evaluation and uncertainty estimation auditable.

Feature and regression-target scalers use training data only. Missing labels are masked per task; binary tasks use logits/BCE and regressions use standardized MSE with equal per-observed-task weight. The current implementation does not rebalance rare classes: inspect class counts and confusion matrices before trusting a classification score. Neural early stopping and winning-model selection use validation only. Winning-model selection minimizes one predeclared task (default bandgap MAE; a classification target uses Brier loss). It does not optimize a unit-mixed aggregate metric. Report any tradeoffs on the other tasks.

Classical baselines use transparent fixed hyperparameters; this is not an exhaustive hyperparameter search and does not establish architectural superiority. Tune validation hyperparameters with equal budgets, then **freeze the protocol** before inspecting tests. MLP uses the same optimizer settings as the Transformer for a controlled starting comparison, rather than claiming the settings are optimal for both. The dummy model is an intentional sanity control; it may win on weak data, which must be reported honestly.

Only after selection are test predictions computed. Models are not refit on calibration or test rows. Iterative screening that adds new DFT points requires a new versioned study and an untouched external validation set; repeatedly optimizing against a previously inspected test set compromises it.

## Uncertainty

Split-conformal intervals use absolute residuals of the selected frozen model on the separate calibration subset, with the finite-sample rank `ceil((n+1)*(1-alpha))`. The run refuses a finite-radius claim when calibration is too small. These are marginal, task-wise intervals under exchangeability, not joint four-task guarantees. Radial selection and spatial correlations can break exchangeability, especially in `test_outer`. Test coverage and width are reported rather than assumed. Negative lower gap endpoints are not silently clipped.

Deep-ensemble standard deviation is disagreement, not calibrated uncertainty. Single classical models have no ensemble-spread column. DFT-point acquisition then uses distance/diversity only; an ensemble adds normalized disagreement. Candidate suggestions exclude all existing dataset points (including held-out coordinates without using their labels); also filter against any separately queued calculations. Do not consume a fixed test set in the acquisition loop.

Bootstrap MAE intervals resample held-out errors at a fixed trained model. They do not include training-seed variation, spatial dependence, DFT systematic error or experimental uncertainty. Classification includes class counts through confusion matrices, Brier/log loss, AUC, accuracy, balanced accuracy, precision, recall and F1. AUC is null for a one-class test subset.

## Required physical validation before a supervisor-facing accuracy claim

1. Document the selected structure, phase, lattice reference, potential hashes, functional, magnetism/SOC, vacuum and every numerical setting.
2. Converge energy differences, band edges, forces, phonon supercell/displacement/electronic mesh/q mesh. Preserve a few independently checked outputs and quantify numerical uncertainty.
3. Compare normal and shear pilot calculations by hand, including energy per atom and preservation of the requested cell.
4. Acquire enough labels and both classes; show where calculations failed or labels are absent, rather than hiding those regions.
5. Run the frozen comparison on at least three predeclared seeds. Keep each run separate. Report mean/std and all individual outcomes, not the best seed. An independent held-out DFT set selected before tuning is stronger than random nearby-point splitting.
6. Check whether learned stable candidates really pass fresh phonon calculations. Report the number tested, the success definition and its uncertainty; do not reuse the paper's percentages.
7. Repeat numerical sensitivities at several strain steps. Derivatives of tree models are step-dependent and can be zero/discontinuous; attention and permutation rankings are diagnostic, not causal mechanisms.

## Single-target ablation / seed study

For a single-task study, make a new CSV preserving all rows/IDs/features/provenance but dropping the other target columns, and use the same seed/configuration/model dimensions. With complete labels this preserves splits. Record parameter counts and training budgets; missing-label filtering can otherwise change rows. Compare each task separately; do not pool eV, meV/atom and class errors.

To vary the seed, copy a configuration to a new version, change `seed`, and train to a new output directory. Use train/validation only for development. The project does not automatically hunt for a favorable seed. Preserve all runs and use `metrics.csv`/`manifest.json` for aggregation.

## Supported scientific scope

One relaxed branch of one 2D material; modest homogeneous in-plane symmetric deformations at a consistent electronic-structure level. No universal cross-material generalization, finite-temperature dynamics, substrate/contact physics, transport coefficients, optical excitons or experimental bandgap correction is inferred. Fixed-cell ionic relaxation is intentional. Structural transitions and buckling require additional branch descriptors or separate models. The implementation is a research pipeline; readiness of software is separate from readiness of a physical conclusion.
