# Strain2D — machine learning for strain engineering of 2D materials

A new implementation of the workflow in Ma et al., *Transformer-based prediction of two-dimensional material electronic properties under elastic strain engineering*, arXiv:2603.20141v1. It prepares VASP calculations, learns a **material-specific** strain/property surrogate, compares models, quantifies held-out error, and screens new strain candidates.

**Current status:** the software is tested on synthetic data and controlled VASP/phonon fixtures. There are no real material results or pretrained physical models here. The paper's dataset and supplementary information were not supplied, so this is a documented independent implementation, not an exact numerical reproduction. Real VASP execution must be validated on your university cluster.

## What you get

- Maximin Latin-hypercube sampling over three in-plane strain components, including an unstrained reference.
- Fixed-cell VASP ionic relaxation and static calculations; a Slurm array template; finite-displacement Phonopy preparation and collection.
- Strict CSV validation, explicit units, missing-target support, and material/protocol isolation.
- Dummy, polynomial, random-forest, SVR, MLP and Transformer models; optional XGBoost.
- Masked multi-task neural training: two scaled regression targets and two binary classification targets.
- A common five-way data split, validation-only model selection, independent calibration, and untouched interior/outer-strain tests.
- Saved models, predictions, metrics, MAE bootstrap intervals, regression prediction intervals, plots, attention diagnostics, permutation importance, sensitivities, constrained screening and new DFT-point suggestions.
- Data/configuration/artifact hashes, exact split membership, software versions and training histories.

A model trained on one material does **not** predict a different material reliably. Supply a separate consistent dataset and train a new run for each material, phase and DFT protocol.

## New to machine learning?

Read the **[70-page beginner handbook (PDF)](docs/Strain2D_Beginners_Handbook.pdf)**. It starts from zero and explains data, training/validation/test choices, model selection, metrics, the Transformer, uncertainty, VASP data preparation, practical commands, troubleshooting, and presenting the project to a supervisor. It includes worked examples, exercises with answers, a glossary and a study plan.

The [editable guide source](docs/BEGINNER_GUIDE.md) is also available. To rebuild the PDF, use `bash tools/build_beginner_guide.sh` in an environment with the project dependencies, Pandoc, XeLaTeX, DejaVu fonts, and the LaTeX `fvextra`/`fancyhdr` packages. This documentation toolchain is separate from the ML runtime.

## 1. Install

Python 3.11–3.12 is recommended; this project was exercised with Python 3.12 on CPU Linux.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dft,dev]'
python -m pytest -q
```

For the exact tested CPU dependency versions, use `python -m pip install -r requirements-cpu.lock` followed by `python -m pip install -e . --no-deps`. The snapshot targets Linux/Python 3.12; use `pyproject.toml` for other platforms. GPU PyTorch can be installed using the official wheel appropriate for your cluster. Training currently runs deterministically on CPU, with the configured thread count.

In the prepared Codex machine: `source /workspace/strain2d-clean-venv/bin/activate`, then `cd /workspace/ML_strain`.

## 2. Run the complete software demonstration

```bash
strain2d demo-data --n 400 --out examples/SYNTHETIC_demo.csv
strain2d check examples/SYNTHETIC_demo.csv
strain2d train --data examples/SYNTHETIC_demo.csv --config configs/smoke.yaml --out runs/demo
strain2d predict --run runs/demo --points examples/prediction_points.csv --out runs/demo/predictions.csv
strain2d interpret --run runs/demo
strain2d sensitivity --run runs/demo --points examples/prediction_points.csv --out runs/demo/sensitivity.csv
strain2d sample --n 500 --out runs/candidates.csv
strain2d screen --run runs/demo --points runs/candidates.csv --gap-low 4.2 --gap-high 4.6 --out runs/demo/screen.csv
strain2d suggest --run runs/demo --points runs/candidates.csv --n 20 --out runs/demo/next_dft.csv
```

Open `runs/demo/report.md` and `runs/demo/figures/`. A complete pre-executed synthetic smoke run is also included in `examples/SYNTHETIC_verification_run/` (models and reports); see `docs/SOFTWARE_VERIFICATION.md`. A screen can legitimately return zero candidates. Output training directories must be new, so old and new artifacts cannot be mixed. `configs/smoke.yaml` is a short **software check**, not a production accuracy setting. Use `configs/default.yaml` as the starting point for real training, then tune only with validation data.

For notebooks, install `python -m pip install -e '.[notebook]'` and select that Python environment in Jupyter.

`notebooks/Start_here.ipynb` walks through the same commands. No unpublished synthetic figures are presented as paper results.

### Optional XGBoost benchmark

```bash
python -m pip install -e '.[xgboost]'
strain2d train --data examples/SYNTHETIC_demo.csv --config configs/xgboost.yaml --out runs/boosting
```

On Linux this uses the official `xgboost-cpu` wheel. `requirements-xgboost-cpu.lock` adds the exact version checked here. This optional baseline has also been exercised; it is not silently substituted by another algorithm.

## 3. Prepare your real calculations

Choose and converge a material's unstrained reference structure first. See [VASP and HPC guide](docs/HPC.md). With a relaxed, xy-aligned `POSCAR`:

```bash
strain2d sample --n 1000 --low -2 --high 10 --seed 42 --out data_points.csv
strain2d dft-prepare --poscar POSCAR --points data_points.csv --material YOUR_MATERIAL_PHASE_PBE_PROTOCOL_v1 --out calculations/study --cutoff 600 --kmesh 6 6 1
```

The paper's range/cutoff/mesh are **starting values, not converged recommendations for an unchosen material**. Pilot 3–5 cases before a full array; do not assume a 10% strain remains elastic. You provide the licensed VASP executable, potentials in the exact POSCAR species order, and cluster-specific scheduler settings. Nothing is submitted automatically.

After running relaxation/static and phonon cases:

```bash
strain2d dft-collect --calc calculations/study --out data_real.csv
strain2d check data_real.csv
strain2d train --data data_real.csv --config configs/default.yaml --out runs/material_v1
```

Unsuccessful cases are recorded in `data_real.csv.rejected.csv`. Missing phonons produce missing stability labels, never invented labels. An unstrained reference is required for energy differences. For existing data, use the [dataset schema](docs/DATA.md); there is no need to use the VASP generator.

## 4. Understand the scientific choices

Read [paper mapping and limitations](docs/PAPER.md), [validation protocol](docs/VALIDATION.md), and [supervisor briefing](docs/SUPERVISOR_BRIEF.md).

The Transformer uses the three variable strain tokens; the paper's constant-zero out-of-plane components are omitted. Its default width/heads/layers/feed-forward size follow the paper's stated 128/8/4/256 dimensions; this implementation records its actual parameter count and does not claim to match the paper's unexplained 209k count. Binary directness and phonon stability use classification losses and metrics. Attention is descriptive, not proof of causal coupling.

All input strains are **percentage points**: `2` means 2%, `exy` is tensor shear (`gamma_xy/2`). `F = I + epsilon`; the implementation does not call this Green–Lagrange strain. Strain energy is `(E_strained - E_reference)*1000/N` in **meV/atom**.

No model can promise DFT accuracy before it is measured on your material. Test errors are relative to your DFT protocol, not experimental truth. New proposed designs need independent DFT/phonon confirmation.

## Project layout

```
src/strain2d/    data, sampling, models, metrics, workflow, report, analysis, dft, phonons, cli
configs/        full and smoke training configurations
examples/       labelled synthetic data and strain queries
notebooks/      executable starting notebook
tests/          schema, learning, inference, VASP, phonon and full-workflow tests
docs/           data contract, HPC instructions, methodology and supervisor briefing
```

Local model files use Python serialization; load only models you created or trust.
