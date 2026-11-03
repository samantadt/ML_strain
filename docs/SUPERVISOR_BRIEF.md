# Supervisor briefing: proposed material-specific strain surrogate study

## Objective

Reduce the number of expensive first-principles calculations needed to explore how a chosen 2D material responds electronically and dynamically to combined normal/shear strain. Inputs are exx, eyy, exy; outputs are fundamental bandgap, energy change per atom, mesh-resolved directness, and phonon stability under a declared tolerance.

## Method

A space-filling strain design feeds fixed-cell VASP ionic relaxations and static calculations. Phonopy finite-displacement calculations provide stability labels on a converged q mesh. A Transformer learns coupled strain representations with jointly optimized regression/classification heads. Identical data partitions compare it with MLP, polynomial, SVR, random-forest and optional XGBoost controls. Validation selects a model; calibration estimates task-wise regression intervals; untouched interior and high-strain subsets quantify error and distribution shift. Proposed designs are subsequently validated by new DFT calculations.

## Deliverables already implemented

Source package, command-line workflow, training configurations, typed data contract, synthetic software demonstration, structure/job generation, XML collection guards, phonon workflow, saved predictions/models/provenance, report figures, attention/permutation diagnostics, sensitivity analysis, candidate screening and diverse next-calculation selection. Automated tests distinguish actual learning from trivial executable checks.

## Evidence available now

Software tests and a synthetic end-to-end run; generated results are explicitly marked synthetic. There is **no material-specific accuracy result**, real VASP execution, demonstrated physical stability window, or independent reproduction of the paper's headline error. See `SOFTWARE_VERIFICATION.md` for the exact executed checks and counts.

## Decisions needed for the actual study

Choose material/phase, relaxed reference, physically meaningful strain window, charge/spin/SOC treatment, DFT convergence budget, phonon label policy and target application. Decide whether directness and phonons are sufficiently meaningful for that material. Confirm the cluster's scheduler/MPI modules and potential selection with the local VASP administrator.

## Proposed milestones

1. Converged reference plus 3–5 manually reviewed pilot cases.
2. A broad initial design (hundreds of points), increasing toward the paper's order of 1000 where feasible; quantify missing-label and calculation-failure patterns.
3. Frozen model comparison with multiple seeds, data-size learning curves where budget allows, uncertainty coverage and errors reported per target/unit.
4. Fresh DFT/phonon validation of shortlisted designs, with an explicitly defined success rate and uncertainty.

## Claims appropriate at this stage

“The calculation and ML software has been implemented and checked on synthetic/controlled fixtures; the material study and VASP validation are the next experimental-computational phase.”

Avoid claiming “DFT-level accuracy”, “97.7% success”, causal shear dominance, or universal applicability until those claims have independent evidence for the chosen material. The paper is an architectural/workflow reference; deviations and unavailable reproduction details are documented in `PAPER.md`.
