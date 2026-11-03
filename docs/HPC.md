# VASP / university-cluster workflow

## Before launching a study

1. Obtain your university's VASP module/executable and licensed PBE PAW potentials. The project does not redistribute these. Confirm POTCAR species order against **each POSCAR**, including displaced supercells. Keep the same potential versions in every stage; generated scripts record SHA256 hashes.
2. Choose the material, phase, magnetism, functional, charge and (if needed) spin–orbit treatment. The generated starter INCAR is nonmagnetic/PBE without SOC or dipole correction. It is directly suitable only after those choices are justified. SOC/noncollinear XML is deliberately rejected until you supply a validated extraction route. Configure magnetic/polar/charged materials separately before using the generic workflow.
3. Converge the zero-strain structure, vacuum, cutoff, k mesh, smearing and force tolerance. Verify a dense electronic mesh includes the actual band extrema; a 6×6×1 mesh is not a universal bandgap calculation. Relax the reference in-plane lattice while controlling the vacuum; then use that reference for all **fixed-cell** strained ionic relaxations.
4. Pilot a few normal and shear strains. Confirm the material remains in the intended structural branch, the imposed cell persists, forces converge, static geometries match the relaxed outputs, and the gap/energy are sensible.

Optional illustrative geometry, NOT a converged structure:

```bash
strain2d example-structure --out hBN_UNRELAXED.POSCAR
```

## Prepare a small pilot

```bash
strain2d sample --n 5 --low -2 --high 10 --out pilot_points.csv
strain2d dft-prepare --poscar POSCAR --points pilot_points.csv --out calculations/pilot --material MATERIAL_PHASE_PBE_v1 --cutoff 600 --kmesh 6 6 1
```

Ranges and meshes are configurable. At finite deformation we use symmetric `F=I+epsilon`, epsilon in fractional units after division by 100. This is a chosen strain convention, not Green–Lagrange strain. Cell vectors in ASE's row representation transform as `cell @ F.T`. Vacuum is preserved.

`dft-prepare` refuses existing output directories. It records the reference, sampling CSV, protocol, and generated input hashes. Inputs must be consistent with that prepared protocol. If you change numerical settings, regenerate a new protocol/directory; do not mix edited and original cases. For extra material-specific INCAR tags, first adapt and test the generator's templates, then regenerate and version your study. Do not change imposed strain by using unconstrained cell relaxation.

## Run a case

Install this package with `[dft]` on the cluster and activate it inside batch jobs. Load the university's compiler/MPI/VASP modules according to local documentation. Then, for example:

```bash
export POTCAR_PATH=/absolute/licensed/path/to/POTCAR
export VASP_COMMAND='srun vasp_std'
bash calculations/pilot/run_case.sh s00000
```

The script runs relaxation then static in separate subdirectories, validates output after each, and stops on failure. Static POSCAR is copied from the relaxed CONTCAR. It refuses existing `vasprun.xml` files to prevent accidental overwrite. To resume, inspect the failure and run the appropriate stage manually; do not blindly remove outputs or reuse unconverged CONTCARs.

For Slurm arrays, configure the generated `array.slurm` with your account, partition, wall time, tasks and memory; activate Python and load VASP modules in your submission environment/job as required locally. Set **absolute** `CALC_ROOT`, POTCAR_PATH and VASP_COMMAND. Submit the pilot first:

```bash
export CALC_ROOT=/absolute/path/to/calculations/pilot
sbatch --array=0-4%2 "$CALC_ROOT/array.slurm"
```

The array index addresses a row of `points.csv`, including row 0 reference. For n=1000, indices are 0–999; choose concurrency with cluster staff rather than launching 1000 simultaneously. No submission command has been executed by this project.

## Phonons

After a case's static run succeeds:

```bash
strain2d phonon-prepare --case calculations/pilot/s00000 --supercell 4 4 1 --displacement 0.01 --force-kmesh 2 2 1
bash calculations/pilot/s00000/phonon/run_forces.sh
strain2d phonon-collect --case calculations/pilot/s00000 --qmesh 9 9 1
```

Run the force script **inside a properly allocated cluster job**, not on a login node. It loops over displacements; you can distribute these directories with a site-specific array once the pilot is checked. The force mesh controls electronic integration of the supercell; q mesh controls phonon sampling and is a different quantity. Converge both. Ensure supercell POTCAR species order remains correct.

The collector checks displaced positions/cells, atom order, potential hashes and electronic completion. It subtracts each supercell's mean force and records the largest correction. It builds force constants using Phonopy and saves `phonopy.yaml`, frequencies and provenance. No silent force-constant symmetrization, acoustic-sum-rule enforcement or polar NAC is performed. Study those corrections explicitly where appropriate. The saved q-mesh minimum is not a full phonon dispersion or a guarantee of stability over the Brillouin zone.

## Collect

```bash
strain2d dft-collect --calc calculations/pilot --out pilot.csv --phonon-tolerance 20 --direct-tolerance 0.01
```

The tolerance means `is_stable = min_frequency_cm1 >= -20`, a numerical decision rule, not a universal stability criterion. Repeat analysis with tighter/looser tolerances and converged phonons. Missing phonon results stay missing. The direct-gap label is mesh-resolved, with a 0.01 eV tolerance; metallic/fractionally occupied cases have gap 0 and no direct-gap label. Near-metal finite-smearing occupations need manual review or an improved electronic calculation.

The collector requires the zero-strain reference; rejects truncated XML, electronic steps hitting NELM, failed ionic force criteria, cell/geometry drift, changed prepared inputs and inconsistent potential hashes. It uses final `e_0_energy` and computes `1000*(E-E0)/N`. It writes rejection reasons and collection provenance. These checks cannot detect every bad physical setup: independently inspect a few real outputs, especially when first using a different VASP version.

A pilot contains too few cases for the five-way ML protocol. After it succeeds, prepare the full sampling study (usually hundreds to thousands of points). Keep rejected strains and outcomes in the scientific record to expose survivor bias; failed numerical calculations must not silently define the stability domain.
