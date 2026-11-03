"""Finite-displacement Phonopy preparation and force collection for VASP."""

import json
from pathlib import Path

import numpy as np

from .data import digest, json_write
from .dft import STATIC_INCAR, kpoints, parse_vasp


def prepare_phonons(case, supercell=(4, 4, 1), displacement=0.01, mesh=(2, 2, 1)):
    from ase import Atoms
    from ase.io import read, write
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    case = Path(case)
    if any(int(v) < 1 for v in supercell) or supercell[2] != 1 or displacement <= 0:
        raise ValueError("Positive nx ny 1 supercell and positive displacement required")
    if any(int(v) < 1 for v in mesh) or mesh[2] != 1:
        raise ValueError("Force mesh must be nx ny 1")
    static = case / "static" / "vasprun.xml"
    parse_vasp(static)
    atoms = read(static, format="vasp-xml")
    cutoff = None
    for line in (case / "static" / "INCAR").read_text().splitlines():
        if line.strip().startswith("ENCUT"):
            cutoff = float(line.split("=")[1].split("#")[0])
    if cutoff is None:
        raise ValueError("Missing static ENCUT")
    cell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(), cell=atoms.cell, scaled_positions=atoms.get_scaled_positions()
    )
    phonon = Phonopy(cell, supercell_matrix=np.diag(supercell), primitive_matrix=np.eye(3))
    phonon.generate_displacements(distance=displacement)
    out = case / "phonon"
    out.mkdir(exist_ok=False)
    phonon.save(filename=str(out / "phonopy_disp.yaml"))
    for i, sc in enumerate(phonon.supercells_with_displacements):
        folder = out / f"disp-{i + 1:04d}"
        folder.mkdir()
        a = Atoms(symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True)
        # Keep atom order consistent with the Phonopy displacement definition.
        write(folder / "POSCAR", a, format="vasp", direct=True, sort=False)
        (folder / "INCAR").write_text(STATIC_INCAR.format(cutoff=cutoff))
        (folder / "KPOINTS").write_text(kpoints(mesh))
    json_write(
        out / "provenance.json",
        dict(
            reference_static_sha256=digest(static),
            supercell=list(supercell),
            displacement_A=displacement,
            force_kmesh=list(mesh),
            n_displacements=len(phonon.supercells_with_displacements),
            symprec=1e-5,
            force_potential_sha256=(case / "static" / "potential.sha256").read_text().strip(),
        ),
    )
    (out / "run_forces.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${POTCAR_PATH:?Set the absolute potential path in supercell POSCAR species order}"
: "${VASP_COMMAND:?Set launcher command}"
[[ "$POTCAR_PATH" = /* ]] || exit 2
for folder in disp-*; do
  (
    cd "$folder"
    [[ ! -e vasprun.xml ]] || { echo "Output exists; inspect/resume manually" >&2; exit 2; }
    cp -- "$POTCAR_PATH" POTCAR
    sha256sum POTCAR | cut -d ' ' -f1 > potential.sha256
    bash -c "$VASP_COMMAND" > vasp.stdout 2> vasp.stderr
    python -m strain2d verify-vasp vasprun.xml
  )
done
""")
    return len(phonon.supercells_with_displacements)


def collect_phonons(case, qmesh=(9, 9, 1)):
    from ase.io import read
    from phonopy import load

    case = Path(case)
    out = case / "phonon"
    meta = json.loads((out / "provenance.json").read_text())
    if meta["reference_static_sha256"] != digest(case / "static" / "vasprun.xml"):
        raise ValueError("Static reference changed since phonon preparation")
    if any(v < 1 for v in qmesh) or qmesh[2] != 1:
        raise ValueError("qmesh must be nx ny 1")
    phonon = load(str(out / "phonopy_disp.yaml"), produce_fc=False)
    forces, drift = [], []
    for i, expected in enumerate(phonon.supercells_with_displacements):
        folder = out / f"disp-{i + 1:04d}"
        xml = folder / "vasprun.xml"
        parse_vasp(xml)
        actual = read(xml, format="vasp-xml")
        if not np.allclose(actual.cell, expected.cell, atol=1e-6, rtol=0):
            raise ValueError("Displaced supercell changed")
        # Periodic comparison avoids false rejection of wrapped positions.
        delta = actual.get_scaled_positions() - expected.scaled_positions
        delta -= np.rint(delta)
        if np.max(np.abs(delta)) > 1e-6 or actual.get_chemical_symbols() != list(expected.symbols):
            raise ValueError("Displacement/atom order changed")
        if (folder / "potential.sha256").read_text().strip() != meta["force_potential_sha256"]:
            raise ValueError("Phonon potentials differ from static reference")
        f = actual.get_forces()
        if not np.isfinite(f).all():
            raise ValueError("Invalid forces")
        mean_force = f.mean(0)
        drift.append(float(np.linalg.norm(mean_force)))
        forces.append(f - mean_force)  # record drift rather than hide the correction
    phonon.forces = forces
    phonon.produce_force_constants()
    # Report raw finite-displacement result; no silent ASR/NAC correction.
    phonon.run_mesh(list(qmesh), is_gamma_center=True, is_mesh_symmetry=False)
    frequencies = phonon.mesh.frequencies
    if not np.isfinite(frequencies).all():
        raise ValueError("Non-finite phonon frequencies")
    np.save(out / "frequencies_THz.npy", frequencies)
    phonon.save(filename=str(out / "phonopy.yaml"), settings={"force_constants": True})
    result = dict(
        min_frequency_cm1=float(frequencies.min() * 33.35640951981521),
        min_frequency_THz=float(frequencies.min()),
        qmesh=list(qmesh),
        reference_static_sha256=meta["reference_static_sha256"],
        max_mean_force_drift_eV_A=max(drift),
        nac=False,
        force_constant_symmetrization=False,
        force_source_hashes={
            f"disp-{i + 1:04d}": digest(out / f"disp-{i + 1:04d}" / "vasprun.xml") for i in range(len(forces))
        },
    )
    json_write(out / "phonon_result.json", result)
    return result
