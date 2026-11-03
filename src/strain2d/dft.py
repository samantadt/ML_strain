"""VASP fixed-strain input generation and conservative XML collection.

No potential files, executable, job submission or physical labels are fabricated.
"""

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, digest, json_write
from .sampling import deformation

RELAX_INCAR = """SYSTEM = fixed-strain ionic relaxation
ENCUT = {cutoff}
PREC = Accurate
EDIFF = 1E-8
EDIFFG = -0.01
ISMEAR = 0
SIGMA = 0.02
ISPIN = 1
ISYM = 0
IBRION = 2
NSW = 200
ISIF = 2
NELM = 200
LREAL = .FALSE.
LASPH = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
STATIC_INCAR = """SYSTEM = fixed-strain static gap/energy
ENCUT = {cutoff}
PREC = Accurate
EDIFF = 1E-8
ISMEAR = 0
SIGMA = 0.02
ISPIN = 1
ISYM = 0
IBRION = -1
NSW = 0
NELM = 200
LREAL = .FALSE.
LASPH = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
"""


def kpoints(mesh):
    return "Gamma mesh\n0\nGamma\n" + " ".join(map(str, mesh)) + "\n0 0 0\n"


def prepare(poscar, points, out, material, cutoff=600, mesh=(6, 6, 1)):
    from ase.io import read, write

    if not material.strip() or cutoff <= 0 or any(int(v) < 1 for v in mesh) or mesh[2] != 1:
        raise ValueError("Need material/protocol ID, positive cutoff and in-plane mesh nx ny 1")
    atoms = read(poscar, format="vasp")
    cell = np.asarray(atoms.cell)
    if not np.allclose(cell[:2, 2], 0.0, atol=1e-8) or not np.allclose(cell[2, :2], 0.0, atol=1e-8):
        raise ValueError("Align the layer in xy and vacuum along z")
    if atoms.constraints:
        raise ValueError(
            "Remove selective-dynamics constraints or implement a material-specific relaxation protocol"
        )
    # Keep POSCAR order: potentials must follow the displayed species order, including repeated blocks.
    data = pd.read_csv(points)
    if set(["sample_id"] + FEATURES) - set(data):
        raise ValueError("Sampling CSV needs sample_id, exx, eyy, exy")
    if data.sample_id.duplicated().any() or data[FEATURES].duplicated().any():
        raise ValueError("Duplicate IDs or strains")
    if not np.isfinite(data[FEATURES].to_numpy(float)).all():
        raise ValueError("Invalid strain")
    zero = np.isclose(data[FEATURES].to_numpy(float), 0, atol=1e-10).all(1)
    if zero.sum() != 1:
        raise ValueError("Exactly one unstrained reference is required")
    for sid in data.sample_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(sid)):
            raise ValueError("sample_id must be a safe alphanumeric directory name")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = out / "reference.POSCAR"
    shutil.copyfile(poscar, source)
    data.to_csv(out / "points.csv", index=False)
    protocol = dict(
        material_id=material,
        reference_id=str(data.loc[zero, "sample_id"].iloc[0]),
        poscar_sha256=digest(source),
        points_sha256=digest(out / "points.csv"),
        cutoff_eV=cutoff,
        kmesh=list(mesh),
        shear="tensor",
        strain_units="percent",
        deformation="F=I+epsilon",
        xc="PBE (must use consistent PBE potentials)",
        spin="non-spin-polarized starter; adapt for magnetic materials",
        relaxation="fixed cell ISIF=2",
        potential_species=atoms.get_chemical_symbols(),
    )
    json_write(out / "protocol.json", protocol)
    for row in data.itertuples():
        folder = out / str(row.sample_id)
        (folder / "relax").mkdir(parents=True)
        (folder / "static").mkdir()
        strained = atoms.copy()
        strained.set_cell(cell @ deformation([row.exx, row.eyy, row.exy]).T, scale_atoms=True)
        write(folder / "relax" / "POSCAR", strained, format="vasp", direct=True, sort=False)
        (folder / "relax" / "INCAR").write_text(RELAX_INCAR.format(cutoff=cutoff))
        (folder / "static" / "INCAR").write_text(STATIC_INCAR.format(cutoff=cutoff))
        for stage in ["relax", "static"]:
            (folder / stage / "KPOINTS").write_text(kpoints(mesh))
        json_write(
            folder / "strain.json", dict(sample_id=str(row.sample_id), exx=row.exx, eyy=row.eyy, exy=row.exy)
        )
    json_write(
        out / "input_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in out.rglob("*")
            if p.is_file() and p.name in {"POSCAR", "INCAR", "KPOINTS"}
        },
    )
    # A task script deliberately stops if outputs already exist; resume stages explicitly.
    runner = """#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
: "${POTCAR_PATH:?Set an absolute path to your licensed, correctly ordered POTCAR}"
: "${VASP_COMMAND:?Set the launcher, e.g. srun vasp_std}"
point=${1:?usage: bash run_case.sh SAMPLE_ID}
[[ "$point" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$POTCAR_PATH" = /* ]] || { echo "POTCAR_PATH must be absolute" >&2; exit 2; }
cd -- "$point"
for stage in relax static; do
  cd -- "$stage"
  [[ ! -e vasprun.xml ]] || { echo "Existing vasprun.xml: inspect/resume manually" >&2; exit 2; }
  if [[ "$stage" == static ]]; then cp ../relax/CONTCAR POSCAR; fi
  cp -- "$POTCAR_PATH" POTCAR
  sha256sum POTCAR | cut -d ' ' -f1 > potential.sha256
  bash -c "$VASP_COMMAND" > vasp.stdout 2> vasp.stderr
  python -m strain2d verify-vasp vasprun.xml
  cd ..
done
"""
    (out / "run_case.sh").write_text(runner)
    (out / "array.slurm").write_text("""#!/usr/bin/env bash
#SBATCH --job-name=strain2d
#SBATCH --output=slurm-%A_%a.out
# Add account, partition, time, tasks and memory required by YOUR cluster.
set -euo pipefail
: "${CALC_ROOT:?Set absolute calculation directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-N}"
cd "$CALC_ROOT"
point=$(python -c 'import pandas as pd,sys; print(pd.read_csv("points.csv").sample_id.iloc[int(sys.argv[1])])' "$SLURM_ARRAY_TASK_ID")
bash run_case.sh "$point"
""")
    return protocol


def _number(root, name, default=None):
    found = root.find(f".//parameters//i[@name='{name}']")
    if found is None:
        if default is None:
            raise ValueError(f"Missing XML parameter {name}")
        return default
    return float(found.text)


def parse_vasp(path, *, gap_tolerance=0.01, occupation_tolerance=0.001):
    """Final-step properties, with complete-file and explicit convergence guards.

    Directness is mesh-resolved and undefined for metals. Collinear spin supported;
    spin-orbit/noncollinear outputs rejected by this parser until validated separately.
    """
    root = ET.parse(path).getroot()  # truncated XML must fail
    if any(
        (root.findtext(f".//parameters//i[@name='{k}']") or "F").strip().upper() in {"T", ".TRUE."}
        for k in ["LNONCOLLINEAR", "LSORBIT"]
    ):
        raise ValueError("SOC/noncollinear extraction needs a validated material-specific parser")
    calc = root.findall("calculation")
    if not calc:
        raise ValueError("No completed calculation steps")
    last = calc[-1]
    if len(last.findall("scstep")) >= _number(root, "NELM"):
        raise ValueError("Electronic convergence not established (NELM reached)")
    if not last.findall("scstep"):
        raise ValueError("Missing electronic iterations")
    nsw = _number(root, "NSW", 0)
    if nsw > 0:
        ediffg = _number(root, "EDIFFG")
        force = last.find("varray[@name='forces']")
        if ediffg >= 0 or force is None:
            raise ValueError("Collector requires negative EDIFFG and final ionic forces")
        forces = np.array([[float(v) for v in row.text.split()] for row in force])
        if not np.isfinite(forces).all() or np.linalg.norm(forces, axis=1).max() > abs(ediffg):
            raise ValueError("Ionic force criterion not reached")
    energy = last.find("energy/i[@name='e_0_energy']")
    if energy is None:
        raise ValueError("No final zero-smearing total energy")
    if not np.isfinite(float(energy.text)):
        raise ValueError("Non-finite total energy")
    atom_count = root.findtext("atominfo/atoms")
    if not atom_count or int(atom_count) < 1:
        raise ValueError("Missing atom count")
    eig = last.find(".//eigenvalues/array/set")
    if eig is None:
        raise ValueError("No eigenvalues")
    spin_sets = eig.findall("set")
    if not spin_sets:
        raise ValueError("Missing spin eigenvalue sets")
    capacity = 2.0 if len(spin_sets) == 1 else 1.0
    vb, cb, local_gap = [], [], []
    partial = False
    for spin in spin_sets:
        for kpoint in spin.findall("set"):
            a = np.array([[float(x) for x in row.text.split()] for row in kpoint.findall("r")])
            if a.ndim != 2 or a.shape[1] < 2 or not np.isfinite(a).all():
                raise ValueError("Invalid eigenvalue table")
            energy_k, occ = a[:, 0], a[:, 1]
            partial |= bool(((occ > occupation_tolerance) & (occ < capacity - occupation_tolerance)).any())
            occupied = occ > capacity / 2
            if not occupied.any() or occupied.all():
                raise ValueError("No occupied/empty bands; increase NBANDS or inspect occupations")
            v, c = energy_k[occupied].max(), energy_k[~occupied].min()
            vb.append(v)
            cb.append(c)
            local_gap.append(c - v)
    fundamental = max(0.0, min(cb) - max(vb))
    metallic = partial or fundamental <= 1e-8
    return dict(
        energy_eV=float(energy.text),
        natoms=int(atom_count),
        bandgap_eV=0.0 if metallic else fundamental,
        is_direct=None if metallic else int(min(local_gap) - fundamental <= gap_tolerance),
        min_direct_gap_eV=None if metallic else float(min(local_gap)),
        metallic_or_fractional_occupations=metallic,
    )


def collect(calc, out, stability_tolerance=20.0, gap_tolerance=0.01):
    from ase.io import read

    calc, out = Path(calc), Path(out)
    protocol = json.loads((calc / "protocol.json").read_text())
    points = pd.read_csv(calc / "points.csv")
    if (
        digest(calc / "points.csv") != protocol["points_sha256"]
        or digest(calc / "reference.POSCAR") != protocol["poscar_sha256"]
    ):
        raise ValueError("Reference structure or sampling manifest changed")
    expected_inputs = json.loads((calc / "input_hashes.json").read_text())
    reference_atoms = read(calc / "reference.POSCAR", format="vasp")
    records, rejected, potentials = {}, [], set()
    for point in points.itertuples():
        sid = str(point.sample_id)
        try:
            folder = calc / sid
            for relative, expected_hash in expected_inputs.items():
                if Path(relative).parts[0] == sid and digest(calc / relative) != expected_hash:
                    raise ValueError(f"Prepared input changed: {relative}; regenerate a consistent protocol")
            parse_vasp(folder / "relax" / "vasprun.xml")
            props = parse_vasp(folder / "static" / "vasprun.xml", gap_tolerance=gap_tolerance)
            original = read(folder / "relax" / "POSCAR", format="vasp")
            static_input = read(folder / "static" / "POSCAR", format="vasp")
            relaxed = read(folder / "relax" / "vasprun.xml", format="vasp-xml")
            expected_cell = (
                np.asarray(reference_atoms.cell) @ deformation([point.exx, point.eyy, point.exy]).T
            )
            if not np.allclose(original.cell, expected_cell, atol=1e-6, rtol=0):
                raise ValueError("POSCAR does not match requested strain")
            if not np.allclose(relaxed.cell, original.cell, atol=1e-6, rtol=0):
                raise ValueError("Relaxation changed the strained cell")
            delta = static_input.get_scaled_positions() - relaxed.get_scaled_positions()
            delta -= np.rint(delta)
            if not np.allclose(delta, 0, atol=1e-6, rtol=0):
                raise ValueError("Static input is not the relaxed geometry")
            final = read(folder / "static" / "vasprun.xml", format="vasp-xml")
            if not np.allclose(original.cell, final.cell, atol=1e-6, rtol=0):
                raise ValueError("Cell changed; applied strain not preserved")
            position_delta = static_input.get_scaled_positions() - final.get_scaled_positions()
            position_delta -= np.rint(position_delta)
            if not np.allclose(position_delta, 0, atol=1e-6, rtol=0):
                raise ValueError("Static calculation changed ionic positions")
            if original.get_chemical_symbols() != final.get_chemical_symbols():
                raise ValueError("Atomic species/order changed")
            pot_relax = (folder / "relax" / "potential.sha256").read_text().strip()
            pot_static = (folder / "static" / "potential.sha256").read_text().strip()
            if pot_relax != pot_static or not re.fullmatch(r"[0-9a-f]{64}", pot_static):
                raise ValueError("Potential hash missing or changed between stages")
            for stage in ["relax", "static"]:
                pot = folder / stage / "POTCAR"
                if pot.exists() and digest(pot) != pot_static:
                    raise ValueError("Potential content differs from recorded hash")
            potentials.add(pot_static)
            freq = None
            phonon_path = folder / "phonon" / "phonon_result.json"
            if phonon_path.exists():
                phonon = json.loads(phonon_path.read_text())
                if phonon["reference_static_sha256"] != digest(folder / "static" / "vasprun.xml"):
                    raise ValueError("Stale phonon result from different static calculation")
                freq = float(phonon["min_frequency_cm1"])
                if not np.isfinite(freq):
                    raise ValueError("Non-finite phonon frequency")
            records[sid] = dict(
                sample_id=sid,
                material_id=protocol["material_id"],
                data_source="dft",
                exx=point.exx,
                eyy=point.eyy,
                exy=point.exy,
                **props,
                min_frequency_cm1=freq,
                is_stable=None if freq is None else int(freq >= -abs(stability_tolerance)),
                vasprun_sha256=digest(folder / "static" / "vasprun.xml"),
                potential_sha256=pot_static,
            )
        except (OSError, ValueError, KeyError, ET.ParseError, IndexError) as exc:
            rejected.append(dict(sample_id=sid, reason=str(exc)))
    pd.DataFrame(rejected, columns=["sample_id", "reason"]).to_csv(str(out) + ".rejected.csv", index=False)
    if len(potentials) > 1:
        raise ValueError("Different pseudopotentials across points; split numerical protocols")
    reference = records.get(protocol["reference_id"])
    if reference is None:
        raise ValueError("Unstrained reference missing/unconverged; inspect rejection CSV")
    n = reference["natoms"]
    for rec in records.values():
        if rec["natoms"] != n:
            raise ValueError("Atom count changed across calculations")
        rec["strain_energy_meV_atom"] = (rec["energy_eV"] - reference["energy_eV"]) * 1000 / n
    pd.DataFrame(records.values()).to_csv(out, index=False)
    json_write(
        str(out) + ".provenance.json",
        dict(
            protocol=protocol,
            protocol_sha256=digest(calc / "protocol.json"),
            accepted=len(records),
            rejected=len(rejected),
            stability_tolerance_cm1=abs(stability_tolerance),
            direct_gap_tolerance_eV=gap_tolerance,
        ),
    )
    return len(records), len(rejected)
