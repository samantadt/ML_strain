from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase.build import graphene
from ase.io import read, write

from strain2d.dft import collect, parse_vasp, prepare
from strain2d.sampling import deformation, lhs


def xml_fixture(path, gap=2.0, indirect=False, fractional=False, unconverged=False, ionic=False):
    # Minimal hand-constructed VASP-style XML; not a claim of testing a production VASP file.
    params = (
        '<parameters><i name="NELM">3</i><i name="NSW">'
        + ("10" if ionic else "0")
        + '</i><i name="EDIFFG">-0.01</i></parameters>'
    )
    k1 = f"<set><r>0 2</r><r>{gap + 1 if indirect else gap} {0.3 if fractional else 0}</r></set>"
    k2 = f"<set><r>-1 2</r><r>{gap} 0</r></set>"
    steps = "<scstep/>" * (3 if unconverged else 2)
    forces = '<varray name="forces"><v>0.001 0 0</v><v>0 0 0</v></varray>'
    path.write_text(
        "<modeling>"
        + params
        + "<atominfo><atoms>2</atoms></atominfo><calculation>"
        + steps
        + '<energy><i name="e_0_energy">-10</i></energy>'
        + forces
        + "<eigenvalues><array><set><set>"
        + k1
        + k2
        + "</set></set></array></eigenvalues></calculation></modeling>"
    )


def test_xml_gap_directness_and_convergence(tmp_path):
    path = tmp_path / "vasprun.xml"
    xml_fixture(path)
    assert parse_vasp(path)["bandgap_eV"] == 2
    assert parse_vasp(path)["is_direct"] == 1
    xml_fixture(path, indirect=True)
    assert parse_vasp(path)["is_direct"] == 0
    xml_fixture(path, fractional=True)
    assert parse_vasp(path)["bandgap_eV"] == 0
    assert parse_vasp(path)["is_direct"] is None
    xml_fixture(path, unconverged=True)
    with pytest.raises(ValueError, match="convergence"):
        parse_vasp(path)
    xml_fixture(path, ionic=True)
    assert parse_vasp(path)["energy_eV"] == -10
    path.write_text(path.read_text().replace("0.001 0 0", "0.1 0 0"))
    with pytest.raises(ValueError, match="force"):
        parse_vasp(path)


def test_prepare_keeps_strain_and_no_potentials(tmp_path):
    atoms = graphene(formula="BN", a=2.5, vacuum=10)
    poscar = tmp_path / "POSCAR"
    write(poscar, atoms, format="vasp")
    points = tmp_path / "points.csv"
    lhs(3).to_csv(points, index=False)
    out = tmp_path / "calc"
    protocol = prepare(poscar, points, out, "BN-PBE-test")
    df = pd.read_csv(points)
    for row in df.itertuples():
        actual = read(out / row.sample_id / "relax" / "POSCAR")
        assert np.allclose(actual.cell, np.asarray(atoms.cell) @ deformation([row.exx, row.eyy, row.exy]).T)
        assert "ISIF = 2" in (out / row.sample_id / "relax" / "INCAR").read_text()
    assert not list(out.rglob("POTCAR"))
    assert protocol["reference_id"] == "s00000"
    import subprocess

    for script in ["run_case.sh", "array.slurm"]:
        assert subprocess.run(["bash", "-n", str(out / script)]).returncode == 0


def test_collection_energy_units_and_rejections(tmp_path, monkeypatch):
    atoms = graphene(formula="BN", a=2.5, vacuum=10)
    poscar = tmp_path / "POSCAR"
    write(poscar, atoms, format="vasp")
    points = tmp_path / "points.csv"
    lhs(3).to_csv(points, index=False)
    calc = tmp_path / "calc"
    prepare(poscar, points, calc, "BN-test")
    for sid in ["s00000", "s00001"]:
        for stage in ["relax", "static"]:
            folder = calc / sid / stage
            if stage == "static":
                (folder / "POSCAR").write_text((calc / sid / "relax" / "POSCAR").read_text())
            xml_fixture(folder / "vasprun.xml")
            if sid == "s00001":
                (folder / "vasprun.xml").write_text(
                    (folder / "vasprun.xml").read_text().replace(">-10<", ">-9.8<")
                )
            (folder / "potential.sha256").write_text("a" * 64)
    # Exercise collection with real POSCARs and parser, substituting ASE's full XML structure reader.
    import ase.io

    original_read = ase.io.read

    def read_structure(path, *args, **kwargs):
        if str(path).endswith("vasprun.xml"):
            return original_read(Path(path).parent / "POSCAR", format="vasp")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(ase.io, "read", read_structure)
    out = tmp_path / "data.csv"
    accepted, rejected = collect(calc, out)
    assert (accepted, rejected) == (2, 1)
    frame = pd.read_csv(out)
    assert np.isclose(frame.strain_energy_meV_atom.iloc[1], 100.0)  # .2 eV/2 atoms *1000
    assert frame.is_stable.isna().all()  # no fabricated stable labels without phonons
    assert (tmp_path / "data.csv.rejected.csv").exists()


def test_changed_sampling_manifest_is_rejected(tmp_path):
    atoms = graphene(formula="BN", a=2.5, vacuum=10)
    poscar = tmp_path / "POSCAR"
    write(poscar, atoms, format="vasp")
    points = tmp_path / "points.csv"
    lhs(3).to_csv(points, index=False)
    calc = tmp_path / "calc"
    prepare(poscar, points, calc, "test")
    with open(calc / "points.csv", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="manifest changed"):
        collect(calc, tmp_path / "out.csv")
