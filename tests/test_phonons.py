import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from strain2d.data import digest
from strain2d.phonons import collect_phonons, prepare_phonons


def test_phonopy_prepare_and_force_constant_workflow(tmp_path, monkeypatch):
    # Harmonic fixture tests Phonopy integration, not a material calculation or a real VASP parser.
    import ase.io

    import strain2d.phonons as module

    case = tmp_path / "case"
    (case / "static").mkdir(parents=True)
    (case / "static" / "vasprun.xml").write_text("harmonic-test-only")
    (case / "static" / "INCAR").write_text("ENCUT = 600\n")
    (case / "static" / "potential.sha256").write_text("a" * 64)
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=np.diag([3, 3, 15]), pbc=True)
    original_read = ase.io.read

    def fake_read(path, *args, **kwargs):
        if str(path).endswith("static/vasprun.xml"):
            return atoms.copy()
        if str(path).endswith("vasprun.xml"):
            a = original_read(path.parent / "POSCAR", format="vasp")
            # Known restoring force on displaced atom, with the drift removed by collector.
            equilibrium = np.rint(a.get_scaled_positions() * np.array([2, 2, 1])) / np.array([2, 2, 1])
            delta = a.get_scaled_positions() - equilibrium
            delta -= np.rint(delta)
            forces = -(delta @ a.cell.array) * 2
            a.calc = SinglePointCalculator(a, forces=forces)
            return a
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(ase.io, "read", fake_read)
    monkeypatch.setattr(module, "parse_vasp", lambda p: {})
    n = prepare_phonons(case, supercell=(2, 2, 1))
    assert n > 0
    for folder in (case / "phonon").glob("disp-*"):
        (folder / "vasprun.xml").write_text("harmonic-test-only")
        (folder / "potential.sha256").write_text("a" * 64)
    result = collect_phonons(case, qmesh=(3, 3, 1))
    assert np.isfinite(result["min_frequency_cm1"])
    assert np.load(case / "phonon" / "frequencies_THz.npy").shape[-1] == 3
    assert result["reference_static_sha256"] == digest(case / "static" / "vasprun.xml")
