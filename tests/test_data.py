import numpy as np
import pytest

from strain2d.data import FEATURES, domain_flags, read_data, split_data
from strain2d.metrics import conformal_radius, score
from strain2d.sampling import deformation, lhs, synthetic


def test_design_and_tensor_shear():
    a = lhs(100, seed=3)
    assert a.equals(lhs(100, seed=3))
    assert np.allclose(a.loc[0, FEATURES].astype(float), 0)
    assert a[FEATURES].to_numpy().min() >= -2
    assert a[FEATURES].to_numpy().max() <= 10
    f = deformation([2, 3, 4])
    assert f[0, 1] == f[1, 0] == 0.04  # tensor, not engineering shear
    assert np.linalg.det(f) > 0
    with pytest.raises(ValueError):
        deformation([-101, 0, 0])


def test_schema_and_splits(tmp_path):
    path = tmp_path / "data.csv"
    df = synthetic(200)
    df.loc[3, "is_stable"] = np.nan
    df.to_csv(path, index=False)
    df = read_data(path)
    splits = split_data(df)
    assert sum(map(len, splits.values())) == len(df)
    assert len(set(np.concatenate(list(splits.values())))) == len(df)
    radius = np.sqrt(df.exx**2 + df.eyy**2 + 2 * df.exy**2)
    assert radius.iloc[splits["test_outer"]].min() >= radius.iloc[splits["train"]].max()
    bad = df.copy()
    bad.loc[1, FEATURES] = bad.loc[0, FEATURES].to_numpy()
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Repeated"):
        read_data(path)
    bad = df.copy()
    bad.loc[0, "bandgap_eV"] = np.inf
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Non-finite"):
        read_data(path)
    bad = df.copy()
    bad.loc[0, "data_source"] = "dft"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="mix"):
        read_data(path)


def test_hull_and_conformal():
    ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    box, hull = domain_flags(ref, [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [2, 2, 2]])
    assert box.tolist() == [False, False, True]
    assert hull.tolist() == [False, True, True]
    assert conformal_radius(np.arange(10), np.zeros(10), 0.1) == 9
    with pytest.raises(ValueError, match="Too few"):
        conformal_radius([1, 2], [0, 0], 0.1)
    assert score(np.array([1, 1]), np.array([0.7, 0.9]), True)["roc_auc"] is None


def test_missing_labels_and_invalid_source(tmp_path):
    path = tmp_path / "bad.csv"
    frame = synthetic(100)
    frame.loc[0, "is_stable"] = 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="0/1"):
        read_data(path)
    frame = synthetic(100)
    frame.loc[0, ["bandgap_eV", "strain_energy_meV_atom", "is_direct", "is_stable"]] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least one label"):
        read_data(path)
