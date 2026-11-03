import json

import pandas as pd
import pytest
import yaml

from strain2d.data import FEATURES
from strain2d.sampling import synthetic
from strain2d.workflow import load_bundle, predict_frame, screen, sensitivity, suggest, train


def test_full_run_selection_artifacts_and_tamper(tmp_path):
    df = synthetic(200, seed=1)
    data = tmp_path / "labels.csv"  # synthetic marker is a column, not a filename guess
    df.to_csv(data, index=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(dict(models=["dummy", "polynomial", "random_forest"])))
    run = tmp_path / "run"
    manifest = train(data, run, cfg)
    assert manifest["selection_partition"] == "validation"
    assert manifest["data_source"] == "synthetic"
    assert manifest["selected_model"] != "dummy"
    validation = json.loads((run / "validation_metrics.json").read_text())
    expected = min(validation, key=lambda name: validation[name]["bandgap_eV"]["mae"])
    assert manifest["selected_model"] == expected
    splits = pd.read_csv(run / "dataset_with_split.csv")
    assert splits.sample_id.nunique() == len(df)
    assert (run / "figures" / "bandgap_eV.png").exists()
    assert "SYNTHETIC DATA" in (run / "report.md").read_text()
    bundle = load_bundle(run)
    result = predict_frame(bundle, df[FEATURES].iloc[:10])
    assert (result.upper_bandgap_eV >= result.pred_bandgap_eV).all()
    assert not result.isna().any().any()
    points = tmp_path / "points.csv"
    synthetic(80, seed=15)[FEATURES].to_csv(points, index=False)
    assert suggest(run, points, tmp_path / "next.csv", 8) == 8
    assert len(pd.read_csv(tmp_path / "next.csv").drop_duplicates(FEATURES)) == 8
    sensitivity(run, points, tmp_path / "sensitivity.csv")
    assert "dEg_dexy_eV_per_percent" in pd.read_csv(tmp_path / "sensitivity.csv")
    screen(run, points, tmp_path / "screen.csv", 0, 20, 0.5)
    with pytest.raises(FileExistsError):
        train(data, run, cfg)
    with open(run / "model.joblib", "ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(run)


def test_test_labels_do_not_select_the_model(tmp_path):
    from strain2d.data import split_data

    df = synthetic(140, seed=10)
    parts = split_data(df)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"models": ["dummy", "polynomial"]}))
    data = tmp_path / "data.csv"
    df.to_csv(data, index=False)
    first = train(data, tmp_path / "first", cfg)
    test_rows = list(parts["test_interior"]) + list(parts["test_outer"])
    df.loc[test_rows, "bandgap_eV"] += 100
    df.to_csv(data, index=False)
    second = train(data, tmp_path / "second", cfg)
    assert first["selected_model"] == second["selected_model"]
    a = json.loads((tmp_path / "first" / "validation_metrics.json").read_text())
    b = json.loads((tmp_path / "second" / "validation_metrics.json").read_text())
    assert a == b
    m1 = json.loads((tmp_path / "first" / "metrics.json").read_text())
    m2 = json.loads((tmp_path / "second" / "metrics.json").read_text())
    winner = first["selected_model"]
    assert (
        m2["test_interior"][winner]["bandgap_eV"]["mae"]
        > m1["test_interior"][winner]["bandgap_eV"]["mae"] + 90
    )
