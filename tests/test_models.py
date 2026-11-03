import numpy as np
import pytest
import torch

from strain2d.data import FEATURES, TARGETS
from strain2d.models import Surrogate, task_loss
from strain2d.sampling import synthetic
from strain2d.workflow import configuration


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_neural_learns_masks_serializes_and_attends(architecture, tmp_path):
    import joblib

    df = synthetic(160, seed=7)
    df.loc[::7, "is_stable"] = np.nan
    x = df[FEATURES].to_numpy(float)
    cfg = configuration()
    cfg["neural"].update(
        d_model=24, heads=4, layers=1, feedforward=48, epochs=80, patience=20, members=2, learning_rate=0.01
    )
    model = Surrogate(architecture, TARGETS, cfg).fit(x[:120], df.iloc[:120], x[120:], df.iloc[120:])
    p = model.predict(x[120:])
    assert p.shape == (40, 4) and np.isfinite(p).all()
    assert np.all((p[:, 2:] >= 0) & (p[:, 2:] <= 1))
    # Real learning, not just a forward-pass test; the synthetic gap is smooth.
    mae = np.mean(np.abs(p[:, 0] - df.bandgap_eV.to_numpy()[120:]))
    prior_mae = np.mean(np.abs(df.bandgap_eV.iloc[:120].mean() - df.bandgap_eV.to_numpy()[120:]))
    assert mae < prior_mae * 0.8
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    assert np.allclose(joblib.load(path).predict(x[120:]), p, atol=1e-7)
    assert min(model.histories[0]) < model.histories[0][0]
    if architecture == "transformer":
        attention = model.attention(x[120:])
        assert attention.shape == (2, 1, 4, 3, 3)
        assert np.allclose(attention.sum(-1), 1, atol=1e-6)


def test_masked_loss_has_finite_gradients():
    pred = torch.zeros((3, 2), requires_grad=True)
    labels = torch.tensor([[1.0, float("nan")], [float("nan"), 1.0], [2.0, 0.0]])
    loss = task_loss(pred, labels, [False, True])
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[0, 1] == 0


def test_optional_xgboost_baseline():
    pytest.importorskip("xgboost")
    df = synthetic(120)
    x = df[FEATURES].to_numpy(float)
    model = Surrogate("xgboost", TARGETS, configuration()).fit(x[:90], df.iloc[:90], x[90:], df.iloc[90:])
    prediction = model.predict(x[90:])
    assert prediction.shape == (30, 4)
    assert np.isfinite(prediction).all()
    assert ((prediction[:, 2:] >= 0) & (prediction[:, 2:] <= 1)).all()
