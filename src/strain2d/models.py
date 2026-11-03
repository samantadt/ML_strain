"""Classical baselines and masked, scaled multi-task neural ensembles."""

import copy
import random

import numpy as np
import torch
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from torch import nn

from .data import CLASSIFICATION


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class AttentionBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, collect=False):
        z = self.norm1(x)
        z, weights = self.attention(z, z, z, need_weights=collect, average_attn_weights=False)
        x = x + self.drop(z)
        return x + self.drop(self.ff(self.norm2(x))), weights


class MultiTaskNet(nn.Module):
    def __init__(self, architecture, outputs, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.architecture = architecture
        if architecture == "transformer":
            # Three independent tokens, with explicit feature identity; no constant-zero tokens.
            self.value_weight = nn.Parameter(torch.randn(3, d) * 0.02)
            self.identity = nn.Parameter(torch.randn(3, d) * 0.02)
            self.blocks = nn.ModuleList(
                [
                    AttentionBlock(d, cfg["heads"], cfg["feedforward"], cfg["dropout"])
                    for _ in range(cfg["layers"])
                ]
            )
            self.norm = nn.LayerNorm(d)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(3, d), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(d, d), nn.GELU()
            )
        self.output = nn.Linear(d, outputs)

    def forward(self, x, collect=False):
        weights = []
        if self.architecture == "transformer":
            x = x[:, :, None] * self.value_weight[None] + self.identity[None]
            for block in self.blocks:
                x, w = block(x, collect)
                if collect:
                    weights.append(w)
            x = self.norm(x).mean(dim=1)
        else:
            x = self.mlp(x)
        return self.output(x), weights


def task_loss(pred, truth, classification):
    """Equal weight per observed task; missing values never enter arithmetic."""
    losses = []
    for j, is_class in enumerate(classification):
        mask = torch.isfinite(truth[:, j])
        if mask.any():
            if is_class:
                loss = nn.functional.binary_cross_entropy_with_logits(pred[mask, j], truth[mask, j])
            else:
                loss = nn.functional.mse_loss(pred[mask, j], truth[mask, j])
            losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


class Surrogate:
    def __init__(self, name, targets, config):
        self.name, self.targets, self.config = name, list(targets), copy.deepcopy(config)
        self.is_class = [t in CLASSIFICATION for t in targets]
        self.histories, self.estimators, self.members = [], {}, []

    def fit(self, x, y, xv, yv):
        if self.name in {"transformer", "mlp"}:
            return self._fit_neural(x, y, xv, yv)
        for t in self.targets:
            mask = np.isfinite(y[t].to_numpy(float))
            yt = y.loc[mask, t].to_numpy(float)
            cls = t in CLASSIFICATION
            if cls and len(np.unique(yt)) < 2:
                raise ValueError(f"Training {t} requires both classes")
            if self.name == "dummy":
                model = DummyClassifier(strategy="prior") if cls else DummyRegressor()
            elif self.name == "polynomial":
                estimator = LogisticRegression(C=1.0, max_iter=2000) if cls else Ridge(alpha=0.1)
                model = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), estimator)
            elif self.name == "random_forest":
                kwargs = dict(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=self.config["seed"],
                    n_jobs=self.config["threads"],
                )
                model = RandomForestClassifier(**kwargs) if cls else RandomForestRegressor(**kwargs)
            elif self.name == "svr":
                # Both feature and regression-output scales learned on training only.
                if cls:
                    model = make_pipeline(
                        StandardScaler(), SVC(C=10.0, probability=True, random_state=self.config["seed"])
                    )
                else:
                    from sklearn.compose import TransformedTargetRegressor

                    model = TransformedTargetRegressor(
                        regressor=make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.02)),
                        transformer=StandardScaler(),
                    )
            elif self.name == "xgboost":
                from xgboost import XGBClassifier, XGBRegressor

                kwargs = dict(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=self.config["seed"],
                    n_jobs=self.config["threads"],
                )
                model = XGBClassifier(**kwargs) if cls else XGBRegressor(**kwargs)
            else:
                raise ValueError(f"Unknown model {self.name}")
            model.fit(x[mask], yt)
            self.estimators[t] = model
        return self

    def _fit_neural(self, x, y, xv, yv):
        cfg = self.config["neural"]
        torch.set_num_threads(self.config["threads"])
        self.x_scaler = StandardScaler().fit(x)
        ya, yva = y[self.targets].to_numpy(float), yv[self.targets].to_numpy(float)
        self.y_mean, self.y_scale = np.zeros(len(self.targets)), np.ones(len(self.targets))
        for j, cls in enumerate(self.is_class):
            if not cls:
                self.y_mean[j] = np.nanmean(ya[:, j])
                self.y_scale[j] = max(np.nanstd(ya[:, j]), 1e-8)
        xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        xvt = torch.tensor(self.x_scaler.transform(xv), dtype=torch.float32)
        yt = torch.tensor((ya - self.y_mean) / self.y_scale, dtype=torch.float32)
        yvt = torch.tensor((yva - self.y_mean) / self.y_scale, dtype=torch.float32)
        for member in range(cfg["members"]):
            seed_all(self.config["seed"] + member)
            net = MultiTaskNet(self.name, len(self.targets), cfg)
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
            )
            best, best_state, stale, history = np.inf, None, 0, []
            for epoch in range(cfg["epochs"]):
                net.train()
                order = torch.randperm(len(xt))
                for batch in order.split(cfg["batch_size"]):
                    loss = task_loss(net(xt[batch])[0], yt[batch], self.is_class)
                    if loss is None:
                        continue
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                net.eval()
                with torch.no_grad():
                    vl = float(task_loss(net(xvt)[0], yvt, self.is_class))
                if not np.isfinite(vl):
                    raise RuntimeError("Non-finite validation loss")
                history.append(vl)
                if vl < best - 1e-7:
                    best, best_state, stale = vl, copy.deepcopy(net.state_dict()), 0
                else:
                    stale += 1
                if stale >= cfg["patience"]:
                    break
            net.load_state_dict(best_state)
            net.eval()
            self.members.append(net)
            self.histories.append(history)
        return self

    def predict_members(self, x):
        x = np.asarray(x, float)
        if not self.members:
            predictions = []
            for t in self.targets:
                model = self.estimators[t]
                p = model.predict_proba(x)[:, 1] if t in CLASSIFICATION else model.predict(x)
                predictions.append(p)
            return np.stack(predictions, axis=1)[None]
        xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        predictions = []
        with torch.no_grad():
            for net in self.members:
                parts = []
                for batch in xt.split(2048):
                    raw = net(batch)[0]
                    for j, cls in enumerate(self.is_class):
                        if cls:
                            raw[:, j] = raw[:, j].sigmoid()
                    parts.append(raw.numpy())
                predictions.append(np.concatenate(parts) * self.y_scale + self.y_mean)
        return np.stack(predictions)

    def predict(self, x):
        return self.predict_members(x).mean(0)

    def attention(self, x):
        if self.name != "transformer":
            raise ValueError("Attention requires the Transformer")
        xt = torch.tensor(self.x_scaler.transform(x), dtype=torch.float32)
        results = []
        with torch.no_grad():
            for net in self.members:
                _, weights = net(xt, collect=True)
                # layers × heads × query × key, averaged only over samples
                results.append(torch.stack(weights).mean(1).numpy())
        return np.stack(results)  # members × layers × heads × 3 × 3

    def parameter_count(self):
        return sum(p.numel() for p in self.members[0].parameters()) if self.members else None
