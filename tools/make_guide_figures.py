"""Rebuild the handbook's teaching diagrams (not material results)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import qmc

OUT = Path(__file__).resolve().parents[1] / "docs" / "guide_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)
BLUE, TEAL, ORANGE, PURPLE = "#275D8C", "#198C82", "#C4762B", "#7055A2"


def save(fig, name):
    fig.savefig(OUT / name, dpi=190, bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 6))
ax.set(xlim=(0, 12), ylim=(0, 6))
ax.axis("off")
boxes = [
    (0.3, 3.6, "1. Choose the question", "One material, protocol, strain range", BLUE),
    (4.3, 3.6, "2. Generate reference labels", "VASP + converged phonons", BLUE),
    (8.3, 3.6, "3. Organize the dataset", "Inputs, targets, quality, fixed split", BLUE),
    (8.3, 0.8, "4. Fit and select", "Train weights; choose on validation", TEAL),
    (4.3, 0.8, "5. Evaluate and propose", "Calibrate; test; screen candidates", TEAL),
    (0.3, 0.8, "6. Verify physically", "Fresh DFT/phonons; record outcomes", ORANGE),
]
for x, y, title, body, color in boxes:
    ax.add_patch(
        FancyBboxPatch((x, y), 3.35, 1.35, boxstyle="round,pad=0.12", ec=color, fc=color + "12", lw=1.7)
    )
    ax.text(x + 1.675, y + 0.94, title, ha="center", va="center", weight="bold", fontsize=11, color=color)
    ax.text(x + 1.675, y + 0.45, body, ha="center", va="center", fontsize=9)
for start, end in [
    ((3.8, 4.25), (4.1, 4.25)),
    ((7.8, 4.25), (8.1, 4.25)),
    ((9.98, 3.4), (9.98, 2.4)),
    ((8.1, 1.45), (7.8, 1.45)),
    ((4.1, 1.45), (3.8, 1.45)),
]:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 2, "color": "#596774"})
ax.text(
    6,
    5.6,
    "A surrogate organizes calculations; it does not replace physical evidence",
    ha="center",
    fontsize=14,
    weight="bold",
)
ax.text(
    6,
    0.15,
    "Iterative studies require versioned data and genuinely fresh evaluation evidence.",
    ha="center",
    fontsize=10,
    color="#596774",
)
save(fig, "workflow.png")

fig, ax = plt.subplots(figsize=(11, 3.3))
parts = [
    ("Training", 550, BLUE),
    ("Validation", 100, TEAL),
    ("Calibration", 100, PURPLE),
    ("Interior test", 100, ORANGE),
    ("Outer test", 150, "#9B4455"),
]
left = 0
for name, count, color in parts:
    ax.barh(0, count, left=left, height=0.65, color=color, edgecolor="white", linewidth=2)
    ax.text(
        left + count / 2, 0, str(count), ha="center", va="center", color="white", weight="bold", fontsize=13
    )
    left += count
ax.set(xlim=(0, 1000), ylim=(-0.8, 0.8), xlabel="Rows in a 1000-example dataset", yticks=[])
ax.set_title("Five distinct roles, one common partition for every model", pad=15, weight="bold")
for spine in ["left", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.legend(
    [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in parts],
    [name for name, _, _ in parts],
    loc="lower center",
    bbox_to_anchor=(0.5, -0.65),
    ncol=3,
    frameon=False,
)
save(fig, "splits.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
truth = np.array([4.0, 4.5, 5.0])
pred = np.array([4.1, 4.3, 5.3])
axes[0].plot([3.8, 5.5], [3.8, 5.5], "--", color="#888888", label="Perfect prediction")
axes[0].scatter(truth, pred, color=BLUE, s=65, zorder=3)
for x, y in zip(truth, pred):
    axes[0].plot([x, x], [x, y], color=ORANGE, lw=2)
axes[0].set(
    xlabel="Reference gap (eV)",
    ylabel="Predicted gap (eV)",
    title="Distance from the diagonal is error",
    xlim=(3.8, 5.5),
    ylim=(3.8, 5.5),
)
axes[0].legend(fontsize=9)
axes[1].bar(["A", "B", "C"], pred - truth, color=[TEAL, ORANGE, TEAL], width=0.6)
axes[1].axhline(0, color="#777777", lw=0.8)
axes[1].set(ylabel="Prediction − reference (eV)", title="Signed residuals", ylim=(-0.3, 0.4))
fig.suptitle("Arithmetic illustration: MAE = 0.200 eV; RMSE ≈ 0.216 eV", fontsize=13, weight="bold")
fig.tight_layout()
save(fig, "errors.png")

rng = np.random.default_rng(17)
x = np.linspace(0, 1, 18)
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.16, len(x))
xx = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
for ax, degree, title in zip(
    axes, [1, 5, 16], ["Too rigid: underfit", "A useful middle ground", "Very flexible: can overfit"]
):
    fit = np.polynomial.Polynomial.fit(x, y, degree)
    ax.plot(xx, np.sin(2 * np.pi * xx), "--", color="#777777", lw=1, label="Toy generating function")
    ax.plot(xx, fit(xx), color=BLUE, lw=1.8, label=f"Degree {degree} fit")
    ax.scatter(x, y, color=ORANGE, s=17, zorder=3, label="Noisy training examples")
    ax.set(title=title, xlabel="Toy input", ylim=(-2, 2))
axes[0].set_ylabel("Toy output (arbitrary units)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    ["Toy generating function", "Fitted function", "Noisy training examples"],
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.04),
    frameon=False,
    fontsize=9,
)
fig.suptitle(
    "Teaching illustration only · Display range limited to −2 to 2 for comparison", fontsize=12, weight="bold"
)
fig.tight_layout(rect=(0, 0.08, 1, 0.94))
save(fig, "model_complexity.png")

fig, axes = plt.subplots(1, 3, figsize=(11.7, 4))
grid = np.array(np.meshgrid(np.linspace(0.05, 0.95, 6), np.linspace(0.05, 0.95, 6))).reshape(2, -1).T
random = rng.uniform(0, 1, (36, 2))
latin = qmc.LatinHypercube(2, seed=42).random(36)
for ax, points, title in zip(
    axes, [grid, random, latin], ["Grid: 6 × 6", "Independent random: 36", "Latin hypercube: 36"]
):
    ax.scatter(points[:, 0], points[:, 1], s=22, color=TEAL)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Design coordinate 1", title=title, aspect="equal")
axes[0].set_ylabel("Design coordinate 2")
fig.suptitle(
    "Sampling illustration in two dimensions (the actual project uses three)", weight="bold", fontsize=12
)
fig.tight_layout()
save(fig, "sampling.png")
print(f"Wrote teaching diagrams to {OUT}")
