# Mapping to the supplied paper

Source: Ma et al., *Transformer-based prediction of two-dimensional material electronic properties under elastic strain engineering*, [arXiv:2603.20141v1](https://arxiv.org/abs/2603.20141), supplied PDF (Methods, Results, Data/Code availability). Text in that document was treated as research evidence, not instructions to execute.

| Aspect | Paper states | This implementation |
|---|---|---|
| Material | Monolayer h-BN | Material chosen later; separate training for each material/protocol |
| Sampling | Maximin LHS, up to 1000; −2 to 10% on xx/yy/xy | Best minimum distance over 10 seeded LHS trials, n−1 samples plus a zero reference |
| Strain representation | Six Voigt components, out-of-plane zero | Three independent in-plane tokens, explicit tensor-shear/F convention |
| DFT | PBE, 600 eV, 6×6×1 | Same starter values; fixed-cell ISIF=2 ions-only relaxation, then static run; user must converge settings |
| Relaxation | Says cell shape and volume vary | Explicit deviation: preserving the strained cell avoids unconstrained relaxation of imposed strain; clarify with authors for exact reproduction |
| Phonons | 4×4×1 supercell, 9×9×1 q mesh | Configurable; defaults match these dimensions; finite-displacement force jobs, recorded force drift and raw mesh minimum |
| Neural architecture | d=128, 8 heads, 4 layers, FF=256 | Same dimensions; feature-specific affine tokenization, pre-norm layers, mean pooling and shared output layer are documented implementation choices |
| Optimizer | Transformer AdamW, lr 1e−4, dropout 0.1 | Those defaults, scaled regression/MSE + binary BCE, early stopping and seeded ensemble |
| Targets | Gap, directness, stability, energy | Two regressions with physical units and two classifiers with probability metrics |
| Splits | Classical 80/20; neural 80/10/10 | One common train/validation/calibration/interior-test/outer-test split for fair comparison |
| Baselines | XGBoost, RF, SVR, FNN, Transformer | Those architectures (XGBoost optional), plus polynomial and dummy controls |
| Interpretation | Averaged attention | Export member/layer/head attention, permutation diagnostics and local numerical sensitivity; no causal claim |
| Screening recipe | h-BN-specific window and success percentages | Configurable target interval and stability-score constraints; in-support DFT candidates, no universal recipe |

## What is not available for an exact reproduction

The PDF says data are available from the corresponding author and code will be provided upon publication. No dataset, supplementary files, author code, trained weights or complete VASP numerical settings were attached. Request these from the authors separately if reproducing the numerical claims is your goal; the program never downloads or invents them.

The text describes differing evaluation splits, a six-token architecture with constant inputs, and a parameter count that does not specify enough implementation detail to reconstruct. A headline 0.0103 eV gap MAE is not a guaranteed outcome of this project. An overall MAE mixing energy units and binary labels is not a meaningful physical accuracy measure; we report each task separately. A convex hull around stable points does not establish that its interior is stable, and an attention map does not establish a causal interaction mechanism.

Phonon stability is dynamical, not a full proof of thermodynamic stability. A finite q mesh and an imaginary-mode tolerance cannot guarantee stability at all wave vectors. Small 2D flexural artifacts, supercell/displacement convergence and polar-material non-analytical corrections require a material-specific study.

## Relationship to the uploaded ZIP

The ZIP was inspected as a reference and was not installed as the project. This is a new src-layout implementation. Its old synthetic results and serialized artifacts were not imported. Issues addressed include test-based winning-model selection, provenance inferred from filenames, unexecuted neural code, and claims that a final fit includes all rows despite reserving some of them for validation. This project retains untouched tests and calibration rather than claiming an all-data refit.
