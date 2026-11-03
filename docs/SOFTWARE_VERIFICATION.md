# Software verification record

Executed on 3 November 2026, Linux x86_64, Python 3.12.14, CPU.

## Completed checks

- **15 tests passed**, with no skipped tests, in the clean virtual environment, including the optional XGBoost test.
- `ruff check src tests`: passed.
- `pip check`: no broken requirements in the isolated environment.
- The exact dependency snapshot installed in a clean environment; the saved environment install script was also executed successfully.
- CLI sampling, schema validation, six-model smoke training, prediction, permutation analysis, sensitivity, screening and 20-point acquisition completed.
- The default 128-dimensional, eight-head, four-layer, 256-feed-forward Transformer configuration trained successfully; measured parameter count: **531,460** per member.
- Optional XGBoost training completed using the official CPU package.
- A five-case VASP input pilot was generated; generated runner scripts passed shell syntax checks in the test suite.
- Phonopy preparation/force-constant APIs were exercised with controlled harmonic forces. VASP parsing was exercised with constructed XML fixtures and substituted structure readers where documented in tests.
- Notebook code cells compiled; Jupyter UI/kernel execution was not performed. Equivalent CLI calculations were executed.

## Synthetic training runs

Every run below used the same explicitly synthetic 400-row demonstration, with 220 training, 40 validation, 40 calibration, 40 interior-test and 60 outer-test rows. The functions were invented for software verification; these errors are **not material results**.

| Configuration | Selected on validation | MLP epochs per member | Transformer epochs per member |
|---|---|---|---|
| smoke | polynomial | 80, 80 | 33, 58 |
| default / paper-sized architecture | polynomial | 600, 600, 600 | 184, 310, 300 |
| optional XGBoost benchmark | polynomial | not included | not included |

The polynomial baseline wins because the synthetic regression functions are predominantly quadratic. This is a useful control; the Transformer is not declared superior by design.

## Not executed / not established

- No VASP executable or university cluster job was run; no cluster connectivity or licensed potential was configured.
- No real strained-material dataset was supplied or generated.
- No experimental/material-specific accuracy, stability window, causal strain mechanism, or paper numerical reproduction is established.
- This record describes local validation before GitHub publication. The supplied GitHub Actions workflow runs separately after publication; its outcome is not established by the local checks.
- Environment installation/startup instructions are saved as a draft; publication and fresh-task restoration have not been performed.

## Included demonstration

`examples/SYNTHETIC_verification_run/` contains the full smoke run, including its models, report, exact split, predictions, figures, metrics, hashes and training history. It can be loaded using `strain2d predict --run examples/SYNTHETIC_verification_run ...` after installation. Read the synthetic provenance in every report. Full-size and optional-boosting runs remain in the local ignored `runs/` directory; their execution is summarized above.

Rerun using the documented configs and a new output directory. Wall times depend on hardware; floating-point results across different dependency versions/platforms can differ. The locked snapshot targets this tested Linux/Python environment.
