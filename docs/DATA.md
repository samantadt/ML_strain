# Dataset contract

Train one material, phase, reference structure, charge/spin state and numerical protocol at a time. Combine multiple calculations only when the quantities have the same physical definition and convergence level. Percent versus fractional strain cannot be reliably inferred from small numbers: verify the conversion yourself before importing.

| Column | Required | Definition |
|---|---|---|
| sample_id | training | Unique stable identifier; IDs are retained in test artifacts |
| material_id | training | One constant identifier including material/phase/protocol version |
| data_source | training | Exactly `dft`, `experiment` or `synthetic`; do not mix sources |
| exx, eyy, exy | always | Percent, symmetric tensor shear (`exy = gamma_xy/2`) |
| bandgap_eV | optional target | Fundamental gap on a converged electronic k mesh; 0 for metals |
| strain_energy_meV_atom | optional target | `(E_strained - E_zero)*1000/N`, with consistent zero-smearing energies |
| is_direct | optional target | 0/1; absent for metals or unknown; VASP collector uses minimum direct gap minus fundamental gap ≤0.01 eV |
| is_stable | optional target | 0/1; phonon minimum frequency ≥ −tolerance, with an explicitly recorded tolerance |

At least one target and one observed label per row are required. Missing target values are blank, not 0. Non-numeric values, infinity, repeated strain tensors and mixed material/source identities are rejected. Do not label a failed calculation as unstable: numerical failure and a physical imaginary mode are different outcomes.

Strain tensors are rounded to eight decimal percentage points for duplicate detection. Replicates should be aggregated with a separate uncertainty record; nearby strains can still be correlated. The radial holdout is a stress test, not a guarantee of independent sampling.

Minimum five observed labels per task in training, validation and calibration are enforced. Regression calibration at alpha=0.1 requires at least nine observed calibration labels. The five-way protocol requires at least 60 rows; a scientifically credible four-task study usually needs hundreds to thousands. If partial labels make the common split unsupported, the program stops and requests more representative data rather than silently skipping targets. Change the selection target explicitly for a study without bandgap labels.

Prediction/query CSVs need only `exx,eyy,exy`. Results report the trained material identifier, source, support flags, predicted values, classification scores and regression intervals. The code does not silently convert negative extrapolated gaps or energies to plausible values.

Additional provenance columns can be kept in the input and saved split table; only the three strain features enter the model. Keep original outputs, potential versions/hashes, relaxation settings, energy definitions, convergence studies and rejected-point reasons with the study.

## Example real-data header

```csv
sample_id,material_id,data_source,exx,eyy,exy,bandgap_eV,strain_energy_meV_atom,is_direct,is_stable
```

This file intentionally supplies no invented DFT row. `examples/SYNTHETIC_demo.csv` is a runnable schema example with explicit synthetic provenance.
