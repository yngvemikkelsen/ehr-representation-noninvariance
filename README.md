# ehr-representation-noninvariance

Analysis code for:

> Mikkelsen Y. *Representation non-invariance with bounded performance loss
> in clinical prediction: an empirical EHR study across MIMIC-IV and
> eICU-CRD.* Manuscript submitted for publication.

The study shows that a clinical EHR feature can be recorded on structurally
non-equivalent categorical supports while downstream prediction performance
stays within a small, bounded loss, so that successful external validation
cannot establish that a model's inputs carry the same meaning at a new site.
It also describes a feature-level representation audit that detects such
differences without outcome data.

Every result reported in the manuscript is produced by a script in
`scripts/`. The mapping is given below.

## Data

No data are included or redistributed. Both databases require
[PhysioNet](https://physionet.org) credentialed access and a signed data-use
agreement:

- **MIMIC-IV v3.1** — `hosp/` and `icu/` modules
- **eICU-CRD v2.0**

The scripts read the files exactly as PhysioNet distributes them (`.csv` or
`.csv.gz`).

> **Do not commit anything written to `cache/`.** It is derived from
> credentialed patient-level data and falls under the same data-use agreement.
> `.gitignore` excludes it.

## Requirements

[uv](https://docs.astral.sh/uv/). Each script declares its own dependencies
inline (PEP 723), so no environment setup is needed; `uv run` resolves them on
first use. Python 3.11 or later.

## Running

```bash
export MIMIC_IV_DIR=/path/to/physionet.org/files/mimiciv/3.1
export EICU_CRD_DIR=/path/to/physionet.org/files/eicu-crd/2.0
bash run_all.sh
```

Scripts can also be run individually from the repository root, for example
`uv run scripts/rerun.py`. All intermediate and final outputs are written to
`cache/`. `lung_crosstab_full.py` must be run before `verify_kappa.py`.

### Runtime

Observed on an Apple M2 laptop. Each full pass over MIMIC-IV `chartevents`
(433 million rows) takes about seven minutes. `rerun.py`, which produces
Tables 2 and 3, completes in about 30 minutes including its own two passes.

## Scripts and the results they produce

| Script | Manuscript result |
|---|---|
| `noise_feature_check.py` | Statistical analysis: implementation check: an uninformative 20-level feature gives a positive in-sample and negative held-out gain. No clinical data. |
| `corr_heatmap.py` | Within-MIMIC representation analysis: selection scan: the lung-sound presence indicators are the most strongly correlated pair among the 40 most frequently charted items (r = 0.996). |
| `check_lungsounds.py` | Results: co-population: 93,596 of 93,603 stays have both lower-lobe fields; 99.64% of right-lobe entries have a same-timestamp left-lobe entry. |
| `lung_crosstab_full.py` | Results: full right × left cross-tabulation; 15-string union; κ = 0.8704. |
| `verify_kappa.py` | Results: kappa 0.8801 after merging the side-specific spellings; 9,890 reclassified pairs. |
| `caregiver_check.py` | Results: the same `caregiver_id` entered both fields in 99.91% of 1,806,527 pairs. |
| `eicu_breath_sounds.py` | Results: eICU-CRD co-population (99.76%), pooled κ = 0.9009, per-hospital range, and within-eICU vocabulary comparison (mean Jaccard 0.838). |
| `rerun.py` | Results: Table 2, the sensitivity analysis, Table 3, the patient-level permutation noise floor, and verification of the fitter against L-BFGS. |

`rerun.py` implements the downstream experiment in a single pipeline:
cross-validation folds and bootstrap resampling grouped by patient
(`subject_id`); standardisation and imputation estimated inside each training
fold; one ridge-penalised logistic fitter with a constant penalty
(λ = 10⁻⁴) for every table and model specification; AUC on pooled
out-of-fold predictions for both point estimates and intervals; and, within
each stay-hour, the last value by `charttime` with `storetime` as tie-breaker.
It runs a self-test first and stops before the data scan if the fitter or the
bootstrap weighting fails. `uv run scripts/rerun.py --selftest` runs the test
alone without clinical data, and `--dry-run` runs the full analysis on
synthetic data.

## Superseded scripts

`superseded/` contains the scripts that produced the downstream results in
release 1.0.1. They were replaced by `rerun.py` after pre-submission review
identified four problems: folds grouped by ICU stay rather than patient,
standardisation estimated on the full sample, a penalty that scaled with the
number of columns (confounding model flexibility with regularisation
strength), and within-hour values selected by file order. They are retained
for provenance and do not produce any result in the current manuscript. The
1.0.1 release remains archived under its own DOI.

## Notes on provenance

- **Selection was exploratory.** The lung-sound fields were identified by the
  scan in `corr_heatmap.py`; the manuscript presents them as a worked
  demonstration, not as a prevalence estimate.
- **Margin.** The 10% negligibility margin was defined after the point
  estimates were known and is not a preregistered threshold; the manuscript
  reports sensitivity to a 5% one-sided criterion.

## Citation

See `CITATION.cff`.

## Licence

MIT — see `LICENSE`.
