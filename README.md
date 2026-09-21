# ehr-representation-noninvariance

Analysis code for:

> Mikkelsen Y. *Representation non-invariance without performance loss in
> clinical prediction: an empirical EHR study across MIMIC-IV and eICU-CRD.*
> Manuscript submitted to the Journal of Biomedical Informatics.

The study shows that a clinical EHR feature can be recorded on structurally
non-equivalent categorical supports while downstream prediction performance
remains equivalent within a pre-specified margin, so that successful external
validation cannot establish that a model's inputs carry the same meaning at a
new site. It also describes a feature-level representation audit that detects
such differences without outcome data.

Every figure reported in the manuscript is produced by a script in
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
./run_all.sh
```

Scripts can also be run individually from the repository root, for example
`uv run scripts/caregiver_check.py`. All intermediate and final outputs are
written to `cache/`.

`cmi_screen.py --extract-only` must be run before `collapse_experiment.py`,
`paired_bootstrap.py` and `robustness.py`, which read the extract it builds.
`lung_crosstab_full.py` must be run before `verify_kappa.py`.

### Runtime

Observed on an Apple M2 laptop. Each full pass over MIMIC-IV `chartevents`
(433 million rows) takes on the order of tens of minutes. The two bootstrap
analyses dominate: `paired_bootstrap.py` with 2,000 replicates across four
fields takes several hours per run, and `robustness.py` with 2,000 replicates
across eight outcome–specification cells takes roughly 7–10 hours.

## Scripts and the results they produce

| Script | Manuscript result |
|---|---|
| `noise_feature_check.py` | §3.6 implementation check: an uninformative 20-level feature gives a positive in-sample and negative held-out gain. No clinical data. |
| `corr_heatmap.py` | §3.2 selection scan: the lung-sound presence indicators are the most strongly correlated pair among the 40 most frequently charted items (r = 0.996). |
| `check_lungsounds.py` | §4.1 co-population: 93,596 of 93,603 stays have both lower-lobe fields; 99.64% of right-lobe entries have a same-timestamp left-lobe entry. |
| `lung_crosstab_full.py` | §4.1 full right × left cross-tabulation; 15-string union; κ = 0.8704. |
| `verify_kappa.py` | §4.1 κ = 0.8801 after merging the side-specific spellings; 9,890 reclassified pairs. |
| `caregiver_check.py` | §4.1 the same `caregiver_id` entered both fields in 99.91% of 1,806,527 pairs. |
| `eicu_breath_sounds.py` | §4.2 eICU-CRD co-population (99.76%), pooled κ = 0.9009, per-hospital range, and within-eICU vocabulary comparison (mean Jaccard 0.838). |
| `cmi_screen.py --extract-only` | Builds the stay-hour extract used by the downstream experiment. |
| `collapse_experiment.py` | §4.3 R0–R3 encodings and the 20-permutation noise floor. |
| `paired_bootstrap.py` | §4.3 Table 2 and paired-bootstrap contrasts; with `--drop-unmatched`, the sensitivity analysis. |
| `robustness.py` | §4.4 Table 3: four outcomes × two model specifications, 16 contrasts. Includes verification of the fitter against L-BFGS (`--selftest`). |

## Notes on provenance

- **Two logistic implementations.** `paired_bootstrap.py` (Table 2) and
  `robustness.py` (Table 3) use separately written fitters with different
  convergence criteria, so the native log score for the primary outcome
  differs slightly between them (0.001168 and 0.001160). The manuscript
  reports this; Table 2 is the reference for the primary result. The
  difference does not affect any contrast.
- **`cmi_screen.py` screening output is not used.** Apart from building the
  extract, the script computes screening statistics with a within-stay
  permutation null that is invalid, because the outcome is near-constant
  within a stay. Those statistics are not reported. The code is retained
  unchanged so that the file is the one that produced the extract; the
  `--extract-only` flag stops before it.
- **Selection was exploratory.** The lung-sound fields were identified by the
  scan in `corr_heatmap.py`; the manuscript presents them as a worked
  demonstration, not as a prevalence estimate.

## Citation

See `CITATION.cff`. Archived release: [ZENODO DOI].

## Licence

MIT — see `LICENSE`.
