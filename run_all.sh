#!/usr/bin/env bash
# Reproduce every result reported in the manuscript, in dependency order.
# Requires uv (https://docs.astral.sh/uv/) and local PhysioNet copies of
# MIMIC-IV v3.1 and eICU-CRD v2.0. See README.md.
set -euo pipefail

: "${MIMIC_IV_DIR:?Set MIMIC_IV_DIR to your local MIMIC-IV v3.1 directory}"
: "${EICU_CRD_DIR:?Set EICU_CRD_DIR to your local eICU-CRD v2.0 directory}"

cd "$(dirname "$0")"
mkdir -p cache

run() { echo; echo "=== $* ==="; uv run "scripts/$@"; }

# --- no clinical data required ---------------------------------------------
run noise_feature_check.py                      # estimator check

# --- representation analyses -------------------------------------------------
run corr_heatmap.py --dataset mimic              # selection scan, r = 0.996
run check_lungsounds.py                          # co-population, 99.64%
run lung_crosstab_full.py                        # 15-string union, kappa 0.8704
run verify_kappa.py                              # kappa 0.8801, 9,890 pairs
run caregiver_check.py                           # 99.91% same caregiver_id
run eicu_breath_sounds.py                        # eICU: 99.76%, kappa 0.9009

# --- downstream experiment: Tables 2 and 3 ------------------------------------
run rerun.py                                     # self-test, scan, Tables 2-3

echo; echo "All analyses complete. Outputs are in cache/."
