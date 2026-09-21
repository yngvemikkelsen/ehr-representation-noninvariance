#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Recompute Cohen's kappa for RLL vs LLL lung sounds with the side-specific
spellings merged, from the full cross-tabulation rather than by hand.

The note currently carries ~0.880, obtained arithmetically from printed
marginals under the assumption that all 264 'Pleural friction' entries pair
with 'Pleural fricton'. That is an upper bound, not a measurement. This
reads the table and computes it.

Alias merges (identical clinical findings, separately configured picklists):
    RLL 'Insp/Exp Wheeze'   ==  LLL 'Ins/Exp Wheeze'
    RLL 'Pleural friction'  ==  LLL 'Pleural fricton'

Reports kappa before and after merging, and the exact number of pairs whose
concordance status the merge changes.

Usage:  ./verify_kappa.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("cache/lung_crosstab_full.csv")

ALIASES = [("Insp/Exp Wheeze", "Ins/Exp Wheeze"),
           ("Pleural friction", "Pleural fricton")]


def kappa(T: pd.DataFrame):
    M = T.values.astype(float)
    n = M.sum()
    po = np.trace(M) / n
    pe = ((M.sum(axis=1) / n) * (M.sum(axis=0) / n)).sum()
    return po, pe, (po - pe) / (1 - pe), n


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found -- run lung_crosstab_full.py first")
    T = pd.read_csv(SRC, index_col=0)
    T.index = [str(i) for i in T.index]
    T.columns = [str(c) for c in T.columns]

    po, pe, k, n = kappa(T)
    print(f"raw table: {T.shape[0]}x{T.shape[1]}, {n:,.0f} paired observations")
    print(f"  po={po:.4f}  pe={pe:.4f}  kappa={k:.4f}")

    # merge each alias pair into a single canonical label on both axes
    M = T.copy()
    changed = 0
    for rll_name, lll_name in ALIASES:
        canon = rll_name
        if rll_name in M.index and lll_name in M.index and rll_name != lll_name:
            M.loc[canon] = M.loc[rll_name] + M.loc[lll_name]
            M = M.drop(index=[x for x in (rll_name, lll_name) if x != canon])
        if rll_name in M.columns and lll_name in M.columns:
            M[canon] = M[rll_name] + M[lll_name]
            M = M.drop(columns=[x for x in (rll_name, lll_name) if x != canon])
        # pairs whose status the merge flips: RLL spelling x LLL spelling
        if rll_name in T.index and lll_name in T.columns:
            flip = int(T.loc[rll_name, lll_name])
            changed += flip
            print(f"  merge {rll_name!r} + {lll_name!r}: "
                  f"{flip:,} pairs become concordant")

    cats = sorted(set(M.index) | set(M.columns))
    M = M.reindex(index=cats, columns=cats, fill_value=0)

    po2, pe2, k2, n2 = kappa(M)
    print(f"\nmerged table: {M.shape[0]}x{M.shape[1]}, {n2:,.0f} observations")
    print(f"  po={po2:.4f}  pe={pe2:.4f}  kappa={k2:.4f}")
    print(f"\npairs reclassified by the merge: {changed:,} "
          f"({changed/n:.4f} of all pairs)")
    print(f"kappa shift: {k:.4f} -> {k2:.4f}")
    print(f"\nvalue for the note: kappa = {k2:.4f} (raw {k:.4f})")


if __name__ == "__main__":
    main()
