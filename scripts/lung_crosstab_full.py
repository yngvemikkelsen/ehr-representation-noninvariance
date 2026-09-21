#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Full RLL x LLL lung-sounds cross-tabulation, untruncated.

The earlier run printed .iloc[:8, :8], which covered 86% of paired
observations. The missing 14% sits in categories that sort after
'Ins/Exp Wheeze' alphabetically, plus every off-diagonal cell involving them.
Cohen's kappa computed on a truncated table is biased: dropping categories
removes both agreement and disagreement mass, and the direction is not
predictable a priori.

This recomputes on the complete table and reports:
  - every category with its marginal share on each side
  - kappa on the full table vs the truncated one
  - asymmetry of the discordance (RLL=a,LLL=b vs RLL=b,LLL=a): a systematic
    imbalance would indicate side-specific charting behaviour rather than
    symmetric judgement at a category boundary
  - blank/whitespace values, which pandas.crosstab silently drops

Usage:  ./lung_crosstab_full.py
"""

import csv
import gzip
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))
RLL_ID, LLL_ID = "223987", "223989"


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not under {root}")


def kappa(T: pd.DataFrame) -> tuple[float, float, float]:
    M = T.values.astype(float)
    n = M.sum()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    po = np.trace(M) / n
    pe = ((M.sum(1) / n) * (M.sum(0) / n)).sum()
    return po, pe, (po - pe) / (1 - pe)


def main() -> None:
    pairs = defaultdict(int)
    single = defaultdict(int)
    blank = {"RLL": 0, "LLL": 0}
    seen = defaultdict(dict)     # (stay, charttime) -> side -> value
    n = 0

    with open_t(find(MIMIC / "icu", "chartevents")) as fh:
        for r in csv.DictReader(fh):
            n += 1
            if n % 50_000_000 == 0:
                print(f"  {n:,} rows ...")
            iid = r["itemid"]
            if iid not in (RLL_ID, LLL_ID):
                continue
            side = "RLL" if iid == RLL_ID else "LLL"
            val = (r.get("value") or "").strip()
            if not val:
                blank[side] += 1
                val = "(blank)"
            seen[(r["stay_id"], r["charttime"])][side] = val
    print(f"scanned {n:,} rows")

    for d in seen.values():
        if len(d) == 2:
            pairs[(d["RLL"], d["LLL"])] += 1
        else:
            side = next(iter(d))
            single[side] += 1

    cats = sorted({c for p in pairs for c in p})
    T = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
    for (a, b), c in pairs.items():
        T.loc[a, b] = c
    T.to_csv("cache/lung_crosstab_full.csv")

    N = T.values.sum()
    print(f"\npaired observations: {N:,}")
    print(f"unpaired (one side only at a timestamp): RLL {single.get('RLL',0):,}  "
          f"LLL {single.get('LLL',0):,}")
    print(f"blank values: RLL {blank['RLL']:,}  LLL {blank['LLL']:,}")
    print(f"categories: {len(cats)}")

    print("\n=== marginal shares ===")
    mr, mc = T.sum(1) / N, T.sum(0) / N
    md = pd.DataFrame({"RLL": mr, "LLL": mc, "n_RLL": T.sum(1)})
    print(md.sort_values("RLL", ascending=False).to_string(
        float_format=lambda x: f"{x:.5f}"))

    po, pe, k = kappa(T)
    keep = [c for c in cats[:8]]
    po8, pe8, k8 = kappa(T.loc[keep, keep])
    print("\n=== kappa ===")
    print(f"full  ({len(cats)} cats, {N:,} obs): po={po:.4f} pe={pe:.4f} kappa={k:.4f}")
    print(f"trunc (8 cats, {T.loc[keep,keep].values.sum():,} obs): "
          f"po={po8:.4f} pe={pe8:.4f} kappa={k8:.4f}")

    print("\n=== categories excluded by the earlier truncation ===")
    drop = [c for c in cats if c not in keep]
    for c in drop:
        print(f"  {c:<28} RLL {T.loc[c].sum():>9,}  LLL {T[c].sum():>9,}  "
              f"concordant {T.loc[c, c]:>8,}")

    print("\n=== discordance asymmetry (top pairs) ===")
    rows = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            ab, ba = int(T.loc[a, b]), int(T.loc[b, a])
            if ab + ba >= 500:
                rows.append((ab + ba, a, b, ab, ba, (ab - ba) / (ab + ba)))
    rows.sort(reverse=True)
    print(f"{'total':>9} {'RLL=a,LLL=b':>12} {'RLL=b,LLL=a':>12} {'skew':>7}   pair")
    for tot, a, b, ab, ba, sk in rows[:12]:
        print(f"{tot:>9,} {ab:>12,} {ba:>12,} {sk:>+7.3f}   {a} / {b}")

    print("\nwrote cache/lung_crosstab_full.csv")


if __name__ == "__main__":
    main()
