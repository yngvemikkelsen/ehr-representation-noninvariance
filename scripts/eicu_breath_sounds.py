#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
eICU breath sounds: the MIMIC lung-sounds analysis, plus the cross-site rung.

eICU nurseAssessment carries the same construct in the same shape:
    Nursing Assessment|Respiratory|Breath Sounds|{Right,Left} {Upper,Lower}
Four quadrant fields instead of MIMIC's two, across 208 hospitals.

Three tests, one pass:

A. CO-POPULATION (MIMIC replication, within site)
   Are the quadrant fields written by one act? Measured as the share of
   Right Lower entries with a Left Lower entry at the same offset, computed
   per hospital. MIMIC gave 0.9964 within one institution. If eICU hospitals
   vary here, co-population is itself a site property.

B. BILATERAL AGREEMENT (MIMIC replication)
   Cohen's kappa for Right Lower vs Left Lower values at coincident offsets,
   pooled and per hospital. MIMIC gave kappa 0.870 on the raw vocabulary,
   ~0.880 after merging side-specific spellings.

C. VOCABULARY DIVERGENCE ACROSS HOSPITALS  <-- the test MIMIC cannot run
   Does the same named field carry different permitted values at different
   hospitals? Reported as pairwise Jaccard between hospital vocabularies,
   the count of site-private values, and disjoint near-duplicate strings
   (edit distance <= 2, used by non-overlapping hospital sets) -- the
   cross-site equivalent of MIMIC's "Pleural friction" / "Pleural fricton".

CAVEAT ON NEGATIVE RESULTS: eICU was normalised during construction, so
uniform vocabularies may reflect the research-release ETL rather than the
source systems. Divergence is informative; absence of divergence is weak.

Usage:  ./eicu_breath_sounds.py
"""

import argparse
import csv
import gzip
import itertools
import re
import sys
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

EICU = Path(os.environ.get("EICU_CRD_DIR", "EICU_CRD_DIR_not_set"))
QUADS = ("Right Lower", "Left Lower", "Right Upper", "Left Upper")


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = EICU / (name + ext)
        if p.exists():
            return p
    sys.exit(f"{name} not found under {EICU}")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def edit_le(a: str, b: str, k: int) -> bool:
    if abs(len(a) - len(b)) > k:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] <= k


def kappa(T: pd.DataFrame):
    M = T.values.astype(float)
    n = M.sum()
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    po = np.trace(M) / n
    pe = ((M.sum(axis=1) / n) * (M.sum(axis=0) / n)).sum()
    k = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return po, pe, k, int(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stays", type=int, default=50)
    ap.add_argument("--min-n", type=int, default=5,
                    help="drop a value at a hospital if seen fewer times")
    a = ap.parse_args()
    Path("cache").mkdir(exist_ok=True)

    hosp = {}
    with open_t(find("patient")) as fh:
        for r in csv.DictReader(fh):
            hosp[r["patientunitstayid"]] = r["hospitalid"]
    print(f"stays: {len(hosp):,}")

    # ---- single pass ----------------------------------------------------
    obs = defaultdict(dict)            # (stay, offset) -> quad -> value
    vocab = defaultdict(lambda: defaultdict(Counter))   # quad -> hosp -> Counter
    stays_at = defaultdict(set)
    blank = Counter()
    n = 0
    with open_t(find("nurseAssessment")) as fh:
        for r in csv.DictReader(fh):
            n += 1
            if n % 10_000_000 == 0:
                print(f"  {n:,} rows ...", file=sys.stderr)
            path = r.get("cellattributepath") or ""
            if "Breath Sounds|" not in path:
                continue
            quad = path.rsplit("|", 1)[-1].strip()
            if quad not in QUADS:
                continue
            sid = r["patientunitstayid"]
            h = hosp.get(sid)
            if h is None:
                continue
            v = (r.get("cellattributevalue") or "").strip()
            if not v:
                blank[quad] += 1
                continue
            obs[(sid, r["nurseassessoffset"])][quad] = v
            vocab[quad][h][v] += 1
            stays_at[h].add(sid)
    print(f"scanned {n:,} rows")
    print(f"blank values by quadrant: {dict(blank)}")

    hosps = [h for h in stays_at if len(stays_at[h]) >= a.min_stays]
    hosps.sort(key=lambda h: -len(stays_at[h]))
    print(f"hospitals with >= {a.min_stays} breath-sounds stays: {len(hosps)}")
    keep = set(hosps)

    # ---- A. co-population ------------------------------------------------
    print("\n=== A. co-population: Right Lower with Left Lower at same offset ===")
    rl_tot = Counter()
    rl_match = Counter()
    quad_n = Counter()
    for (sid, _off), d in obs.items():
        h = hosp.get(sid)
        if h not in keep:
            continue
        for q in d:
            quad_n[q] += 1
        if "Right Lower" in d:
            rl_tot[h] += 1
            if "Left Lower" in d:
                rl_match[h] += 1
    print(f"observations per quadrant: {dict(quad_n)}")
    tot, mat = sum(rl_tot.values()), sum(rl_match.values())
    print(f"pooled coincidence: {mat:,}/{tot:,} = {mat/max(tot,1):.4f}"
          f"   (MIMIC: 0.9964)")
    rates = pd.Series({h: rl_match[h] / rl_tot[h] for h in hosps if rl_tot[h] >= 100})
    print(f"per-hospital (n={len(rates)}): min {rates.min():.3f}  "
          f"p10 {rates.quantile(.10):.3f}  median {rates.median():.3f}  "
          f"max {rates.max():.3f}")
    print(f"hospitals below 0.90: {(rates < 0.90).sum()}")

    # ---- B. bilateral agreement -----------------------------------------
    print("\n=== B. Right Lower vs Left Lower value agreement ===")
    pairs = Counter()
    per_h = defaultdict(Counter)
    for (sid, _off), d in obs.items():
        h = hosp.get(sid)
        if h not in keep or "Right Lower" not in d or "Left Lower" not in d:
            continue
        p = (d["Right Lower"], d["Left Lower"])
        pairs[p] += 1
        per_h[h][p] += 1
    cats = sorted({c for p in pairs for c in p})
    T = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
    for (x, y), c in pairs.items():
        T.loc[x, y] = c
    T.to_csv("cache/eicu_breath_crosstab.csv")
    po, pe, k, N = kappa(T)
    print(f"paired observations {N:,}   categories {len(cats)}")
    print(f"po={po:.4f}  pe={pe:.4f}  kappa={k:.4f}   (MIMIC full: 0.8704)")
    print("\nmarginals:")
    md = pd.DataFrame({"RL": T.sum(axis=1) / N, "LL": T.sum(axis=0) / N})
    print(md.sort_values("RL", ascending=False).head(12).to_string(
        float_format=lambda x: f"{x:.5f}"))
    ks = {}
    for h in hosps:
        if sum(per_h[h].values()) < 500:
            continue
        Th = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
        for (x, y), c in per_h[h].items():
            Th.loc[x, y] = c
        ks[h] = kappa(Th)[2]
    if ks:
        s = pd.Series(ks).dropna()
        print(f"\nper-hospital kappa (n={len(s)}): min {s.min():.3f}  "
              f"p10 {s.quantile(.10):.3f}  median {s.median():.3f}  "
              f"max {s.max():.3f}")

    # ---- C. vocabulary divergence ---------------------------------------
    print("\n=== C. cross-hospital vocabulary divergence (Right Lower) ===")
    V = vocab["Right Lower"]
    sets = {h: {v for v, c in V[h].items() if c >= a.min_n}
            for h in hosps if V[h]}
    sets = {h: s for h, s in sets.items() if s}
    print(f"hospitals with a usable vocabulary: {len(sets)}")
    allv = Counter()
    for s in sets.values():
        for v in s:
            allv[v] += 1
    H = len(sets)
    print(f"distinct values overall     : {len(allv)}")
    print(f"used by ALL {H} hospitals   : {sum(1 for c in allv.values() if c == H)}")
    print(f"used by exactly ONE hospital: {sum(1 for c in allv.values() if c == 1)}")

    J = []
    for x, y in itertools.combinations(sets, 2):
        u = len(sets[x] | sets[y])
        if u:
            J.append(len(sets[x] & sets[y]) / u)
    J = np.array(J, float)
    print(f"pairwise Jaccard: pairs {len(J):,}  mean {J.mean():.3f}  "
          f"p10 {np.percentile(J,10):.3f}  median {np.median(J):.3f}  "
          f"p90 {np.percentile(J,90):.3f}")

    sizes = pd.Series({h: len(s) for h, s in sets.items()})
    print(f"vocabulary size: min {sizes.min()}  median {int(sizes.median())}  "
          f"max {sizes.max()}")

    print("\nnear-duplicate values with DISJOINT hospital sets")
    print("(edit distance <= 2, no hospital uses both -- site-specific spellings)")
    vals = sorted(allv)
    found = 0
    for x, y in itertools.combinations(vals, 2):
        if not edit_le(x.lower(), y.lower(), 2):
            continue
        hx = {h for h, s in sets.items() if x in s}
        hy = {h for h, s in sets.items() if y in s}
        if hx and hy and not (hx & hy):
            print(f"  {x!r} ({len(hx)} hosp)  vs  {y!r} ({len(hy)} hosp)")
            found += 1
    if not found:
        print("  none")

    print("\nvalues used by exactly one hospital (top 20 by count)")
    priv = [(sum(V[h][v] for h in sets), v) for v, c in allv.items() if c == 1]
    for c, v in sorted(priv, reverse=True)[:20]:
        print(f"  {c:>8,}  {v!r}")

    pd.DataFrame([{"hospital": h, "stays": len(stays_at[h]),
                   "n_values": len(sets[h])} for h in sets]).to_csv(
        "cache/eicu_breath_vocab.csv", index=False)
    print("\nwrote cache/eicu_breath_crosstab.csv, cache/eicu_breath_vocab.csv")


if __name__ == "__main__":
    main()
