#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Two checks on the corr_mimic result.

CHECK 1 - is RLL/LLL r=0.996 one form, or two assessments that agree?
  Three competing explanations:
    (a) one form writes both fields   -> timestamps coincide almost always
    (b) LOS artefact (both flags rise -> timestamp coincidence would be LOW
        with stay length)                 while stay-level correlation stays high
    (c) real bilateral agreement       -> timestamps coincide AND the recorded
                                          values agree; distinguishable from (a)
                                          only by looking at the values, which
                                          (a) does not predict
  So: measure timestamp coincidence, and cross-tabulate the actual text values
  at coincident timestamps. Discordant values at identical timestamps rules out
  (c) and confirms (a).

CHECK 2 - does the flag block survive conditioning on length of stay?
  Recompute correlations among presence flags after partialling out log(LOS).
  Presence over a whole stay is mechanically increasing in stay length, so the
  raw flag block may be duration rather than documentation behaviour.

Usage:  ./check_lungsounds.py
"""

import csv
import gzip
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not under {root}")


def main() -> None:
    # ---- item ids -------------------------------------------------------
    want = {}
    flags = []
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] != "chartevents":
                continue
            lab = r["label"] or ""
            if lab in ("RLL Lung Sounds", "LLL Lung Sounds"):
                want[r["itemid"]] = lab
            if lab in ("Turn", "Head of Bed", "Position", "Activity Tolerance",
                       "Safety Measures", "Pain Present", "Pain Assessment Method",
                       "Temperature Site", "O2 Delivery Device(s)", "Heart Rhythm",
                       "RLL Lung Sounds", "LLL Lung Sounds", "Therapeutic Bed",
                       "Education Topic", "Pain Management", "Anti Embolic Device",
                       "Pressure Reducing Device", "Ectopy Type 1", "Assistance"):
                flags.append((r["itemid"], lab))
    flag_of = dict(flags)
    print(f"lung itemids: {want}")
    print(f"flag itemids: {len(flag_of)}")

    # ---- stay length ----------------------------------------------------
    los = {}
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        for r in csv.DictReader(fh):
            try:
                los[r["stay_id"]] = float(r["los"])
            except (TypeError, ValueError):
                pass
    print(f"stays: {len(los):,}")

    # ---- single pass ----------------------------------------------------
    ts = defaultdict(lambda: defaultdict(set))     # stay -> label -> {charttime}
    vals = defaultdict(dict)                       # (stay,charttime) -> label -> value
    present = defaultdict(set)                     # label -> {stay}
    n = 0
    with open_t(find(MIMIC / "icu", "chartevents")) as fh:
        for r in csv.DictReader(fh):
            n += 1
            if n % 50_000_000 == 0:
                print(f"  {n:,} rows ...")
            iid = r["itemid"]
            lab = flag_of.get(iid)
            if lab is None:
                continue
            sid = r["stay_id"]
            present[lab].add(sid)
            if iid in want:
                t = r["charttime"]
                ts[sid][want[iid]].add(t)
                vals[(sid, t)][want[iid]] = (r.get("value") or "").strip()
    print(f"scanned {n:,} rows")

    # ---- check 1 --------------------------------------------------------
    print("\n=== CHECK 1: timestamp coincidence ===")
    r_only = l_only = both = 0
    rt_matched = rt_total = 0
    for sid, d in ts.items():
        R, L = d.get("RLL Lung Sounds", set()), d.get("LLL Lung Sounds", set())
        if R and L:
            both += 1
        elif R:
            r_only += 1
        elif L:
            l_only += 1
        rt_total += len(R)
        rt_matched += len(R & L)
    print(f"stays with both charted : {both:,}")
    print(f"stays with RLL only     : {r_only:,}")
    print(f"stays with LLL only     : {l_only:,}")
    if rt_total:
        print(f"RLL events with an LLL event at the SAME charttime: "
              f"{rt_matched:,}/{rt_total:,} = {rt_matched/rt_total:.4f}")

    print("\n=== value agreement at coincident timestamps ===")
    pairs = [(v["RLL Lung Sounds"], v["LLL Lung Sounds"])
             for v in vals.values() if len(v) == 2]
    if pairs:
        df = pd.DataFrame(pairs, columns=["RLL", "LLL"])
        agree = (df.RLL == df.LLL).mean()
        print(f"paired observations: {len(df):,}   identical value: {agree:.4f}")
        print(pd.crosstab(df.RLL, df.LLL).iloc[:8, :8].to_string())
    else:
        print("no coincident pairs found")

    # ---- check 2 --------------------------------------------------------
    print("\n=== CHECK 2: flag block after partialling out log(LOS) ===")
    labs = sorted(present)
    stays = sorted(los)
    idx = {s: i for i, s in enumerate(stays)}
    F = np.zeros((len(stays), len(labs)))
    for j, lab in enumerate(labs):
        for s in present[lab]:
            if s in idx:
                F[idx[s], j] = 1.0
    z = np.log(np.array([los[s] for s in stays]) + 1e-6)
    z = (z - z.mean()) / z.std()
    R = F - np.outer(z, (F * z[:, None]).mean(axis=0))   # residualise on log LOS

    raw = pd.DataFrame(F, columns=labs).corr()
    par = pd.DataFrame(R, columns=labs).corr()
    off = ~np.eye(len(labs), dtype=bool)
    print(f"mean |r| among flags, raw          : {raw.values[off].__abs__().mean():.3f}")
    print(f"mean |r| among flags, LOS-partialled: {par.values[off].__abs__().mean():.3f}")
    if "RLL Lung Sounds" in labs and "LLL Lung Sounds" in labs:
        print(f"RLL x LLL  raw {raw.loc['RLL Lung Sounds','LLL Lung Sounds']:.4f}"
              f"   partialled {par.loc['RLL Lung Sounds','LLL Lung Sounds']:.4f}")
    par.to_csv("cache/flag_corr_los_partialled.csv")
    print("wrote cache/flag_corr_los_partialled.csv")


if __name__ == "__main__":
    main()
