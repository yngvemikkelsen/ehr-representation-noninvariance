#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy"]
# ///
"""
Graded collapse of the lung-sound vocabulary, with a lossless negative
control, plus an empirical noise floor for dAUC.

THE DESIGN
  Four representations of the same recorded findings, applied to the same
  rows, evaluated out of sample:

    R0  native MIMIC vocabulary
    R1  spelling aliases merged     LOSSLESS  <- negative control
          Insp/Exp Wheeze == Ins/Exp Wheeze
          Pleural friction == Pleural fricton
    R2  the three wheeze levels collapsed to one, as eICU does   LOSSY
    R3  mapped to the eICU support (8 levels)                    LOSSY

  Information removed at level k:   L_k = MI(R0) - MI(Rk)

  PREDICTION
    L1 ~= 0            renaming destroys no information
    L2, L3 > 0         and ordered, if the collapsed distinctions carry
                       task-relevant information

  R1 is what separates a mechanism from an encoding artefact. A naive
  categorical model breaks when handed an unseen token, which proves
  nothing. Here R1 changes the symbols without changing the partition, so
  any movement in held-out MI indicates a problem with the setup rather
  than a property of the data.

  R2 and R3 change the partition. Only they can remove information.

NOISE FLOOR
  --perms N repeats the whole thing with outcomes permuted across stays,
  giving the sampling distribution of dAUC and MI under no signal. A single
  permutation draw cannot distinguish noise from bias; this reports the
  spread.

Reads cache/cmi_extract.csv.gz. No rescan.

Usage:
  ./collapse_experiment.py --perms 20
"""

import argparse
import csv
import gzip
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))
BASE = ["hr", "rr", "spo2", "map", "temp"]
LUNG = ["RLL Lung Sounds", "LLL Lung Sounds", "RUL Lung Sounds", "LUL Lung Sounds"]

# R1: spelling aliases. Same clinical finding, separately configured picklists.
ALIAS = {"Ins/Exp Wheeze": "Insp/Exp Wheeze",
         "Pleural fricton": "Pleural friction"}

# R2: eICU represents wheeze as a single level.
WHEEZE = {"Insp Wheeze": "Wheezing", "Exp Wheeze": "Wheezing",
          "Insp/Exp Wheeze": "Wheezing"}

# R3: the eICU support -- diminished, clear, rhonchi, rales, wheezing,
# absent, pleural rub, stridor. Everything else maps to its nearest eICU
# equivalent or to the residual level eICU would record.
EICU = {"Diminished": "diminished", "Clear": "clear", "Rhonchi": "rhonchi",
        "Crackles": "rales", "Wheezing": "wheezing", "Absent": "absent",
        "Pleural friction": "pleural rub", "Stridor": "stridor",
        "Bronchial": "diminished", "Tubular": "diminished",
        "Egophony": "rhonchi"}


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    sys.exit(f"{name} not found under {root}")


def epoch(ts: str):
    try:
        y = int(ts[0:4]); mo = int(ts[5:7]); d = int(ts[8:10])
        h = int(ts[11:13]); mi = int(ts[14:16])
        yy = y - (1 if mo <= 2 else 0)
        era = yy // 400
        yoe = yy - era * 400
        doy = (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return (era * 146097 + doe - 719468) * 86400 + h * 3600 + mi * 60
    except (ValueError, IndexError, TypeError):
        return None


def logistic(X, y, iters=40, ridge=1e-4):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - ridge * b
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def ll(X, y, b):
    eta = np.clip(X @ b, -30, 30)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), p


def auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float("nan") if n1 == 0 or n0 == 0 else \
        float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(vals, Xb, ys, fs, folds):
    """Held-out MI (nats/obs) and per-fold-averaged dAUC for one encoding."""
    g = n = 0.0
    aucs = []
    for k in range(folds):
        tr, te = fs != k, fs == k
        if te.sum() < 100 or ys[tr].sum() < 20:
            continue
        cats = sorted(set(vals[tr]))
        if len(cats) < 2:
            continue
        F = np.column_stack([(vals == c).astype(float) for c in cats[1:]])
        b0 = logistic(Xb[tr], ys[tr])
        b1 = logistic(np.column_stack([Xb, F])[tr], ys[tr])
        l0, p0 = ll(Xb[te], ys[te], b0)
        l1, p1 = ll(np.column_stack([Xb, F])[te], ys[te], b1)
        g += l1 - l0
        n += te.sum()
        a0, a1 = auc(ys[te], p0), auc(ys[te], p1)
        if np.isfinite(a0) and np.isfinite(a1):
            aucs.append((a1 - a0, te.sum()))
    if n == 0 or not aucs:
        return np.nan, np.nan
    w = np.array([a[1] for a in aucs], float)
    return g / n, float(np.sum(w / w.sum() * np.array([a[0] for a in aucs])))


def encode(vals, level):
    v = np.array([ALIAS.get(x, x) for x in vals]) if level >= 1 else vals.copy()
    if level >= 2:
        v = np.array([WHEEZE.get(x, x) for x in v])
    if level >= 3:
        v = np.array([EICU.get(x, "diminished") for x in v])
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--cache", type=Path, default=Path("cache/cmi_extract.csv.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("cache"))
    a = ap.parse_args()
    if not a.cache.exists():
        sys.exit(f"{a.cache} not found -- run cmi_screen.py first")

    intime = {}
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        for r in csv.DictReader(fh):
            intime[r["stay_id"]] = r["intime"]
    vent = set()
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] == "procedureevents" and \
                    (r["label"] or "").strip().lower() == "invasive ventilation":
                vent.add(r["itemid"])
    event_hour = {}
    with open_t(find(MIMIC / "icu", "procedureevents")) as fh:
        for r in csv.DictReader(fh):
            if r["itemid"] not in vent:
                continue
            sid = r["stay_id"]
            b = epoch(intime.get(sid, "")) if sid in intime else None
            t = epoch(r["starttime"])
            if b is None or t is None:
                continue
            h = int((t - b) // 3600)
            if sid not in event_hour or h < event_hour[sid]:
                event_hour[sid] = h

    ext = pd.read_csv(a.cache, dtype={"stay_id": str},
                      keep_default_na=False, na_values=[""])
    for c in BASE + ["nibp_map", "hour"]:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")
    ext["map"] = ext["map"].fillna(ext["nibp_map"])
    ext = ext.dropna(subset=["hr", "rr", "spo2", "map"]).copy()
    ext["temp"] = ext["temp"].fillna(ext["temp"].median())
    eh = ext["stay_id"].map(event_hour)
    ext = ext[~(eh.notna() & (eh <= ext["hour"]))].copy()
    eh = ext["stay_id"].map(event_hour)
    ext["y"] = ((eh.notna()) & (eh > ext["hour"]) &
                (eh <= ext["hour"] + a.window)).astype(int)

    stays = ext["stay_id"].unique()
    rng = np.random.default_rng(0)
    fold_of = dict(zip(stays, rng.integers(0, a.folds, len(stays))))
    ext["fold"] = ext["stay_id"].map(fold_of)

    Z = ext[BASE].values.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    Xb_all = np.column_stack([np.ones(len(ext)), Z])
    y_all = ext["y"].values.astype(float)
    folds_all = ext["fold"].values

    names = {0: "R0 native", 1: "R1 aliases merged (LOSSLESS)",
             2: "R2 wheeze collapsed", 3: "R3 eICU support"}
    rows = []
    for lab in LUNG:
        if lab not in ext.columns:
            continue
        m = ext[lab].notna().values
        if m.sum() < 2000:
            continue
        vals0 = ext.loc[m, lab].astype(str).values
        Xb, ys, fs = Xb_all[m], y_all[m], folds_all[m]
        print(f"\n{'='*72}\n{lab}   n {int(m.sum()):,}   "
              f"events {int(ys.sum()):,}")
        unmapped = sorted(set(encode(vals0, 2)) - set(EICU))
        if unmapped:
            print(f"  values with no explicit eICU mapping (-> diminished): "
                  f"{unmapped}")
        mi0 = None
        for lvl in (0, 1, 2, 3):
            v = encode(vals0, lvl)
            mi, da = evaluate(v, Xb, ys, fs, a.folds)
            if lvl == 0:
                mi0 = mi
            loss = mi0 - mi
            rows.append({"field": lab, "level": names[lvl],
                         "levels": len(set(v)), "MI_oos": mi,
                         "L_k": loss, "pct_lost": 100 * loss / mi0 if mi0 else np.nan,
                         "dAUC": da})
            print(f"  {names[lvl]:<30} k={len(set(v)):>2}  "
                  f"MI {mi:+.6f}  L_k {loss:+.6f}  "
                  f"({100*loss/mi0 if mi0 else 0:+.1f}%)  dAUC {da:+.4f}")

    t = pd.DataFrame(rows)
    t.to_csv(a.out_dir / "collapse_experiment.csv", index=False)
    print(f"\nwrote {a.out_dir / 'collapse_experiment.csv'}")

    # ---- noise floor -----------------------------------------------------
    if a.perms > 0:
        print(f"\n{'='*72}\nNOISE FLOOR: {a.perms} outcome permutations across stays")
        stay_y = ext.groupby("stay_id", sort=False)["y"].max()
        lab = LUNG[0]
        m = ext[lab].notna().values
        vals0 = ext.loc[m, lab].astype(str).values
        Xb, fs = Xb_all[m], folds_all[m]
        mis, das = [], []
        for i in range(a.perms):
            pv = pd.Series(np.random.default_rng(100 + i).permutation(stay_y.values),
                           index=stay_y.index)
            yp = ext["stay_id"].map(pv).values.astype(float)[m]
            if yp.sum() < 50:
                continue
            mi, da = evaluate(vals0, Xb, yp, fs, a.folds)
            mis.append(mi); das.append(da)
        mis, das = np.array(mis), np.array(das)
        print(f"  field: {lab}, native encoding, {len(mis)} draws")
        print(f"  MI    mean {mis.mean():+.6f}  sd {mis.std():.6f}  "
              f"range [{mis.min():+.6f}, {mis.max():+.6f}]")
        print(f"  dAUC  mean {das.mean():+.6f}  sd {das.std():.6f}  "
              f"range [{das.min():+.6f}, {das.max():+.6f}]")
        print(f"  2.5th-97.5th percentile of dAUC under no signal: "
              f"[{np.percentile(das,2.5):+.4f}, {np.percentile(das,97.5):+.4f}]")
        obs = t[(t.field == lab) & (t.level == names[0])].dAUC.iloc[0]
        print(f"  observed dAUC for {lab}: {obs:+.4f}  "
              f"= {obs/max(das.std(),1e-9):.1f} sd above the permuted mean")
        print("\n  A mean near zero with spread of this size means the single")
        print("  positive draw seen earlier was noise, not bias. A mean")
        print("  consistently above zero would mean the split still leaks.")


if __name__ == "__main__":
    main()
