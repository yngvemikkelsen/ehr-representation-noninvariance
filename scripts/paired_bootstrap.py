#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy"]
# ///
"""
Paired uncertainty for the encoding contrasts, and the rare-category
sensitivity analysis.

WHY THIS REPLACES THE PERMUTATION FOR THE CENTRAL CLAIM
  The stay-level outcome permutation estimates how the feature's incremental
  gain behaves under a null outcome. That is a test of whether the feature
  carries signal. It is NOT the sampling distribution of the contrast between
  two encodings, which is what the null result requires:

      Delta_{R2-R0}   and   Delta_{R3-R0}

  Those contrasts are paired by construction: the encodings are applied to
  the same rows, share almost all their variance, and differ only where the
  partition changes. A paired bootstrap exploits that pairing; the
  permutation standard deviation ignores it and is the wrong denominator.

DESIGN
  Out-of-fold predictions are computed ONCE per encoding, under a FIXED fold
  assignment shared across all arms. Bootstrap replicates then resample
  STAYS (not rows) and recompute both metrics on the stored predictions.
  Re-splitting per replicate would inject fold noise into a contrast that is
  nearly deterministic given the folds.

  Reported per contrast: point estimate and percentile 95% CI for
    - incremental log score (nats per observation)
    - AUC
  Both computed as paired differences within each replicate.

EQUIVALENCE
  A confidence interval around zero supports "no detectable decrement". It
  does not by itself support "equivalence", which needs a pre-specified
  negligibility margin. --margin sets one; the script reports whether the CI
  falls entirely inside it, so the stronger word can be used only when the
  claim is actually met.

SENSITIVITY
  --drop-unmatched repeats everything with the three MIMIC values that have
  no eICU equivalent (Bronchial, Tubular, Egophony) removed, rather than
  assigned to a nearest category.

Reads cache/cmi_extract.csv.gz. No rescan.

Usage:
  ./paired_bootstrap.py --boot 2000
  ./paired_bootstrap.py --boot 2000 --drop-unmatched
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

ALIAS = {"Ins/Exp Wheeze": "Insp/Exp Wheeze",
         "Pleural fricton": "Pleural friction"}
WHEEZE = {"Insp Wheeze": "Wheezing", "Exp Wheeze": "Wheezing",
          "Insp/Exp Wheeze": "Wheezing"}
EICU = {"Diminished": "diminished", "Clear": "clear", "Rhonchi": "rhonchi",
        "Crackles": "rales", "Wheezing": "wheezing", "Absent": "absent",
        "Pleural friction": "pleural rub", "Stridor": "stridor",
        "Bronchial": "diminished", "Tubular": "diminished",
        "Egophony": "rhonchi"}
UNMATCHED = ["Bronchial", "Tubular", "Egophony"]


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


def predict(X, b):
    return np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-12, 1 - 1e-12)


def auc_of(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def encode(vals, level):
    v = np.array([ALIAS.get(x, x) for x in vals]) if level >= 1 else np.asarray(vals)
    if level >= 2:
        v = np.array([WHEEZE.get(x, x) for x in v])
    if level >= 3:
        v = np.array([EICU.get(x, "diminished") for x in v])
    return v


def oof_predictions(vals, Xb, y, folds, n_folds):
    """Out-of-fold probabilities for baseline and baseline+feature, under a
    fixed fold assignment shared across encodings."""
    p_base = np.full(len(y), np.nan)
    p_full = np.full(len(y), np.nan)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or y[tr].sum() < 20:
            continue
        cats = sorted(set(vals[tr]))
        F = (np.zeros((len(vals), 0)) if len(cats) < 2 else
             np.column_stack([(vals == c).astype(float) for c in cats[1:]]))
        b0 = logistic(Xb[tr], y[tr])
        p_base[te] = predict(Xb[te], b0)
        if F.shape[1] == 0:
            p_full[te] = p_base[te]
        else:
            Xf = np.column_stack([Xb, F])
            b1 = logistic(Xf[tr], y[tr])
            p_full[te] = predict(Xf[te], b1)
    return p_base, p_full


def metrics(y, p_base, p_full):
    ll_b = np.sum(y * np.log(p_base) + (1 - y) * np.log(1 - p_base))
    ll_f = np.sum(y * np.log(p_full) + (1 - y) * np.log(1 - p_full))
    return (ll_f - ll_b) / len(y), auc_of(y, p_full) - auc_of(y, p_base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--margin", type=float, default=0.10,
                    help="negligibility margin for the log-score contrast, as "
                         "a fraction of the native encoding's own gain")
    ap.add_argument("--drop-unmatched", action="store_true")
    ap.add_argument("--field", default="RLL Lung Sounds")
    ap.add_argument("--all-fields", action="store_true")
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
    fold_all = ext["fold"].values

    fields = LUNG if a.all_fields else [a.field]
    names = {0: "R0 native", 1: "R1 identity", 2: "R2 wheeze collapsed",
             3: "R3 eICU support"}
    out = []

    for lab in fields:
        if lab not in ext.columns:
            print(f"{lab}: not in cache", file=sys.stderr)
            continue
        m = ext[lab].notna().values
        vals0 = ext.loc[m, lab].astype(str).values
        if a.drop_unmatched:
            keep = ~np.isin(vals0, UNMATCHED)
            idx = np.flatnonzero(m)[keep]
            vals0 = vals0[keep]
        else:
            idx = np.flatnonzero(m)
        Xb, y, fd = Xb_all[idx], y_all[idx], fold_all[idx]
        stay_ids = ext["stay_id"].values[idx]

        print(f"\n{'='*74}\n{lab}"
              f"{'  [unmatched values dropped]' if a.drop_unmatched else ''}")
        print(f"n {len(y):,}   stays {len(set(stay_ids)):,}   "
              f"events {int(y.sum()):,} ({y.mean():.4f})")

        # one set of out-of-fold predictions per encoding, fixed folds
        preds = {}
        for lvl in (0, 1, 2, 3):
            v = encode(vals0, lvl)
            preds[lvl] = oof_predictions(v, Xb, y, fd, a.folds)
            ls, da = metrics(y, *preds[lvl])
            print(f"  {names[lvl]:<22} k={len(set(v)):>2}  "
                  f"log-score {ls:+.6f}  dAUC {da:+.5f}")

        base_ls, _ = metrics(y, *preds[0])

        # paired bootstrap over stays, reusing the stored predictions
        uniq = np.unique(stay_ids)
        idx_by_stay = {s: np.flatnonzero(stay_ids == s) for s in uniq}
        brng = np.random.default_rng(1)
        draws = {2: [], 3: []}
        for _ in range(a.boot):
            pick = brng.choice(len(uniq), size=len(uniq), replace=True)
            rows = np.concatenate([idx_by_stay[uniq[i]] for i in pick])
            yb = y[rows]
            if yb.sum() < 20 or yb.sum() == len(yb):
                continue
            m0 = metrics(yb, preds[0][0][rows], preds[0][1][rows])
            for lvl in (2, 3):
                mk = metrics(yb, preds[lvl][0][rows], preds[lvl][1][rows])
                draws[lvl].append((mk[0] - m0[0], mk[1] - m0[1]))

        print(f"\n  paired stay-level bootstrap, {len(draws[3]):,} replicates")
        print(f"  {'contrast':<14}{'metric':<12}{'point':>12}{'95% CI':>26}")
        for lvl in (2, 3):
            arr = np.array(draws[lvl])
            pt_ls = metrics(y, *preds[lvl])[0] - base_ls
            pt_au = metrics(y, *preds[lvl])[1] - metrics(y, *preds[0])[1]
            for j, (nm, pt) in enumerate((("log score", pt_ls), ("AUC", pt_au))):
                lo, hi = np.percentile(arr[:, j], [2.5, 97.5])
                print(f"  {'R'+str(lvl)+' - R0':<14}{nm:<12}{pt:>+12.6f}"
                      f"   [{lo:+.6f}, {hi:+.6f}]")
                out.append({"field": lab, "contrast": f"R{lvl}-R0", "metric": nm,
                            "point": pt, "lo": lo, "hi": hi,
                            "dropped_unmatched": a.drop_unmatched})
            # equivalence against a pre-specified margin on the log-score scale
            marg = a.margin * abs(base_ls)
            lo, hi = np.percentile(arr[:, 0], [2.5, 97.5])
            inside = (lo > -marg) and (hi < marg)
            print(f"  {'':<14}margin +/-{marg:.6f} ({a.margin:.0%} of native "
                  f"gain): CI {'INSIDE' if inside else 'NOT inside'}")

    if out:
        df = pd.DataFrame(out)
        suffix = "_dropunmatched" if a.drop_unmatched else ""
        p = a.out_dir / f"paired_bootstrap{suffix}.csv"
        df.to_csv(p, index=False)
        print(f"\nwrote {p}")
    print("\nA CI containing zero supports 'no detectable decrement'. Only a CI")
    print("lying entirely inside a pre-specified margin supports 'equivalent'.")


if __name__ == "__main__":
    main()
