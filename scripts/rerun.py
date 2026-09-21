#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy"]
# ///
"""
Final rerun of the downstream experiment, Tables 2 and 3 and the sensitivity
analysis, through ONE pipeline.

Changes relative to the analyses currently in the manuscript, each answering a
point raised in pre-submission review:

  1. One logistic fitter and one CONSTANT ridge penalty (lambda = 1e-4) for
     every table and both model specifications. Previously the robustness
     analysis scaled the penalty by the number of columns, which confounded
     added flexibility with stronger shrinkage.
  2. Standardisation (and temperature imputation) estimated inside each
     training fold and applied to its held-out fold.
  3. Cross-validation folds and bootstrap resampling grouped by SUBJECT
     (subject_id), not by ICU stay.
  4. AUC computed on pooled out-of-fold predictions for BOTH point estimates
     and bootstrap intervals.
  5. Within each stay-hour, the value retained for each variable is the LAST
     BY CHARTTIME (ties broken by storetime), not the last in file order.
     This requires a fresh chartevents scan, cached for reuse.

Bootstrap: subjects are resampled with replacement and represented as
integer multiplicity weights on their rows. For the pooled statistics used
here (mean log score, pooled AUC with ties counted as one half) this is
exactly equivalent to materialising the duplicated rows; the self-test
verifies that equivalence numerically.

Runs fully offline. A self-test runs first and aborts before the long scan if
the fitter, the weighted AUC or the bootstrap equivalence fails.

Usage:
  ./rerun.py                   # self-test, scan (if not cached), full analysis
  ./rerun.py --selftest        # self-test only (no clinical data)
  ./rerun.py --dry-run         # full analysis on synthetic data (no clinical data)
  ./rerun.py --rescan          # force a fresh chartevents scan
Outputs: cache/rerun_results.json, cache/rerun_report.txt, and CSV tables.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))
LAMBDA = 1e-4
MARGIN = 0.10
ONE_SIDED = 0.05
BASE = ["hr", "rr", "spo2", "map", "temp"]
LUNG = ["RLL Lung Sounds", "LLL Lung Sounds", "RUL Lung Sounds", "LUL Lung Sounds"]
VITALS = {"Heart Rate": "hr", "Respiratory Rate": "rr",
          "O2 saturation pulseoxymetry": "spo2",
          "Arterial Blood Pressure mean": "map",
          "Non Invasive Blood Pressure mean": "nibp_map",
          "Temperature Fahrenheit": "temp"}
ALIAS = {"Ins/Exp Wheeze": "Insp/Exp Wheeze", "Pleural fricton": "Pleural friction"}
WHEEZE = {"Insp Wheeze": "Wheezing", "Exp Wheeze": "Wheezing",
          "Insp/Exp Wheeze": "Wheezing"}
EICU = {"Diminished": "diminished", "Clear": "clear", "Rhonchi": "rhonchi",
        "Crackles": "rales", "Wheezing": "wheezing", "Absent": "absent",
        "Pleural friction": "pleural rub", "Stridor": "stridor",
        "Bronchial": "diminished", "Tubular": "diminished", "Egophony": "rhonchi"}
UNMATCHED = ["Bronchial", "Tubular", "Egophony"]
OUTCOMES = {"vent24": ("Invasive Ventilation", 24),
            "vent48": ("Invasive Ventilation", 48),
            "niv24": ("Non-invasive Ventilation", 24),
            "death": (None, None)}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:6.1f} min] {msg}",
          flush=True)


# ============================================================ utilities
def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    sys.exit(f"{name} not found under {root} (set MIMIC_IV_DIR)")


def epoch(ts: str):
    """'YYYY-MM-DD HH:MM:SS' -> seconds; None if unparseable."""
    try:
        y = int(ts[0:4]); mo = int(ts[5:7]); d = int(ts[8:10])
        h = int(ts[11:13]); mi = int(ts[14:16])
        s = int(ts[17:19]) if len(ts) >= 19 else 0
        yy = y - (1 if mo <= 2 else 0)
        era = yy // 400
        yoe = yy - era * 400
        doy = (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        return (era * 146097 + doe - 719468) * 86400 + h * 3600 + mi * 60 + s
    except (ValueError, IndexError, TypeError):
        return None


class FitError(RuntimeError):
    pass


# ============================================================ model
def logistic(X, y, lam=LAMBDA, iters=100, tol=1e-10):
    """Ridge-penalised logistic regression, Newton with backtracking line
    search and a relative convergence criterion. One constant penalty."""
    n, k = X.shape
    b = np.zeros(k)

    def nll(beta):
        eta = np.clip(X @ beta, -30, 30)
        p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        return (-float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
                + 0.5 * lam * float(beta @ beta))

    f = nll(b)
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - lam * b
        H = (X * W[:, None]).T @ X + lam * np.eye(k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            raise FitError("singular Hessian")
        if not np.all(np.isfinite(step)):
            raise FitError("non-finite step")
        tol_f = 1e-12 * max(abs(f), 1.0)
        t, ok = 1.0, False
        for _ in range(40):
            fn = nll(b + t * step)
            if np.isfinite(fn) and fn <= f + tol_f:
                ok = True
                break
            t *= 0.5
        if not ok:
            if np.max(np.abs(step)) < 1e-7:
                return b
            raise FitError("line search failed")
        b = b + t * step
        rel = abs(f - fn) / max(abs(f), 1.0)
        f = fn
        if rel < tol or float(np.max(np.abs(g))) < 1e-7 * max(n, 1):
            return b
    raise FitError("no convergence")


def predict(X, b):
    return np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-12, 1 - 1e-12)


def design_fold(Ztr, Zte, spec):
    """Fold-local preprocessing: temperature imputed with the TRAINING median;
    standardisation with TRAINING moments; flexible expansion then re-scaled
    with TRAINING moments. Nothing is estimated from the held-out fold."""
    Ztr = Ztr.copy(); Zte = Zte.copy()
    ti = BASE.index("temp")
    med = np.nanmedian(Ztr[:, ti])
    Ztr[np.isnan(Ztr[:, ti]), ti] = med
    Zte[np.isnan(Zte[:, ti]), ti] = med
    mu, sd = Ztr.mean(0), Ztr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A, B = (Ztr - mu) / sd, (Zte - mu) / sd
    if spec == "flex":
        k = A.shape[1]
        def expand(M):
            prods = np.column_stack([M[:, i] * M[:, j]
                                     for i in range(k) for j in range(i + 1, k)])
            return np.column_stack([M, M ** 2, prods])
        A, B = expand(A), expand(B)
        mu2, sd2 = A.mean(0), A.std(0)
        sd2 = np.where(sd2 > 0, sd2, 1.0)
        A, B = (A - mu2) / sd2, (B - mu2) / sd2
    return (np.column_stack([np.ones(len(A)), A]),
            np.column_stack([np.ones(len(B)), B]))


def onehot(vtr, vte):
    cats = sorted(set(vtr))
    if len(cats) < 2:
        return np.zeros((len(vtr), 0)), np.zeros((len(vte), 0))
    return (np.column_stack([(vtr == c).astype(float) for c in cats[1:]]),
            np.column_stack([(vte == c).astype(float) for c in cats[1:]]))


def encode(vals, level):
    v = np.array([ALIAS.get(x, x) for x in vals]) if level >= 1 else np.asarray(vals)
    if level >= 2:
        v = np.array([WHEEZE.get(x, x) for x in v])
    if level >= 3:
        v = np.array([EICU.get(x, "diminished") for x in v])
    return v


# ============================================================ weighted AUC
def tie_groups(p):
    """Group identical scores once; groups are numbered in ascending order."""
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    new = np.r_[True, sp[1:] != sp[:-1]]
    gid = np.empty(len(p), dtype=np.int64)
    gid[order] = np.cumsum(new) - 1
    return gid, int(new.sum())


def wauc(gid, ng, y, w):
    """Pooled AUC with integer weights; ties count one half. Equivalent to
    the unweighted AUC on data with each row duplicated w times."""
    wp = np.bincount(gid, weights=w * y, minlength=ng)
    wn = np.bincount(gid, weights=w * (1 - y), minlength=ng)
    below = np.cumsum(wn) - wn
    den = wp.sum() * wn.sum()
    return float(np.sum(wp * (below + 0.5 * wn)) / den) if den > 0 else np.nan


def ll_rows(y, p):
    return y * np.log(p) + (1 - y) * np.log(1 - p)


# ============================================================ one analysis cell
def run_cell(Z, vals0, y, folds, spec, levels=(0, 1, 2, 3)):
    """Out-of-fold predictions for the baseline and for each encoding."""
    n = len(y)
    pb = np.full(n, np.nan)
    pf = {L: np.full(n, np.nan) for L in levels}
    enc = {L: encode(vals0, L) for L in levels}
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or y[tr].sum() < 20:
            continue
        Xtr, Xte = design_fold(Z[tr], Z[te], spec)
        pb[te] = predict(Xte, logistic(Xtr, y[tr]))
        for L in levels:
            Ftr, Fte = onehot(enc[L][tr], enc[L][te])
            if Ftr.shape[1] == 0:
                pf[L][te] = pb[te]
                continue
            Atr, Ate = np.column_stack([Xtr, Ftr]), np.column_stack([Xte, Fte])
            pf[L][te] = predict(Ate, logistic(Atr, y[tr]))
    if np.isnan(pb).any() or any(np.isnan(v).any() for v in pf.values()):
        raise FitError("folds without predictions")
    return pb, pf, {L: len(set(enc[L])) for L in levels}


def summarise(y, subj, pb, pf, boot, rng, levels=(0, 1, 2, 3)):
    llb = ll_rows(y, pb)
    llf = {L: ll_rows(y, pf[L]) for L in levels}
    gb = tie_groups(pb)
    gf = {L: tie_groups(pf[L]) for L in levels}
    ones = np.ones(len(y))
    auc_b = wauc(*gb, y, ones)
    point = {L: {"gain": float(np.mean(llf[L] - llb)),
                 "dauc": wauc(*gf[L], y, ones) - auc_b} for L in levels}
    codes, uniq = pd.factorize(subj)
    S = len(uniq)
    draws = {L: [] for L in levels if L in (2, 3)}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, S, S), minlength=S)[codes].astype(float)
        if (w * y).sum() < 20:
            continue
        a0 = wauc(*gf[0], y, w)
        for L in draws:
            dls = float(np.sum(w * (llf[L] - llf[0])) / w.sum())
            draws[L].append((dls, wauc(*gf[L], y, w) - a0))
    native = point[0]["gain"]
    contrasts = {}
    for L, arr in draws.items():
        arr = np.array(arr)
        lo, hi = np.percentile(arr[:, 0], [2.5, 97.5])
        alo, ahi = np.percentile(arr[:, 1], [2.5, 97.5])
        m = MARGIN * abs(native)
        contrasts[f"R{L}-R0"] = {
            "point": point[L]["gain"] - native, "lo": float(lo), "hi": float(hi),
            "auc_point": point[L]["dauc"] - point[0]["dauc"],
            "auc_lo": float(alo), "auc_hi": float(ahi),
            "margin": m, "inside_10pct": bool(lo > -m and hi < m),
            "lo_pct_of_native": float(100 * lo / abs(native)) if native else None,
            "hi_pct_of_native": float(100 * hi / abs(native)) if native else None,
            "onesided_5pct_ok": bool(lo > -ONE_SIDED * abs(native)),
            "replicates": int(len(arr))}
    return point, contrasts


# ============================================================ self-test
def selftest():
    log("SELF-TEST (no clinical data)")
    rng = np.random.default_rng(5)
    n = 490000
    Zr = rng.normal(size=(n, 5)) * np.array([20, 5, 3, 15, 1.5]) + \
        np.array([85, 18, 96, 75, 98.6])
    Zr[rng.random(n) < 0.002, 4] = 0.0
    Zr[rng.random(n) < 0.001, 0] = 300.0
    Zr[rng.random(n) < 0.001, 2] = 0.0
    Zr[rng.random(n) < 0.05, 4] = np.nan
    lin = -3.3 + 0.5 * (Zr[:, 1] - 18) / 5 - 0.3 * np.clip(Zr[:, 2] - 96, -20, 20) / 3
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)

    # (a) fitter vs independent L-BFGS, constant lambda, both specifications
    for spec in ("linear", "flex"):
        X, _ = design_fold(Zr, Zr[:10], spec)
        b = logistic(X, y)
        def f(beta):
            eta = np.clip(X @ beta, -30, 30)
            p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + 0.5 * LAMBDA * beta @ beta
        def gr(beta):
            eta = np.clip(X @ beta, -30, 30)
            return -(X.T @ (y - 1 / (1 + np.exp(-eta)))) + LAMBDA * beta
        ref = optimize.minimize(f, np.zeros(X.shape[1]), jac=gr, method="L-BFGS-B",
                                options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10})
        dev = float(np.max(np.abs(b - ref.x)))
        if not dev < 1e-4:
            raise FitError(f"{spec}: fitter disagrees with L-BFGS by {dev:.2e}")
        log(f"  fitter {spec:<6} cols {X.shape[1]:>2}  vs L-BFGS {dev:.1e}  OK")

    # (b) weighted AUC equals rank AUC, and equals row duplication
    p = np.round(rng.random(20000), 3)                 # deliberate ties
    yy = (rng.random(20000) < p).astype(float)
    r = stats.rankdata(p)
    ref_auc = (r[yy == 1].sum() - yy.sum() * (yy.sum() + 1) / 2) / (yy.sum() * (len(yy) - yy.sum()))
    got = wauc(*tie_groups(p), yy, np.ones(len(yy)))
    if abs(got - ref_auc) > 1e-12:
        raise FitError(f"weighted AUC {got} != rank AUC {ref_auc}")
    w = rng.integers(0, 4, len(p)).astype(float)
    dup_p, dup_y = np.repeat(p, w.astype(int)), np.repeat(yy, w.astype(int))
    r2 = stats.rankdata(dup_p)
    dup_auc = (r2[dup_y == 1].sum() - dup_y.sum() * (dup_y.sum() + 1) / 2) / (dup_y.sum() * (len(dup_y) - dup_y.sum()))
    got_w = wauc(*tie_groups(p), yy, w)
    if abs(got_w - dup_auc) > 1e-12:
        raise FitError(f"weighted AUC {got_w} != duplicated-row AUC {dup_auc}")
    log(f"  weighted AUC = rank AUC and = duplicated-row AUC (diff < 1e-12)  OK")

    # (c) uninformative feature: held-out gain near zero / negative
    folds = rng.integers(0, 5, n)
    x = rng.integers(0, 13, n).astype(str)
    pb, pf, _ = run_cell(Zr, x, y, folds, "linear", levels=(0,))
    g = float(np.mean(ll_rows(y, pf[0]) - ll_rows(y, pb)))
    if not g < 0.0005:
        raise FitError(f"noise feature held-out gain {g}")
    log(f"  noise feature held-out gain {g:+.6f}  OK")
    log("SELF-TEST PASSED")


# ============================================================ scan
def scan(cache: Path) -> pd.DataFrame:
    intime, subj_of = {}, {}
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        for r in csv.DictReader(fh):
            intime[r["stay_id"]] = epoch(r["intime"])
            subj_of[r["stay_id"]] = r["subject_id"]
    lung_id, vit_id = {}, {}
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] != "chartevents":
                continue
            if r["label"] in LUNG:
                lung_id[r["itemid"]] = r["label"]
            if r["label"] in VITALS:
                vit_id[r["itemid"]] = VITALS[r["label"]]
    log(f"lung-sound item ids: {dict(sorted(lung_id.items()))}")
    if len(set(lung_id.values())) != 4:
        sys.exit(f"expected four lung-sound fields, found {sorted(set(lung_id.values()))}")

    src = find(MIMIC / "icu", "chartevents")

    def rows(wanted):
        with open_t(src) as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            ix = {c: hdr.index(c) for c in ("stay_id", "charttime", "storetime",
                                             "itemid", "value", "valuenum")
                  if c in hdr}
            if "storetime" not in ix:
                log("  no storetime column; ties broken by file order")
            n = 0
            for r in rd:
                n += 1
                if n % 50_000_000 == 0:
                    log(f"    {n:,} rows")
                if r[ix["itemid"]] in wanted:
                    yield r, ix
            log(f"    scanned {n:,} rows")

    def slot(r, ix):
        sid = r[ix["stay_id"]]
        b = intime.get(sid)
        t = epoch(r[ix["charttime"]])
        if b is None or t is None:
            return None
        h = (t - b) // 3600
        if h < 0 or h >= 72:
            return None
        st = epoch(r[ix["storetime"]]) if "storetime" in ix else None
        return sid, int(h), (t, st if st is not None else -1)

    log("PASS 1/2: lung-sound fields, last value by charttime within each hour")
    lung = defaultdict(dict)
    for r, ix in rows(lung_id):
        s = slot(r, ix)
        if s is None:
            continue
        sid, h, key = s
        v = (r[ix["value"]] or "").strip()
        if not v:
            continue
        fld = lung_id[r[ix["itemid"]]]
        cur = lung[(sid, h)].get(fld)
        if cur is None or key >= cur[0]:
            lung[(sid, h)][fld] = (key, v)
    keys = set(lung)
    log(f"  stay-hours with any lung-sound value: {len(keys):,}")

    log("PASS 2/2: vital signs for those stay-hours, last value by charttime")
    vit = defaultdict(dict)
    for r, ix in rows(vit_id):
        s = slot(r, ix)
        if s is None or (s[0], s[1]) not in keys:
            continue
        sid, h, key = s
        try:
            v = float(r[ix["valuenum"]])
        except (TypeError, ValueError):
            continue
        var = vit_id[r[ix["itemid"]]]
        cur = vit[(sid, h)].get(var)
        if cur is None or key >= cur[0]:
            vit[(sid, h)][var] = (key, v)

    recs = []
    for (sid, h), d in lung.items():
        rec = {"stay_id": sid, "subject_id": subj_of.get(sid), "hour": h}
        for fld, (_, v) in d.items():
            rec[fld] = v
        for var, (_, v) in vit.get((sid, h), {}).items():
            rec[var] = v
        recs.append(rec)
    df = pd.DataFrame(recs)
    for c in LUNG + list(VITALS.values()):
        if c not in df.columns:
            df[c] = np.nan
    df = df[["stay_id", "subject_id", "hour"] + LUNG + list(VITALS.values())]
    df.to_csv(cache, index=False)
    log(f"wrote {cache} ({len(df):,} stay-hours)")
    return df


def load_extract(cache: Path) -> pd.DataFrame:
    df = pd.read_csv(cache, dtype={"stay_id": str, "subject_id": str},
                     keep_default_na=False, na_values=[""])
    for c in list(VITALS.values()) + ["hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def event_hours(name):
    label, horizon = OUTCOMES[name]
    intime = {}
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        stays = list(csv.DictReader(fh))
    for r in stays:
        intime[r["stay_id"]] = epoch(r["intime"])
    if name == "death":
        death = {}
        with open_t(find(MIMIC / "hosp", "admissions")) as fh:
            for r in csv.DictReader(fh):
                dt = (r.get("deathtime") or "").strip()
                if dt:
                    death[r["hadm_id"]] = epoch(dt)
        eh = {}
        for r in stays:
            t = death.get(r["hadm_id"])
            b, o = epoch(r["intime"]), epoch(r["outtime"])
            if t is None or b is None or o is None or t > o:
                continue
            eh[r["stay_id"]] = int((t - b) // 3600)
        return eh, None
    ids = set()
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] == "procedureevents" and \
                    (r["label"] or "").strip().lower() == label.lower():
                ids.add(r["itemid"])
    if not ids:
        sys.exit(f"no procedureevents item labelled exactly {label!r}")
    eh = {}
    with open_t(find(MIMIC / "icu", "procedureevents")) as fh:
        for r in csv.DictReader(fh):
            if r["itemid"] not in ids:
                continue
            b, t = intime.get(r["stay_id"]), epoch(r["starttime"])
            if b is None or t is None:
                continue
            h = int((t - b) // 3600)
            if r["stay_id"] not in eh or h < eh[r["stay_id"]]:
                eh[r["stay_id"]] = h
    log(f"  {name}: item id(s) {sorted(ids)}, {len(eh):,} stays with the event")
    return eh, horizon


# ============================================================ analysis
def analyse(ext, events, boot, perms, seed=0):
    ext = ext.copy()
    ext["map"] = ext["map"].fillna(ext["nibp_map"])
    ext = ext.dropna(subset=["hr", "rr", "spo2", "map"]).reset_index(drop=True)
    subjects = ext["subject_id"].unique()
    fold_of = dict(zip(subjects, np.random.default_rng(seed).integers(0, 5, len(subjects))))
    ext["fold"] = ext["subject_id"].map(fold_of)
    results = {"settings": {"lambda": LAMBDA, "margin": MARGIN, "boot": boot,
                            "perms": perms, "folds": "subject_id",
                            "scaling": "fold-local", "auc": "pooled out-of-fold",
                            "within_hour": "last by charttime, storetime tiebreak"},
               "table2": {}, "sensitivity": {}, "table3": {}, "noise_floor": {}}

    def rows_for(outcome, field, drop_unmatched=False):
        eh, horizon = events[outcome]
        e = ext["stay_id"].map(eh)
        risk = ~(e.notna() & (e <= ext["hour"]))
        # to_numpy(copy=True): pandas can return a read-only view here
        m = (risk & ext[field].notna()).to_numpy(copy=True)
        if drop_unmatched:
            m = m & ~ext[field].isin(UNMATCHED).to_numpy(copy=True)
        sub = ext[m]
        e = e[m]
        if horizon is None:
            y = (e.notna() & (e > sub["hour"])).astype(float).values
        else:
            y = (e.notna() & (e > sub["hour"]) & (e <= sub["hour"] + horizon)).astype(float).values
        return (sub[BASE].values.astype(float), sub[field].astype(str).values, y,
                sub["fold"].values, sub["subject_id"].values, sub["stay_id"].nunique())

    def cell(outcome, field, spec, drop=False, tag=""):
        Z, v, y, f, s, nst = rows_for(outcome, field, drop)
        rng = np.random.default_rng(1)
        pb, pf, levels = run_cell(Z, v, y, f, spec)
        point, con = summarise(y, s, pb, pf, boot, rng)
        out = {"n": int(len(y)), "stays": int(nst), "subjects": int(len(set(s))),
               "events": int(y.sum()), "rate": float(y.mean()),
               "levels": levels, "point": point, "contrasts": con}
        c3 = con["R3-R0"]
        log(f"  {tag:<34} n {len(y):>7,}  gain {point[0]['gain']:.6f}  "
            f"R3-R0 {c3['point']:+.6f} [{c3['lo']:+.6f}, {c3['hi']:+.6f}]  "
            f"{'inside' if c3['inside_10pct'] else 'OUTSIDE'}")
        return out

    log("TABLE 2: four fields, invasive ventilation 24 h, linear")
    for fld in LUNG:
        results["table2"][fld] = cell("vent24", fld, "linear", tag=fld)
    log("SENSITIVITY: unmatched values dropped")
    for fld in LUNG:
        results["sensitivity"][fld] = cell("vent24", fld, "linear", drop=True, tag=fld)
    log("TABLE 3: right lower lobe, four outcomes x two specifications")
    for oc in OUTCOMES:
        for spec in ("linear", "flex"):
            results["table3"][f"{oc}|{spec}"] = cell(oc, "RLL Lung Sounds", spec,
                                                     tag=f"{oc} {spec}")
    log(f"NOISE FLOOR: {perms} subject-level outcome permutations, RLL, vent24")
    Z, v, y, f, s, _ = rows_for("vent24", "RLL Lung Sounds")
    codes, uniq = pd.factorize(s)
    sy = np.zeros(len(uniq))
    np.maximum.at(sy, codes, y)
    gains, daucs = [], []
    prng = np.random.default_rng(100)
    for i in range(perms):
        yp = sy[prng.permutation(len(uniq))][codes]
        if yp.sum() < 50:
            continue
        pb, pf, _ = run_cell(Z, v, yp, f, "linear", levels=(0,))
        ones = np.ones(len(yp))
        gains.append(float(np.mean(ll_rows(yp, pf[0]) - ll_rows(yp, pb))))
        daucs.append(wauc(*tie_groups(pf[0]), yp, ones) - wauc(*tie_groups(pb), yp, ones))
    results["noise_floor"] = {"n_perms": len(gains),
                              "gain_mean": float(np.mean(gains)), "gain_sd": float(np.std(gains)),
                              "dauc_mean": float(np.mean(daucs)), "dauc_sd": float(np.std(daucs))}
    log(f"  gain {np.mean(gains):+.6f} +/- {np.std(gains):.6f};  "
        f"dAUC {np.mean(daucs):+.5f} +/- {np.std(daucs):.5f}")
    return results


def report(res, path):
    L = []
    def w(s=""): L.append(s)
    st = res["settings"]
    w(f"Settings: lambda {st['lambda']}, margin {st['margin']:.0%}, bootstrap {st['boot']}, "
      f"permutations {st['perms']}, folds by {st['folds']}, {st['scaling']} scaling, "
      f"{st['auc']} AUC, {st['within_hour']}")
    w()
    w("TABLE 2  (invasive ventilation 24 h, linear)")
    for fld, r in res["table2"].items():
        w(f"  {fld}: n {r['n']:,}  stays {r['stays']:,}  subjects {r['subjects']:,}  "
          f"events {r['events']:,} ({100*r['rate']:.2f}%)")
        for Lv in ("0", "1", "2", "3"):
            p = r["point"][int(Lv)] if int(Lv) in r["point"] else r["point"][Lv]
            w(f"    R{Lv} levels {r['levels'][int(Lv)] if int(Lv) in r['levels'] else r['levels'][Lv]:>2}  "
              f"gain {p['gain']:.6f}  dAUC {p['dauc']:.5f}")
        for c, v in r["contrasts"].items():
            w(f"    {c}: {v['point']:+.6f} [{v['lo']:+.6f}, {v['hi']:+.6f}]  margin +/-{v['margin']:.6f}  "
              f"{'inside' if v['inside_10pct'] else 'OUTSIDE'}  "
              f"AUC {v['auc_point']:+.5f} [{v['auc_lo']:+.5f}, {v['auc_hi']:+.5f}]")
    w()
    w("SENSITIVITY  (unmatched dropped)")
    for fld, r in res["sensitivity"].items():
        c3 = r["contrasts"]["R3-R0"]; c2 = r["contrasts"]["R2-R0"]
        w(f"  {fld}: R2-R0 {c2['point']:+.6f} [{c2['lo']:+.6f}, {c2['hi']:+.6f}]  "
          f"R3-R0 {c3['point']:+.6f} [{c3['lo']:+.6f}, {c3['hi']:+.6f}]  "
          f"levels {r['levels']}")
    w()
    w("TABLE 3  (right lower lobe)")
    allc = []
    for key, r in res["table3"].items():
        oc, spec = key.split("|")
        p0 = r["point"][0] if 0 in r["point"] else r["point"]["0"]
        w(f"  {oc:<7}{spec:<7} n {r['n']:,}  events {r['events']:,}  native {p0['gain']:.6f}  dAUC {p0['dauc']:.5f}")
        for c, v in r["contrasts"].items():
            allc.append(v)
            w(f"    {c}: {v['point']:+.6f} [{v['lo']:+.6f}, {v['hi']:+.6f}]  margin +/-{v['margin']:.6f}  "
              f"{'inside' if v['inside_10pct'] else 'OUTSIDE'}  "
              f"bounds {v['lo_pct_of_native']:+.1f}% / {v['hi_pct_of_native']:+.1f}% of native")
    w()
    worst = min(v["lo_pct_of_native"] for v in allc)
    beyond = [v for v in allc if abs(v["hi_pct_of_native"]) > 5 or abs(v["lo_pct_of_native"]) > 5]
    w(f"contrasts inside 10% margin: {sum(v['inside_10pct'] for v in allc)}/{len(allc)}")
    w(f"one-sided 5% non-degradation met: {sum(v['onesided_5pct_ok'] for v in allc)}/{len(allc)}")
    w(f"most extreme degradation-side bound: {worst:+.1f}% of native gain")
    w(f"bounds beyond 5% of native (either side): {len(beyond)}")
    w()
    nf = res["noise_floor"]
    w(f"NOISE FLOOR ({nf['n_perms']} permutations): gain {nf['gain_mean']:+.6f} +/- {nf['gain_sd']:.6f}; "
      f"dAUC {nf['dauc_mean']:+.5f} +/- {nf['dauc_sd']:.5f}")
    txt = "\n".join(L)
    Path(path).write_text(txt)
    print("\n" + txt, flush=True)


def synthetic(n_subj=4000, seed=7):
    """Synthetic extract with the real structure, for a full dry run."""
    rng = np.random.default_rng(seed)
    cats = ["Clear", "Diminished", "Rhonchi", "Crackles", "Insp Wheeze", "Exp Wheeze",
            "Insp/Exp Wheeze", "Absent", "Bronchial", "Tubular", "Stridor", "Egophony",
            "Pleural friction"]
    probs = np.array([20, 60, 12, 4, .3, 1, .6, .2, .1, .04, .02, .01, .02]); probs /= probs.sum()
    recs, events = [], {k: {} for k in OUTCOMES}
    sid = 0
    for subj in range(n_subj):
        for _ in range(rng.integers(1, 3)):
            sid += 1
            s = str(30000000 + sid)
            risk = rng.normal()
            for oc in OUTCOMES:
                if rng.random() < 0.08 + 0.05 * (risk > 1):
                    events[oc][s] = int(rng.integers(2, 90))
            for h in range(int(rng.integers(10, 72))):
                rec = {"stay_id": s, "subject_id": str(10000000 + subj), "hour": h,
                       "hr": rng.normal(85, 20), "rr": rng.normal(18 + 2 * risk, 5),
                       "spo2": rng.normal(96, 3), "map": rng.normal(75, 15),
                       "nibp_map": np.nan,
                       "temp": np.nan if rng.random() < .05 else rng.normal(98.6, 1.5)}
                for fld in LUNG:
                    if rng.random() < 0.8:
                        v = rng.choice(cats, p=probs)
                        if fld.startswith("LLL") and v == "Insp/Exp Wheeze":
                            v = "Ins/Exp Wheeze"
                        if fld.startswith("LLL") and v == "Pleural friction":
                            v = "Pleural fricton"
                        rec[fld] = v
                recs.append(rec)
    df = pd.DataFrame(recs)
    for c in LUNG:
        if c not in df.columns:
            df[c] = np.nan
    ev = {k: (v, OUTCOMES[k][1]) for k, v in events.items()}
    return df, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=20)
    a = ap.parse_args()

    selftest()
    if a.selftest:
        return
    out = Path("cache"); out.mkdir(exist_ok=True)
    if a.dry_run:
        log("DRY RUN on synthetic data")
        ext, events = synthetic()
        res = analyse(ext, events, boot=min(a.boot, 50), perms=min(a.perms, 3))
        report(res, out / "rerun_dryrun_report.txt")
        (out / "rerun_dryrun_results.json").write_text(json.dumps(res, indent=1, default=str))
        log("DRY RUN COMPLETE")
        return

    cache = out / "rerun_extract.csv.gz"
    if cache.exists() and not a.rescan:
        log(f"using cached extract {cache}")
        ext = load_extract(cache)
    else:
        ext = scan(cache)
        ext = load_extract(cache)
    log(f"extract: {len(ext):,} stay-hours, {ext['stay_id'].nunique():,} stays, "
        f"{ext['subject_id'].nunique():,} subjects")
    log("outcomes")
    events = {k: event_hours(k) for k in OUTCOMES}
    res = analyse(ext, events, a.boot, a.perms)
    (out / "rerun_results.json").write_text(json.dumps(res, indent=1, default=str))
    report(res, out / "rerun_report.txt")
    log("ALL DONE -- results in cache/rerun_report.txt and cache/rerun_results.json")


if __name__ == "__main__":
    main()
