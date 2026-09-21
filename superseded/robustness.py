#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy"]
# ///
"""
Robustness of the representation-collapse null across outcomes and model
classes.

WHY
  The manuscript reports the collapse contrast for one outcome (invasive
  ventilation within 24 h) under one model class (logistic, linear in the
  physiologic baseline). Two reviewer objections follow: the null may be
  specific to that target, and it may be an artefact of a baseline too
  inflexible to expose the distinctions the collapse removes.

  This tests both. It does NOT search outcomes for a positive result: every
  outcome here is pre-specified below, all are reported whatever they show,
  and the purpose is to see whether the null is stable, not to find a target
  where it breaks.

OUTCOMES (all pre-specified, all reported)
  vent24   invasive ventilation started within 24 h   [primary, as published]
  vent48   invasive ventilation started within 48 h
  niv24    non-invasive ventilation started within 24 h
  death    in-ICU death at any point after the observation

MODEL CLASSES
  linear   logistic, baseline entered linearly            [as published]
  flex     logistic, baseline entered with quadratic terms and all pairwise
           products. If a richer baseline absorbs what the collapsed
           categories carried, the contrast would move here and not in the
           linear fit.

For each outcome x model class x field, the four encodings R0-R3 are fitted
on identical rows under a fixed stay-level fold assignment, and the paired
stay-level bootstrap gives the R2-R0 and R3-R0 contrasts with 95% intervals.

Reads cache/cmi_extract.csv.gz. No rescan of chartevents.

Usage:
  ./robustness.py --boot 500
  ./robustness.py --boot 500 --outcomes vent24,niv24 --models linear,flex
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

OUTCOMES = {
    "vent24": ("Invasive Ventilation", 24),
    "vent48": ("Invasive Ventilation", 48),
    "niv24": ("Non-invasive Ventilation", 24),
    "death": (None, None),
}


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


class FitError(RuntimeError):
    """Raised when the fitter fails to converge. Never report such a fit."""


def logistic(X, y, iters=100, tol=1e-8):
    """Newton with step halving, ridge scaled to the column count, and an
    explicit convergence requirement. The line search is what prevents the
    divergence seen with the previous fixed-ridge, undamped version on
    columns carrying high-leverage points."""
    n, k = X.shape
    ridge = 1e-3 * k
    b = np.zeros(k)

    def nll(beta):
        eta = np.clip(X @ beta, -30, 30)
        p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))) \
            + 0.5 * ridge * float(beta @ beta)

    f_cur = nll(b)
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - ridge * b
        H = (X * W[:, None]).T @ X + ridge * np.eye(k)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            raise FitError("Hessian singular")
        if not np.all(np.isfinite(step)):
            raise FitError("non-finite Newton step")

        # Armijo-style backtracking with a tolerance: near the optimum the
        # objective changes by less than floating-point noise, so requiring
        # strict decrease would reject a converged fit. Accept any step that
        # does not increase the objective by more than that noise floor, and
        # treat an exhausted search at a tiny step as convergence rather
        # than failure.
        tol_f = 1e-9 * max(abs(f_cur), 1.0)
        t, accepted = 1.0, False
        for _ in range(40):
            f_new = nll(b + t * step)
            if np.isfinite(f_new) and f_new <= f_cur + tol_f:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            if np.max(np.abs(step)) < 1e-6:
                return b            # already at the optimum
            raise FitError("line search failed")

        b = b + t * step
        rel = abs(f_cur - f_new) / max(abs(f_cur), 1.0)
        gmax = float(np.max(np.abs(g)))
        f_cur = f_new
        # Converged when the objective stops moving relative to its own
        # size, or the gradient is negligible relative to n. An absolute
        # tolerance on the coefficient step is not attainable at large n.
        if rel < tol or gmax < 1e-6 * max(n, 1):
            return b
    raise FitError("did not converge within iteration limit")


def predict(X, b):
    return np.clip(1 / (1 + np.exp(-np.clip(X @ b, -30, 30))), 1e-12, 1 - 1e-12)


def auc_of(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sanity(ls, da, label):
    """An incremental log score far outside a plausible band means the fit is
    broken, not that the feature is harmful. Refuse to report it."""
    if not np.isfinite(ls) or abs(ls) > 0.05:
        raise FitError(f"{label}: implausible incremental log score {ls:+.4f}")
    if not np.isfinite(da) or abs(da) > 0.5:
        raise FitError(f"{label}: implausible dAUC {da:+.4f}")


def selftest():
    """Verify the fitter at the row count and event rates of the real run,
    with the recording artefacts real vitals contain, and check it against an
    independent solver rather than merely checking it returns.

    Three earlier versions of this script failed here in three different
    ways, none caught by an easier test: a fixed ridge with no line search
    diverged; a strict-decrease line search rejected converged fits; and an
    absolute coefficient-step tolerance of 1e-9 was unreachable at n in the
    hundreds of thousands, so every fit exhausted its iteration limit. The
    test is deliberately expensive because the cheap version passed while
    the real run was failing."""
    from scipy import optimize
    rng = np.random.default_rng(5)
    n = 490000
    Zr = rng.normal(size=(n, 5)) * np.array([20, 5, 3, 15, 1.5]) + \
        np.array([85, 18, 96, 75, 98.6])
    Zr[rng.random(n) < 0.002, 4] = 0.0
    Zr[rng.random(n) < 0.001, 0] = 300.0
    Zr[rng.random(n) < 0.001, 2] = 0.0
    lin = -3.3 + 0.5 * (Zr[:, 1] - 18) / 5 - 0.3 * np.clip(Zr[:, 2] - 96, -20, 20) / 3
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    folds = rng.integers(0, 5, n)
    x = rng.integers(0, 13, n).astype(str)
    print(f"self-test: n={n:,}, event rate {y.mean():.4f}, vital artefacts")
    for model in ("linear", "flex"):
        X = design(Zr, model)
        k = X.shape[1]
        ridge = 1e-3 * k
        b = logistic(X, y)

        def f(beta):
            eta = np.clip(X @ beta, -30, 30)
            p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) \
                + 0.5 * ridge * beta @ beta

        def gr(beta):
            eta = np.clip(X @ beta, -30, 30)
            p = 1 / (1 + np.exp(-eta))
            return -(X.T @ (y - p)) + ridge * beta

        ref = optimize.minimize(f, np.zeros(k), jac=gr, method="L-BFGS-B",
                                options={"maxiter": 2000, "ftol": 1e-14,
                                         "gtol": 1e-10})
        dev = float(np.max(np.abs(b - ref.x)))
        if dev > 1e-4:
            raise FitError(f"{model}: disagrees with L-BFGS by {dev:.2e}")
        ls, da = metrics(y, *oof(x, X, y, folds, 5))
        sanity(ls, da, model)
        print(f"  {model:<7} cols {k:>3}  vs L-BFGS {dev:.1e}  "
              f"noise-feature log score {ls:+.6f}  dAUC {da:+.5f}  OK")


def encode(vals, level):
    v = np.array([ALIAS.get(x, x) for x in vals]) if level >= 1 else np.asarray(vals)
    if level >= 2:
        v = np.array([WHEEZE.get(x, x) for x in v])
    if level >= 3:
        v = np.array([EICU.get(x, "diminished") for x in v])
    return v


def design(Zraw, model):
    """Baseline design. The expanded basis is standardised AFTER expansion.
    This equalises column scales but does not remove outlier leverage; the
    line search in logistic() is what actually prevents divergence."""
    Z = (Zraw - Zraw.mean(0)) / np.where(Zraw.std(0) > 0, Zraw.std(0), 1)
    if model == "linear":
        cols = Z
    elif model == "flex":
        k = Z.shape[1]
        prods = np.column_stack([Z[:, i] * Z[:, j]
                                 for i in range(k) for j in range(i + 1, k)])
        cols = np.column_stack([Z, Z ** 2, prods])
    else:
        sys.exit(f"unknown model {model!r}")
    sd = cols.std(0)
    cols = (cols - cols.mean(0)) / np.where(sd > 0, sd, 1)
    return np.column_stack([np.ones(len(cols)), cols])


def oof(vals, Xb, y, folds, n_folds):
    pb = np.full(len(y), np.nan)
    pf = np.full(len(y), np.nan)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or y[tr].sum() < 20:
            continue
        cats = sorted(set(vals[tr]))
        F = (np.zeros((len(vals), 0)) if len(cats) < 2 else
             np.column_stack([(vals == c).astype(float) for c in cats[1:]]))
        pb[te] = predict(Xb[te], logistic(Xb[tr], y[tr]))
        if F.shape[1] == 0:
            pf[te] = pb[te]
        else:
            Xf = np.column_stack([Xb, F])
            pf[te] = predict(Xf[te], logistic(Xf[tr], y[tr]))
    if np.isnan(pb).any() or np.isnan(pf).any():
        raise FitError("some folds produced no predictions")
    return pb, pf


def metrics(y, pb, pf):
    lb = np.sum(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    lf = np.sum(y * np.log(pf) + (1 - y) * np.log(1 - pf))
    return (lf - lb) / len(y), auc_of(y, pf) - auc_of(y, pb)


def build_outcome(name, intime, ext):
    """Return (event_hour dict, horizon) for the named outcome."""
    label, horizon = OUTCOMES[name]
    if name == "death":
        eh = {}
        # in-ICU death: hospital deathtime falling at or before ICU outtime
        death = {}
        with open_t(find(MIMIC / "hosp", "admissions")) as fh:
            for r in csv.DictReader(fh):
                dt = (r.get("deathtime") or "").strip()
                if dt:
                    death[r["hadm_id"]] = epoch(dt)
        with open_t(find(MIMIC / "icu", "icustays")) as fh:
            for r in csv.DictReader(fh):
                t = death.get(r["hadm_id"])
                if t is None:
                    continue
                b = epoch(r["intime"])
                o = epoch(r["outtime"])
                if b is None or o is None or t > o:
                    continue          # died after ICU discharge: not in-ICU
                eh[r["stay_id"]] = int((t - b) // 3600)
        return eh, None

    ids = set()
    rejected = []
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] != "procedureevents":
                continue
            lab = (r["label"] or "").strip()
            if lab.lower() == label.lower():
                ids.add(r["itemid"])
            elif "vent" in lab.lower():
                rejected.append(f"{r['itemid']}:{lab}")
    if not ids:
        sys.exit(f"no procedureevents item labelled exactly {label!r}")
    print(f"  {name}: itemid(s) {sorted(ids)}; not used: {rejected}",
          file=sys.stderr)
    eh = {}
    with open_t(find(MIMIC / "icu", "procedureevents")) as fh:
        for r in csv.DictReader(fh):
            if r["itemid"] not in ids:
                continue
            sid = r["stay_id"]
            b = epoch(intime.get(sid, "")) if sid in intime else None
            t = epoch(r["starttime"])
            if b is None or t is None:
                continue
            h = int((t - b) // 3600)
            if sid not in eh or h < eh[sid]:
                eh[sid] = h
    return eh, horizon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default="vent24,vent48,niv24,death")
    ap.add_argument("--models", default="linear,flex")
    ap.add_argument("--fields", default="RLL Lung Sounds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--cache", type=Path, default=Path("cache/cmi_extract.csv.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("cache"))
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.cache.exists():
        sys.exit(f"{a.cache} not found -- run cmi_screen.py first")

    selftest()          # never run a long job on an unverified fitter
    print()

    intime = {}
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        for r in csv.DictReader(fh):
            intime[r["stay_id"]] = r["intime"]

    ext = pd.read_csv(a.cache, dtype={"stay_id": str},
                      keep_default_na=False, na_values=[""])
    for c in BASE + ["nibp_map", "hour"]:
        ext[c] = pd.to_numeric(ext[c], errors="coerce")
    ext["map"] = ext["map"].fillna(ext["nibp_map"])
    ext = ext.dropna(subset=["hr", "rr", "spo2", "map"]).copy()
    ext["temp"] = ext["temp"].fillna(ext["temp"].median())

    stays = ext["stay_id"].unique()
    fold_of = dict(zip(stays, np.random.default_rng(0).integers(0, a.folds,
                                                               len(stays))))
    ext["fold"] = ext["stay_id"].map(fold_of)

    Zraw = ext[BASE].values.astype(float)

    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    rows = []

    for oname in [o.strip() for o in a.outcomes.split(",") if o.strip()]:
        if oname not in OUTCOMES:
            sys.exit(f"unknown outcome {oname}; choose from {list(OUTCOMES)}")
        eh_map, horizon = build_outcome(oname, intime, ext)
        eh = ext["stay_id"].map(eh_map)
        at_risk = ~(eh.notna() & (eh <= ext["hour"]))
        if horizon is None:
            y_all = ((eh.notna()) & (eh > ext["hour"])).astype(float).values
        else:
            y_all = ((eh.notna()) & (eh > ext["hour"]) &
                     (eh <= ext["hour"] + horizon)).astype(float).values
        risk = at_risk.values

        for model in models:
            Xb_all = design(Zraw, model)
            for lab in fields:
                m = risk & ext[lab].notna().values
                if m.sum() < 2000 or y_all[m].sum() < 50:
                    print(f"\n{oname}/{model}/{lab}: skipped "
                          f"({int(m.sum()):,} rows, {int(y_all[m].sum())} events)")
                    continue
                idx = np.flatnonzero(m)
                Xb, y, fd = Xb_all[idx], y_all[idx], ext["fold"].values[idx]
                vals0 = ext.loc[m, lab].astype(str).values
                sid = ext["stay_id"].values[idx]

                print(f"\n{'='*74}\n{oname} | {model} | {lab}")
                print(f"n {len(y):,}  stays {len(set(sid)):,}  "
                      f"events {int(y.sum()):,} ({y.mean():.4f})  "
                      f"baseline cols {Xb.shape[1]}", flush=True)

                try:
                    preds = {}
                    for lvl in (0, 1, 2, 3):
                        preds[lvl] = oof(encode(vals0, lvl), Xb, y, fd, a.folds)
                        ls, da = metrics(y, *preds[lvl])
                        sanity(ls, da, f"R{lvl}")
                        print(f"  R{lvl}  log-score {ls:+.6f}  dAUC {da:+.5f}",
                              flush=True)
                except FitError as e:
                    print(f"  FIT FAILED: {e}")
                    print("  cell not reported; a divergent fit is not a result")
                    rows.append({"outcome": oname, "model": model, "field": lab,
                                 "contrast": "n/a", "status": f"fit failed: {e}"})
                    continue

                base_ls = metrics(y, *preds[0])[0]
                uniq = np.unique(sid)
                by = {s: np.flatnonzero(sid == s) for s in uniq}
                brng = np.random.default_rng(1)
                draws = {2: [], 3: []}
                for _ in range(a.boot):
                    pick = brng.choice(len(uniq), len(uniq), replace=True)
                    rr = np.concatenate([by[uniq[i]] for i in pick])
                    yb = y[rr]
                    if yb.sum() < 20 or yb.sum() == len(yb):
                        continue
                    m0 = metrics(yb, preds[0][0][rr], preds[0][1][rr])
                    for lvl in (2, 3):
                        mk = metrics(yb, preds[lvl][0][rr], preds[lvl][1][rr])
                        draws[lvl].append((mk[0] - m0[0], mk[1] - m0[1]))

                marg = a.margin * abs(base_ls)
                for lvl in (2, 3):
                    arr = np.array(draws[lvl])
                    pt = metrics(y, *preds[lvl])[0] - base_ls
                    lo, hi = np.percentile(arr[:, 0], [2.5, 97.5])
                    alo, ahi = np.percentile(arr[:, 1], [2.5, 97.5])
                    inside = (lo > -marg) and (hi < marg)
                    print(f"  R{lvl}-R0 log score {pt:+.6f} "
                          f"[{lo:+.6f}, {hi:+.6f}]  "
                          f"margin \u00b1{marg:.6f}: "
                          f"{'INSIDE' if inside else 'NOT inside'}", flush=True)
                    rows.append({"outcome": oname, "model": model, "field": lab,
                                 "n": int(m.sum()), "events": int(y.sum()),
                                 "contrast": f"R{lvl}-R0",
                                 "native_log_score": base_ls,
                                 "point": pt, "lo": lo, "hi": hi,
                                 "auc_lo": alo, "auc_hi": ahi,
                                 "margin": marg,
                                 "inside_margin": bool(inside),
                                 "status": "ok"})

    if rows:
        df = pd.DataFrame(rows)
        p = a.out_dir / "robustness.csv"
        df.to_csv(p, index=False)
        print(f"\n{'='*74}\nwrote {p}")
        ok = df[df.status == "ok"]
        if len(ok):
            print(ok[["outcome", "model", "contrast", "events", "point",
                      "lo", "hi", "inside_margin"]].to_string(
                index=False, float_format=lambda x: f"{x:,.6f}"))
            print(f"\ncontrasts inside the pre-specified margin: "
                  f"{int(ok.inside_margin.sum())}/{len(ok)}")
        bad = df[df.status != "ok"]
        if len(bad):
            print(f"\ncells NOT reported (fit failure): {len(bad)}")
            print(bad[["outcome", "model", "status"]].to_string(index=False))
        print("A null that holds across outcomes and model classes is a "
              "stable null.\nAny cell that falls outside the margin must be "
              "reported, not dropped.")


if __name__ == "__main__":
    main()
