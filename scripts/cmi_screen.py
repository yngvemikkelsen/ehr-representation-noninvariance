#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy"]
# ///
"""
ROLE IN THE PUBLISHED ANALYSIS
  This script is used ONLY to build cache/cmi_extract.csv.gz, the stay-hour
  extract read by collapse_experiment.py, paired_bootstrap.py and
  robustness.py. Run it with --extract-only. The screening statistics it
  prints otherwise use a within-stay permutation null that is invalid (within
  a stay the outcome is near-constant, so shuffling values among that stay's
  hours preserves the association the null should destroy). They are NOT
  reported in the paper and should not be used. They are retained, rather
  than deleted, so that the file is the one that produced the extract.

Does a categorical assessment field carry task-relevant information
CONDITIONAL on the routine covariates a model would already have?

Feasibility gate for the graded-collapse experiment. Collapsing a feature's
value set can only degrade prediction where

    I(X ; Y | rest) > 0

for the task at hand. If lung-sound subtype is conditionally independent of Y
given vitals, every collapse level measures zero and the factorial cannot be
built on that construct.

TWO STAGES
  The chartevents scan takes ~45 minutes. It runs once and writes
  cache/cmi_extract.csv.gz. Changing the outcome, the field list or the
  permutation count reuses the cache. --rescan forces a rebuild.

OUTCOME
  Invasive ventilation started within the horizon, matched on the EXACT
  d_items label "Invasive Ventilation". Substring matching is unsafe:
  "Non-Invasive Ventilation" contains "invasive vent", and an earlier run
  silently pooled the two. Rejected ventilation items are printed.
  Stay-hours already ventilated are dropped: not at risk.

ESTIMATOR
  Per-observation log-likelihood gain from adding the one-hot feature to a
  baseline logistic model, in nats, reported as a share of outcome entropy.

NULL
  At n in the millions any feature is "significant". Values are permuted
  WITHIN stay, preserving outcome, covariates and per-stay value composition,
  and the fit repeated. Judge the observed gain against that null, not zero.

Usage:
  ./cmi_screen.py                  # uses cache if present
  ./cmi_screen.py --rescan         # rebuild the extract
  ./cmi_screen.py --perms 50       # faster null
"""

import argparse
import csv
import gzip
import sys
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))

CANDIDATES = ["RLL Lung Sounds", "LLL Lung Sounds", "RUL Lung Sounds",
              "LUL Lung Sounds", "Heart Rhythm", "Ectopy Type 1",
              "O2 Delivery Device(s)", "Skin Color", "Pupil Response Right"]

VITALS = {"Heart Rate": "hr", "Respiratory Rate": "rr",
          "O2 saturation pulseoxymetry": "spo2",
          "Arterial Blood Pressure mean": "map",
          "Non Invasive Blood Pressure mean": "nibp_map",
          "Temperature Fahrenheit": "temp"}

VCOLS = ["hr", "rr", "spo2", "map", "nibp_map", "temp"]


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    sys.exit(f"{name} not found under {root}")


def epoch(ts: str):
    """Parse 'YYYY-MM-DD HH:MM:SS' to seconds. Manual, because pd.Timestamp
    per row dominated the runtime of the previous version."""
    try:
        y = int(ts[0:4]); mo = int(ts[5:7]); d = int(ts[8:10])
        h = int(ts[11:13]); mi = int(ts[14:16])
        yy = y - (1 if mo <= 2 else 0)
        era = yy // 400
        yoe = yy - era * 400
        doy = (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + d - 1
        doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
        days = era * 146097 + doe - 719468
        return days * 86400 + h * 3600 + mi * 60
    except (ValueError, IndexError, TypeError):
        return None


def scan(feat_id, vital_id, intime, cache):
    labels = sorted(set(feat_id.values()))
    t0 = {s: epoch(t) for s, t in intime.items()}
    feat = defaultdict(dict)
    vit = defaultdict(dict)
    n = 0
    with open_t(find(MIMIC / "icu", "chartevents")) as fh:
        for r in csv.DictReader(fh):
            n += 1
            if n % 50_000_000 == 0:
                print(f"  {n:,} rows ...", file=sys.stderr)
            iid = r["itemid"]
            lab = feat_id.get(iid)
            vt = vital_id.get(iid)
            if lab is None and vt is None:
                continue
            base = t0.get(r["stay_id"])
            if base is None:
                continue
            t = epoch(r["charttime"])
            if t is None:
                continue
            h = (t - base) // 3600
            if h < 0 or h >= 72:
                continue
            key = (r["stay_id"], int(h))
            if lab is not None:
                v = (r.get("value") or "").strip()
                if v:
                    feat[key][lab] = v
            else:
                try:
                    vit[key][vt] = float(r["valuenum"])
                except (TypeError, ValueError):
                    pass
    print(f"scanned {n:,} rows", file=sys.stderr)

    rows = []
    for key in feat:
        sid, h = key
        rec = {"stay_id": sid, "hour": h}
        rec.update(feat[key])
        rec.update(vit.get(key, {}))
        rows.append(rec)
    df = pd.DataFrame(rows)
    for c in labels + VCOLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[["stay_id", "hour"] + labels + VCOLS]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"wrote {cache} ({len(df):,} stay-hours)", file=sys.stderr)
    return df


def logistic(X, y, iters=40):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p) - 1e-6 * b
        H = (X * W[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ b, -30, 30)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return b, float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), p


def auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float("nan") if n1 == 0 or n0 == 0 else \
        float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def onehot(vals):
    cats = sorted(set(vals))
    return np.zeros((len(vals), 0)) if len(cats) < 2 else \
        np.column_stack([(vals == c).astype(float) for c in cats[1:]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--fields", default="")
    ap.add_argument("--out-dir", type=Path, default=Path("cache"))
    ap.add_argument("--extract-only", action="store_true",
                    help="build cache/cmi_extract.csv.gz and stop; the screening\n                         statistics below are not used in the paper")
    a = ap.parse_args()
    a.out_dir.mkdir(exist_ok=True)
    cache = a.out_dir / "cmi_extract.csv.gz"

    feat_id, vital_id = {}, {}
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] != "chartevents":
                continue
            lab = r["label"] or ""
            if lab in CANDIDATES:
                feat_id[r["itemid"]] = lab
            if lab in VITALS:
                vital_id[r["itemid"]] = VITALS[lab]
    labels = sorted(set(feat_id.values()))
    print(f"candidate fields: {labels}", file=sys.stderr)

    intime = {}
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        for r in csv.DictReader(fh):
            intime[r["stay_id"]] = r["intime"]

    vent_ids, rejected = set(), []
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] != "procedureevents":
                continue
            lab = (r["label"] or "").strip()
            if lab.lower() == "invasive ventilation":
                vent_ids.add(r["itemid"])
            elif "vent" in lab.lower():
                rejected.append(f"{r['itemid']}:{lab}")
    print(f"invasive-ventilation itemid(s): {sorted(vent_ids)}", file=sys.stderr)
    print(f"  ventilation items NOT used: {rejected}", file=sys.stderr)
    if not vent_ids:
        sys.exit("no item labelled exactly 'Invasive Ventilation'")

    event_hour = {}
    with open_t(find(MIMIC / "icu", "procedureevents")) as fh:
        for r in csv.DictReader(fh):
            if r["itemid"] not in vent_ids:
                continue
            sid = r["stay_id"]
            b = epoch(intime.get(sid, "")) if sid in intime else None
            t = epoch(r["starttime"])
            if b is None or t is None:
                continue
            h = int((t - b) // 3600)
            if sid not in event_hour or h < event_hour[sid]:
                event_hour[sid] = h
    print(f"stays with invasive ventilation: {len(event_hour):,}", file=sys.stderr)

    if cache.exists() and not a.rescan:
        # keep_default_na=False: "None" is a real category (no ectopy, no
        # oxygen device) and pandas' default NA list would drop it.
        ext = pd.read_csv(cache, dtype={"stay_id": str},
                          keep_default_na=False, na_values=[""])
        for c in VCOLS + ["hour"]:
            ext[c] = pd.to_numeric(ext[c], errors="coerce")
        for c in labels:
            if c in ext.columns:
                ext[c] = ext[c].replace("", np.nan)
        print(f"loaded {cache} ({len(ext):,} stay-hours); --rescan to rebuild",
              file=sys.stderr)
    else:
        ext = scan(feat_id, vital_id, intime, cache)
        ext["stay_id"] = ext["stay_id"].astype(str)

    if a.extract_only:
        print(f"extract ready: {cache}")
        return

    ext["map"] = ext["map"].fillna(ext["nibp_map"])
    ext = ext.dropna(subset=["hr", "rr", "spo2", "map"]).copy()
    ext["temp"] = ext["temp"].fillna(ext["temp"].median())

    eh = ext["stay_id"].map(event_hour)
    ext = ext[~(eh.notna() & (eh <= ext["hour"]))].copy()
    eh = ext["stay_id"].map(event_hour)
    ext["y"] = ((eh.notna()) & (eh > ext["hour"]) &
                (eh <= ext["hour"] + a.window)).astype(int)

    print(f"\nstay-hours at risk with baseline vitals: {len(ext):,}")
    print(f"event rate (invasive ventilation within {a.window}h): "
          f"{ext.y.mean():.4f}")

    Z = ext[["hr", "rr", "spo2", "map", "temp"]].values.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    Xb_all = np.column_stack([np.ones(len(ext)), Z])
    y_all = ext.y.values.astype(float)

    want = [f.strip() for f in a.fields.split(",") if f.strip()] or \
        [c for c in labels if c in ext.columns]

    rng = np.random.default_rng(0)
    out = []
    for lab in want:
        m = ext[lab].notna().values
        if m.sum() < 2000 or y_all[m].sum() < 50:
            print(f"\n{lab}: skipped ({int(m.sum()):,} rows, "
                  f"{int(y_all[m].sum())} events)")
            continue
        Xb, ys = Xb_all[m], y_all[m]
        vals = ext.loc[m, lab].astype(str).values
        stays = ext.loc[m, "stay_id"].values
        F = onehot(vals)
        if F.shape[1] == 0:
            continue
        _, ll0, p0 = logistic(Xb, ys)
        _, ll1, p1 = logistic(np.column_stack([Xb, F]), ys)
        mi = (ll1 - ll0) / len(ys)
        py = ys.mean()
        h_y = float(-(py * np.log(py) + (1 - py) * np.log(1 - py)))

        order = np.argsort(stays, kind="stable")
        so = stays[order]
        bnd = np.flatnonzero(np.r_[True, so[1:] != so[:-1], True])
        null = []
        for _ in range(a.perms):
            perm = vals.copy()
            vo = perm[order]
            for s_, e_ in zip(bnd[:-1], bnd[1:]):
                if e_ - s_ > 1:
                    vo[s_:e_] = rng.permutation(vo[s_:e_])
            perm[order] = vo
            Fp = onehot(perm)
            if Fp.shape[1] != F.shape[1]:
                continue
            _, llp, _ = logistic(np.column_stack([Xb, Fp]), ys)
            null.append((llp - ll0) / len(ys))
        null = np.array(null) if null else np.array([np.nan])
        p_perm = float((null >= mi).mean())
        ratio = mi / max(float(np.nanmean(null)), 1e-12)

        out.append({"field": lab, "n": int(m.sum()), "levels": F.shape[1] + 1,
                    "events": int(ys.sum()), "MI_nats": mi,
                    "MI_pct_H": 100 * mi / h_y,
                    "null_mean": float(np.nanmean(null)),
                    "ratio_to_null": ratio, "p_perm": p_perm,
                    "AUC_base": auc(ys, p0), "AUC_full": auc(ys, p1)})
        print(f"\n{lab}")
        print(f"  n {int(m.sum()):,}  levels {F.shape[1]+1}  "
              f"events {int(ys.sum()):,} ({py:.4f})")
        print(f"  conditional MI {mi:.6f} nats  ({100*mi/h_y:.3f}% of H(Y))")
        print(f"  null mean {np.nanmean(null):.6f}   ratio {ratio:.2f}   "
              f"p {p_perm:.3f}")
        print(f"  AUC {auc(ys, p0):.4f} -> {auc(ys, p1):.4f}")

    if not out:
        sys.exit("\nno field met the minimum criteria")
    t = pd.DataFrame(out).sort_values("MI_nats", ascending=False)
    t.to_csv(a.out_dir / "cmi_screen.csv", index=False)
    print("\n" + "=" * 74)
    print(t.to_string(index=False, float_format=lambda x: f"{x:,.5f}"))
    print("\nVIABILITY FOR THE GRADED-COLLAPSE EXPERIMENT")
    print("(p alone is meaningless at this n; the ratio to the null and the")
    print(" AUC change are what decide whether a collapse could be measured)")
    for _, r in t.iterrows():
        d_auc = r.AUC_full - r.AUC_base
        if r.p_perm < 0.05 and r.ratio_to_null >= 3 and d_auc >= 0.005:
            v = "viable"
        elif r.p_perm < 0.05 and r.ratio_to_null >= 2:
            v = "marginal \u2014 real but small; collapse effects will be tiny"
        else:
            v = "NOT viable \u2014 conditionally uninformative for this task"
        print(f"  {r.field:<24} ratio {r.ratio_to_null:>6.2f}  "
              f"dAUC {d_auc:+.4f}   {v}")


if __name__ == "__main__":
    main()
