#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Variable x variable correlation heatmap, one per dataset, with the
organisational unit entered as dummy variables in the matrix itself.

Rows and columns are the SAME list:

    [ all clinical variables ]  +  [ one dummy per unit ]

so the bottom-right block is unit x unit, the top-left block is the ordinary
clinical correlation structure, and the off-diagonal block is what actually
matters here: how strongly each clinical variable loads on site identity.

    MIMIC-IV   one institution -> dummies are CARE UNITS
    eICU       208 hospitals   -> dummies are HOSPITALS

Per stay, each clinical variable is its mean recorded value when the values
parse as numeric, and a 0/1 presence flag otherwise (flags are suffixed
[flag] in the labels). Each dummy is 1 if the stay is at that unit.
Correlations are Pearson, pairwise-complete.

Output is a self-contained HTML file (no matplotlib) plus the raw matrix
as CSV.

Usage:
  ./corr_heatmap.py --dataset mimic
  ./corr_heatmap.py --dataset eicu --top 40
"""

import argparse
import csv
import gzip
import html
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))
EICU = Path(os.environ.get("EICU_CRD_DIR", "EICU_CRD_DIR_not_set"))


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not under {root}")


def to_float(s: str):
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def accumulate(rows, stay_ix, var_ix, n_stay, n_var):
    """Sum/count/presence accumulators over (stay, variable) pairs."""
    tot = np.zeros((n_var, n_stay))
    cnt = np.zeros((n_var, n_stay))
    pres = np.zeros((n_var, n_stay), dtype=bool)
    for sid, var, val in rows:
        i, j = var_ix.get(var), stay_ix.get(sid)
        if i is None or j is None:
            continue
        pres[i, j] = True
        if val is not None:
            tot[i, j] += val
            cnt[i, j] += 1
    return tot, cnt, pres


def assemble(tot, cnt, pres, varnames, min_numeric=0.5):
    """Mean value where mostly numeric, else presence flag."""
    cols, labels = [], []
    for i, name in enumerate(varnames):
        seen = pres[i]
        if not seen.any():
            continue
        numeric_share = (cnt[i][seen] > 0).mean()
        if numeric_share >= min_numeric:
            col = np.full(pres.shape[1], np.nan)
            ok = cnt[i] > 0
            col[ok] = tot[i][ok] / cnt[i][ok]
            labels.append(name)
        else:
            col = seen.astype(float)
            labels.append(f"{name} [flag]")
        cols.append(col)
    return np.vstack(cols).T, labels


def load_eicu(top: int, min_stays: int):
    hosp_of = {}
    hosp_n = defaultdict(int)
    with open_t(find(EICU, "patient")) as fh:
        for r in csv.DictReader(fh):
            hosp_of[r["patientunitstayid"]] = r["hospitalid"]
            hosp_n[r["hospitalid"]] += 1
    hosps = sorted([h for h, n in hosp_n.items() if n >= min_stays],
                   key=lambda h: -hosp_n[h])
    keep = set(hosps)
    stays = [s for s, h in hosp_of.items() if h in keep]
    stay_ix = {s: j for j, s in enumerate(stays)}

    print("pass 1/2: variable prevalence ...")
    freq = defaultdict(int)
    src = find(EICU, "nurseCharting")
    with open_t(src) as fh:
        for n, r in enumerate(csv.DictReader(fh), 1):
            if n % 20_000_000 == 0:
                print(f"  {n:,}")
            if r["patientunitstayid"] in stay_ix:
                lab = r.get("nursingchartcelltypevallabel") or ""
                if lab:
                    freq[lab] += 1
    varnames = [v for v, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:top]]
    var_ix = {v: i for i, v in enumerate(varnames)}

    print("pass 2/2: values ...")

    def gen():
        with open_t(src) as fh:
            for n, r in enumerate(csv.DictReader(fh), 1):
                if n % 20_000_000 == 0:
                    print(f"  {n:,}")
                yield (r["patientunitstayid"],
                       r.get("nursingchartcelltypevallabel") or "",
                       to_float(r.get("nursingchartvalue")))

    tot, cnt, pres = accumulate(gen(), stay_ix, var_ix, len(stays), len(varnames))
    X, labels = assemble(tot, cnt, pres, varnames)

    D = np.zeros((len(stays), len(hosps)))
    hix = {h: k for k, h in enumerate(hosps)}
    for s, j in stay_ix.items():
        D[j, hix[hosp_of[s]]] = 1.0
    return X, labels, D, [f"hosp_{h}" for h in hosps]


def load_mimic(top: int, min_stays: int):
    unit_of, unit_n = {}, defaultdict(int)
    with open_t(find(MIMIC / "icu", "icustays")) as fh:
        for r in csv.DictReader(fh):
            unit_of[r["stay_id"]] = r["first_careunit"]
            unit_n[r["first_careunit"]] += 1
    units = sorted([u for u, n in unit_n.items() if n >= min_stays],
                   key=lambda u: -unit_n[u])
    keep = set(units)
    stays = [s for s, u in unit_of.items() if u in keep]
    stay_ix = {s: j for j, s in enumerate(stays)}

    label_of = {}
    with open_t(find(MIMIC / "icu", "d_items")) as fh:
        for r in csv.DictReader(fh):
            if r["linksto"] == "chartevents":
                label_of[r["itemid"]] = r["label"] or r["itemid"]

    print("pass 1/2: variable prevalence ...")
    freq = defaultdict(int)
    src = find(MIMIC / "icu", "chartevents")
    with open_t(src) as fh:
        for n, r in enumerate(csv.DictReader(fh), 1):
            if n % 50_000_000 == 0:
                print(f"  {n:,}")
            if r["stay_id"] in stay_ix and r["itemid"] in label_of:
                freq[label_of[r["itemid"]]] += 1
    varnames = [v for v, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:top]]
    var_ix = {v: i for i, v in enumerate(varnames)}

    print("pass 2/2: values ...")

    def gen():
        with open_t(src) as fh:
            for n, r in enumerate(csv.DictReader(fh), 1):
                if n % 50_000_000 == 0:
                    print(f"  {n:,}")
                lab = label_of.get(r["itemid"])
                if lab is not None:
                    yield (r["stay_id"], lab, to_float(r.get("valuenum")))

    tot, cnt, pres = accumulate(gen(), stay_ix, var_ix, len(stays), len(varnames))
    X, labels = assemble(tot, cnt, pres, varnames)

    D = np.zeros((len(stays), len(units)))
    uix = {u: k for k, u in enumerate(units)}
    for s, j in stay_ix.items():
        D[j, uix[unit_of[s]]] = 1.0
    return X, labels, D, [f"unit_{u}" for u in units]


def diverging(v: float) -> str:
    """blue (-1) -> dark (0) -> orange (+1)."""
    if not np.isfinite(v):
        return "#333"
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        t = v
        r, g, b = 24 + t * 228, 20 + t * 150, 34 + t * 40
    else:
        t = -v
        r, g, b = 24 + t * 20, 20 + t * 150, 34 + t * 200
    return f"rgb({int(r)},{int(g)},{int(b)})"


def build_html(C: pd.DataFrame, n_clin: int, title: str) -> str:
    k = len(C)
    cs = max(3, min(14, int(1000 / k)))
    left = 250
    top = 250
    grid = cs * k

    cells = []
    for i in range(k):
        for j in range(k):
            v = C.iat[i, j]
            cells.append(
                f'<rect x="{left+j*cs}" y="{top+i*cs}" width="{cs}" height="{cs}" '
                f'fill="{diverging(v)}"><title>{html.escape(C.index[i])} vs '
                f'{html.escape(C.columns[j])}: {v:.3f}</title></rect>')

    labs = []
    step = 1 if cs >= 8 else max(1, int(np.ceil(9 / cs)))
    for i in range(0, k, step):
        nm = C.index[i][:36]
        col = "#7fd1e0" if i >= n_clin else "#ddd"
        labs.append(f'<text x="{left-6}" y="{top+i*cs+cs-1}" text-anchor="end" '
                    f'font-size="{min(9,cs)}" fill="{col}">{html.escape(nm)}</text>')
        labs.append(f'<text x="{left+i*cs+cs-1}" y="{top-6}" font-size="{min(9,cs)}" '
                    f'fill="{col}" transform="rotate(-90 {left+i*cs+cs-1} {top-6})">'
                    f'{html.escape(nm)}</text>')

    sep = (f'<line x1="{left+n_clin*cs}" y1="{top}" x2="{left+n_clin*cs}" '
           f'y2="{top+grid}" stroke="#7fd1e0" stroke-width="1"/>'
           f'<line x1="{left}" y1="{top+n_clin*cs}" x2="{left+grid}" '
           f'y2="{top+n_clin*cs}" stroke="#7fd1e0" stroke-width="1"/>')

    leg = "".join(f'<rect x="{left+m*11}" y="{top+grid+20}" width="11" height="11" '
                  f'fill="{diverging(-1 + m/14)}"/>' for m in range(29))

    W, H = left + grid + 40, top + grid + 70
    return f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(title)}</title>
<body style="background:#111;color:#ddd;font-family:system-ui,sans-serif;margin:16px">
<h2 style="font-size:15px">{html.escape(title)}</h2>
<p style="font-size:11px;color:#999;max-width:900px">
Pearson correlation, pairwise-complete. Blue labels past the divider are the
unit dummies. Top-left block = clinical variables against each other;
bottom-right = unit against unit (necessarily negative, dummies are mutually
exclusive); off-diagonal block = each variable's loading on site identity.
<b>[flag]</b> means the variable was non-numeric and is coded as presence.
Hover for exact values.</p>
<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">
{''.join(cells)}{sep}{''.join(labs)}{leg}
<text x="{left}" y="{top+grid+46}" font-size="9" fill="#999">-1</text>
<text x="{left+150}" y="{top+grid+46}" font-size="9" fill="#999">0</text>
<text x="{left+305}" y="{top+grid+46}" font-size="9" fill="#999">+1</text>
</svg></body>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimic", "eicu"], required=True)
    ap.add_argument("--top", type=int, default=40, help="clinical variables to keep")
    ap.add_argument("--min-stays", type=int, default=200)
    ap.add_argument("--max-dummies", type=int, default=30,
                    help="largest N units kept as dummies; eICU has 208")
    ap.add_argument("--out-dir", type=Path, default=Path("cache"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loader = load_mimic if args.dataset == "mimic" else load_eicu
    X, labels, D, dnames = loader(args.top, args.min_stays)

    if D.shape[1] > args.max_dummies:
        D, dnames = D[:, :args.max_dummies], dnames[:args.max_dummies]

    full = pd.DataFrame(np.hstack([X, D]), columns=labels + dnames)
    C = full.corr(method="pearson", min_periods=50)

    stem = f"corr_{args.dataset}"
    C.to_csv(args.out_dir / f"{stem}.csv")
    title = ("MIMIC-IV \u2014 variable correlation with care-unit dummies"
             if args.dataset == "mimic"
             else "eICU-CRD \u2014 variable correlation with hospital dummies")
    (args.out_dir / f"{stem}.html").write_text(
        build_html(C, len(labels), title), encoding="utf-8")

    print(f"\nwrote {args.out_dir / f'{stem}.html'}")
    print(f"wrote {args.out_dir / f'{stem}.csv'}   ({C.shape[0]}x{C.shape[1]})")

    # strongest clinical-clinical pairs (the selection scan reported in the
    # manuscript, section 3.2); reporting only, the matrix is unchanged
    cc = C.iloc[:len(labels), :len(labels)]
    pairs = (cc.where(np.triu(np.ones(cc.shape, dtype=bool), 1))
             .stack().dropna().sort_values(key=abs, ascending=False))
    print("\nstrongest clinical-clinical pairs (top 10)")
    for (a_, b_), r in pairs.head(10).items():
        print(f"  {r:+.4f}  {a_}  x  {b_}")

    block = C.iloc[:len(labels), len(labels):].abs()
    load = block.max(axis=1).sort_values(ascending=False)
    print("\nstrongest single-unit loading per variable (top 15)")
    print(load.head(15).round(3).to_string())


if __name__ == "__main__":
    main()
