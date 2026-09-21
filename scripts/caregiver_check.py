#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy"]
# ///
"""
Does one clinician document both lower-lobe lung-sound fields?

The note currently asserts "one clinician, one moment" from identical
charttimes and 99.64% co-population. That establishes a single assessment
episode, not a single documenter. MIMIC-IV chartevents carries caregiver_id,
so the claim is directly testable.

Reports, over paired right/left lower-lobe entries at identical charttime:
    - share of pairs with identical caregiver_id
    - share where either side has a missing caregiver_id
    - for discordant pairs, whether the two ids recur together across the
      cohort (which would suggest shift handover or a shared login rather
      than two independent assessors)

CAVEAT, stated before the result: earlier work in this cohort found 98.8%
of caregiver-hours are single-patient, which is not how ICU nursing works.
caregiver_id may therefore not map cleanly to persons. That matters less
here than elsewhere, because the question is whether two rows written at
the same instant carry the same value, which is a within-record comparison.
Agreement is evidence of one documenting session. Disagreement would need
explaining before any "one clinician" phrasing could stand.

Usage:  ./caregiver_check.py
"""

import csv
import gzip
import sys
from collections import Counter, defaultdict
import os
from pathlib import Path

import pandas as pd

MIMIC = Path(os.environ.get("MIMIC_IV_DIR", "MIMIC_IV_DIR_not_set"))
RLL, LLL = "223987", "223989"


def open_t(p: Path):
    return gzip.open(p, "rt", newline="") if p.suffix == ".gz" else open(p, "rt", newline="")


def find(root: Path, name: str) -> Path:
    for ext in (".csv.gz", ".csv"):
        p = root / (name + ext)
        if p.exists():
            return p
    sys.exit(f"{name} not found under {root}")


def main() -> None:
    src = find(MIMIC / "icu", "chartevents")
    with open_t(src) as fh:
        cols = csv.DictReader(fh).fieldnames or []
    if "caregiver_id" not in cols:
        sys.exit(f"no caregiver_id column in chartevents; columns are {cols}")

    seen = defaultdict(dict)      # (stay, charttime) -> side -> caregiver_id
    n = 0
    with open_t(src) as fh:
        for r in csv.DictReader(fh):
            n += 1
            if n % 50_000_000 == 0:
                print(f"  {n:,} rows ...", file=sys.stderr)
            iid = r["itemid"]
            if iid not in (RLL, LLL):
                continue
            side = "RLL" if iid == RLL else "LLL"
            seen[(r["stay_id"], r["charttime"])][side] = (r.get("caregiver_id") or "").strip()
    print(f"scanned {n:,} rows")

    same = diff = missing = 0
    pairs = Counter()
    for d in seen.values():
        if len(d) != 2:
            continue
        a, b = d["RLL"], d["LLL"]
        if not a or not b:
            missing += 1
        elif a == b:
            same += 1
        else:
            diff += 1
            pairs[tuple(sorted((a, b)))] += 1

    tot = same + diff + missing
    print(f"\npaired observations with both sides: {tot:,}")
    print(f"  identical caregiver_id : {same:,}  ({same/tot:.4f})")
    print(f"  different caregiver_id : {diff:,}  ({diff/tot:.4f})")
    print(f"  one or both missing    : {missing:,}  ({missing/tot:.4f})")

    if diff:
        print(f"\ndiscordant id pairs: {len(pairs):,} distinct combinations")
        rec = sum(c for c in pairs.values() if c >= 5)
        print(f"  pairs recurring \u22655 times: {rec:,}/{diff:,} = {rec/diff:.3f}")
        print("  (high recurrence suggests handover or shared logins,")
        print("   not independent assessment by two clinicians)")
        print("\n  top recurring combinations:")
        for (a, b), c in pairs.most_common(10):
            print(f"    {c:>7,}  {a} / {b}")

    verdict = same / tot if tot else float("nan")
    print(f"\nvalue for the note: {verdict:.4f} of paired observations share a "
          f"caregiver_id")
    if verdict >= 0.95:
        print("=> 'one clinician' is supported")
    elif verdict >= 0.80:
        print("=> 'one clinician' holds for most but not all pairs; state the share")
    else:
        print("=> 'one clinician' is NOT supported; revise the note to "
              "'one assessment episode'")


if __name__ == "__main__":
    main()
