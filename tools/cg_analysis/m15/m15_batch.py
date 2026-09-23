#!/usr/bin/env python3
"""Stage 2: batch entry for a case root or a whole matrix root.

For every directory that contains CGParticles_ite_*.vtp it runs the frozen
detector once plus two pre-declared robustness perturbations, evaluates the
formal F3 criterion and marks ROBUST-F3 only if F3 survives all three
configurations (otherwise ANALYSIS-SENSITIVE).

Usage:
  python m15_batch.py <root> --R0 <R0> --Lz 80 [--t-lo 40] [--out table.csv]
  python m15_batch.py <matrix_root> --R0 5 --Lz 120 --matrix
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# frozen global detector configurations: the primary one plus two perturbations
CONFIGS = [("frozen", ["--smooth", "2", "--prom-frac", "0.25"]),
           ("strict", ["--smooth", "3", "--prom-frac", "0.35"]),
           ("loose", ["--smooth", "1", "--prom-frac", "0.20"])]


def run_case(d, R0, Lz, t_lo, label):
    out = {}
    for name, extra in CONFIGS:
        subprocess.run([sys.executable, os.path.join(HERE, "m15_beads.py"), d,
                        "--R0", str(R0), "--Lz", str(Lz), "--t-lo", str(t_lo),
                        "--out", os.path.join(d, "m15_%s.csv" % name)] + extra,
                       check=True, capture_output=True)
        rows = [r for r in csv.DictReader(open(os.path.join(d, "m15_%s.csv" % name),
                                               newline="", encoding="utf-8"))
                if r["method"] == "A"]
        cls = [r["class"] for r in rows]
        bc = np.array([int(r["bead_count"]) for r in rows], float)
        out[name] = dict(F3=cls.count("F3") / len(cls),
                         F2=cls.count("F2") / len(cls),
                         max_bead=int(bc.max()),
                         med_CV=float(np.nanmedian([float(r["CV_spacing"]) for r in rows]))
                         if len(rows) else np.nan)
    f3s = [out[c[0]]["F3"] >= 0.5 for c in CONFIGS]
    res = dict(case=label, R0=R0,
               F3_frozen=round(out["frozen"]["F3"], 3),
               F3_strict=round(out["strict"]["F3"], 3),
               F3_loose=round(out["loose"]["F3"], 3),
               F2_frozen=round(out["frozen"]["F2"], 3),
               max_bead_frozen=out["frozen"]["max_bead"],
               median_CV_frozen=round(out["frozen"]["med_CV"], 3),
               ROBUST_F3=bool(all(f3s)),
               ANALYSIS_SENSITIVE=bool(any(f3s) and not all(f3s)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=80.0)
    ap.add_argument("--t-lo", type=float, default=40.0)
    ap.add_argument("--matrix", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.matrix:
        dirs = sorted({os.path.dirname(p) for p in
                       glob.glob(os.path.join(a.root, "**", "CGParticles_ite_*.vtp"),
                                 recursive=True)})
    else:
        dirs = [a.root]
    rows = [run_case(d, a.R0, a.Lz, a.t_lo, os.path.basename(d.rstrip("\\/")))
            for d in dirs]
    out = a.out or os.path.join(a.root, "robust_f3_table.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for r in rows:
        print("%-18s F3(frozen/strict/loose)=%.3f/%.3f/%.3f  ROBUST-F3=%s  SENSITIVE=%s"
              % (r["case"], r["F3_frozen"], r["F3_strict"], r["F3_loose"],
                 r["ROBUST_F3"], r["ANALYSIS_SENSITIVE"]))
    print("-> %s" % out)


if __name__ == "__main__":
    main()
