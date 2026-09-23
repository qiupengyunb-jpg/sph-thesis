#!/usr/bin/env python3
"""M1.4: ensemble (multi-seed) mode analysis.

Aggregates the per-frame m14_modes.csv files of several seeds of the SAME
initial condition and estimates the variance growth rate

    d ln <A_m^2> / dt = 2 omega_m

with a fixed window, plus the same quantity for the A = 0 baseline, so that the
imposed-mode response can be compared with the spontaneous one.

Usage:
  python m14_ensemble.py --mode 4 --pert dir_seed1 dir_seed2 dir_seed3 \
                         --base base_seed1 base_seed2 base_seed3 \
                         --t0 5 --t1 60 [--mmax 12]
"""

import argparse
import csv
import math
import os

import numpy as np


def load_series(dirs, m, tag="A"):
    """Return (t, <A_m^2>) averaged over the given run directories."""
    key = "A%d_%s" % (m, tag)
    stack = None
    t = None
    for d in dirs:
        p = os.path.join(d, "m14_modes.csv")
        rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        tt = np.array([float(r["time"]) for r in rows])
        aa = np.array([float(r[key]) for r in rows]) ** 2
        if stack is None:
            t, stack = tt, [aa]
        elif len(tt) == len(t):
            stack.append(aa)
    return t, np.mean(np.array(stack), axis=0), len(stack)


def fit_growth(t, v, t0, t1):
    sel = (t >= t0) & (t <= t1) & np.isfinite(v) & (v > 0)
    if sel.sum() < 5:
        return None
    tt, yy = t[sel], np.log(v[sel])
    p = np.polyfit(tt, yy, 1)
    pred = np.polyval(p, tt)
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    se = math.sqrt(ss_res / max(len(tt) - 2, 1) /
                   max(np.sum((tt - tt.mean()) ** 2), 1e-30))
    return dict(slope=0.5 * float(p[0]), se=0.5 * se, r2=r2, n=int(sel.sum()),
                v0=float(v[sel][0]), v1=float(v[sel][-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, required=True)
    ap.add_argument("--pert", nargs="+", required=True)
    ap.add_argument("--base", nargs="*", default=[])
    ap.add_argument("--t0", type=float, default=5.0)
    ap.add_argument("--t1", type=float, default=60.0)
    a = ap.parse_args()
    t, vp, np_ = load_series(a.pert, a.mode)
    fp = fit_growth(t, vp, a.t0, a.t1)
    print("mode m=%d  perturbed seeds=%d" % (a.mode, np_))
    if fp:
        print("   <A^2>: %.3e -> %.3e   omega=%.4e +/- %.2e  R2=%.4f  n=%d"
              % (fp["v0"], fp["v1"], fp["slope"], fp["se"], fp["r2"], fp["n"]))
    if a.base:
        tb, vb, nb = load_series(a.base, a.mode)
        fb = fit_growth(tb, vb, a.t0, a.t1)
        print("   baseline seeds=%d" % nb)
        if fb:
            print("   <A^2>_base: %.3e -> %.3e   omega_base=%.4e +/- %.2e  R2=%.4f"
                  % (fb["v0"], fb["v1"], fb["slope"], fb["se"], fb["r2"]))
            print("   omega_pert - omega_base = %.4e" % (fp["slope"] - fb["slope"]))


if __name__ == "__main__":
    main()
