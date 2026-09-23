#!/usr/bin/env python3
"""Stage 13: non-equilibrium bead/coarsening time series for one case.

Metrics per frame (frozen tools only: m14 reconstruction, m17 Method D,
m15 frozen peak rule):
  N_bead, bead density n_b = N/Lz, mean/median spacing, CV spacing,
  largest_fraction, modulation, dominant_lambda, |H_1..4|^2, P_low,
  L_bead = Lz/N (N>=2), L_spec = 2 pi / <k> with <k> = sum k P(k)/sum P(k)
Event times: t_mod, t_bead2, t_bead4, t_Nmax, t_coarsen, t_F2_enter/exit,
T_F2, t_dominant (largest_fraction > 0.5).
Dynamics class D0..D3 from coarsening rate and morphology history.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("../m14", "../m17", "../m15"):
    sys.path.insert(0, os.path.join(HERE, sub))
import m14_axisym_modes as M  # noqa: E402
import m17_recon as R  # noqa: E402
import m15_beads as B  # noqa: E402

MOD_THRESH = 0.15      # frozen: modulation above the Method-D noise floor band
DOM_FRAC = 0.50        # frozen: largest_fraction dominance threshold
MIN_HOLD = 3           # frames a condition must hold to count as "stable"


def series(run_dir, R0, Lz, nbins):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    rows = []
    for f, tf in zip(files, times):
        _t, z, r = M.read_vtp(f)
        h = R.method_D(z, r, R0, Lz, nbins)
        hm = h - np.nanmean(h)
        amp = float(np.nanmax(h) - np.nanmin(h))
        peaks = B.find_peaks_periodic(h, max(1, int(2.0 / (Lz / nbins))),
                                      max(B.PROM_ABS_SIGMA, 0.25 * amp))
        n = len(peaks)
        rec = dict(time=float(tf), N_bead=n, bead_density=n / Lz,
                   mean_h=float(np.nanmean(h)))
        if n >= 2:
            zc = np.sort(np.array(peaks) * (Lz / nbins))
            sp = np.diff(np.concatenate([zc, [zc[0] + Lz]]))
            rec["mean_spacing"] = float(np.mean(sp))
            rec["median_spacing"] = float(np.median(sp))
            rec["CV_spacing"] = float(np.std(sp) / np.mean(sp))
            rec["L_bead"] = Lz / n
        else:
            rec["mean_spacing"] = np.nan
            rec["median_spacing"] = np.nan
            rec["CV_spacing"] = np.nan
            rec["L_bead"] = np.nan
        exc = np.maximum(h[peaks] - np.nanmean(h), 0.0) if n else np.array([0.0])
        rec["largest_fraction"] = float(exc.max() / exc.sum()) if exc.sum() > 0 else 1.0
        A = M.fourier_project(hm, Lz, 8)
        denom = sum(A[m] ** 2 for m in range(5, 9))
        rec["modulation"] = max(A[m] for m in range(1, 5)) / rec["mean_h"] if rec["mean_h"] else np.nan
        rec["dominant_lambda"] = Lz / max(range(1, 9), key=lambda m: A[m])
        for m in range(1, 5):
            rec["P%d" % m] = A[m] ** 2
        rec["P_low"] = sum(A[m] ** 2 for m in range(1, 5))
        num = sum((2 * math.pi * m / Lz) * A[m] ** 2 for m in range(1, 9))
        rec["L_spec"] = 2 * math.pi / (num / denom_total(A)) if denom_total(A) > 0 else np.nan
        rows.append(rec)
    return rows


def denom_total(A):
    return sum(A[m] ** 2 for m in range(1, 9))


def first_stable(rows, key, thresh, above=True, hold=MIN_HOLD):
    cnt = 0
    for r in rows:
        v = r.get(key, np.nan)
        ok = (v > thresh) if above else (v < thresh)
        cnt = cnt + 1 if (ok and np.isfinite(v)) else 0
        if cnt >= hold:
            return r["time"]
    return float("nan")


def classify(rows):
    n = np.array([r["N_bead"] for r in rows], float)
    lf = np.array([r["largest_fraction"] for r in rows], float)
    t = np.array([r["time"] for r in rows])
    if n.max() < 2:
        return "D0", "never reaches 2 beads"
    i_nmax = int(np.argmax(n))
    post = slice(i_nmax, None)
    dt = t[-1] - t[i_nmax]
    slope_lf = (np.polyfit(t[post], lf[post], 1)[0] if dt > 0 else np.nan)
    rate = (n[i_nmax] - n[-1]) / dt if dt > 0 else np.nan
    if rate <= 0.002 and slope_lf <= 2e-4:
        return "D1", "multi-bead long-lived (rate=%.4f, dLF=%.2e)" % (rate, slope_lf)
    if slope_lf > 8e-4 or (np.mean(lf[t > 0.7 * t[-1]]) > 0.75):
        return "D3", "rapid dominance / collapse (dLF=%.2e)" % slope_lf
    return "D2", "active coarsening (rate=%.4f, dLF=%.2e)" % (rate, slope_lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=160.0)
    ap.add_argument("--nbins", type=int, default=320)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rows = series(a.run_dir, a.R0, a.Lz, a.nbins)
    t = np.array([r["time"] for r in rows])
    n = np.array([r["N_bead"] for r in rows], float)
    i_nmax = int(np.argmax(n))
    ev = dict(label=a.label or os.path.basename(a.run_dir.rstrip("\\/")),
              t_end=float(t[-1]), n_frames=len(rows),
              t_mod=first_stable(rows, "modulation", MOD_THRESH),
              t_bead2=first_stable(rows, "N_bead", 1.5),
              t_bead4=first_stable(rows, "N_bead", 3.5),
              N_max=float(n.max()), t_Nmax=float(t[i_nmax]),
              t_coarsen=float(t[i_nmax]) if i_nmax < len(t) - MIN_HOLD else np.nan,
              t_dominant=first_stable(rows, "largest_fraction", DOM_FRAC),
              bead_density_max=float(n.max() / a.Lz),
              P_low_end=float(rows[-1]["P_low"]),
              P_low_mean=float(np.mean([r["P_low"] for r in rows[::max(1, len(rows)//50)]])))
    # F2 lifetime (contiguous F2 = 3<=N<=6 with CV<=0.5)
    f2 = [r for r in rows if 3 <= r["N_bead"] <= 6 and np.isfinite(r["CV_spacing"])
          and r["CV_spacing"] <= 0.5]
    ev["T_F2"] = float(len(f2) / max(len(rows), 1) * (t[-1] - t[0]))
    cls, why = classify(rows)
    ev["dynamics_class"] = cls
    ev["dynamics_reason"] = why
    # coarsening rate over [t_Nmax, t_end]
    dt = t[-1] - t[i_nmax]
    ev["coarsen_rate_N"] = float((n[i_nmax] - n[-1]) / dt) if dt > 0 else np.nan
    ls = np.array([r["L_spec"] for r in rows], float)
    ev["L_spec_slope"] = float(np.polyfit(t, ls, 1)[0]) if np.all(np.isfinite(ls)) else np.nan
    out = os.path.join(a.run_dir, "m23_timeseries.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(a.run_dir, "m23_events.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ev.keys()))
        w.writeheader()
        w.writerow(ev)
    print("%-22s t_bead2=%6.1f t_bead4=%6.1f Nmax=%.0f t_coarsen=%6.1f T_F2=%6.1f "
          "rate=%.4f cls=%s" % (ev["label"], ev["t_bead2"], ev["t_bead4"], ev["N_max"],
                                ev["t_coarsen"], ev["T_F2"], ev["coarsen_rate_N"], cls))


if __name__ == "__main__":
    main()
