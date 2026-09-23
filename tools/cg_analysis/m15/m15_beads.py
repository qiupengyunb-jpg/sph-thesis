#!/usr/bin/env python3
"""Stage 1: automatic multi-bead / periodic-structure analyser.

For every recorded frame of one run it reconstructs the axial thickness
profile h(z) (two independent reconstructions, reused from the M1.4 tool),
finds beads as prominent local maxima, and emits the Stage 1 morphology
metrics:

    bead_count, mean_spacing, CV_spacing, largest_fraction,
    dominant_lambda, mean_h, min_h (residual wet film / neck),
    modulation amplitude, spectrum

and the F0-F5 morphology class of that frame.  A run-level summary then decides
whether the run satisfies the PERIODIC-STRUCTURE working criterion, which is
declared once below and never tuned per case.

Usage: python m15_beads.py <run_dir> --R0 10 --Lz 80 [--t-lo 40 --t-hi 100]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "m14"))
import m14_axisym_modes as M  # noqa: E402

# ---------------------------------------------------------------- criterion --
# Declared ONCE.  A run is a "periodic structure" (F3) only if, over the
# declared analysis window, it holds simultaneously:
BEAD_COUNT_MIN = 4          # at least four beads
CV_SPACING_MAX = 0.35       # spacing regularity (periodic, not random droplets)
LARGEST_FRAC_MAX = 0.45     # no single bead has swallowed the film
MODULATION_MIN = 0.15       # peak-to-mean modulation A_dom / <h>
HOLD_TIME_MIN = 20.0        # sustained for at least this many time units
DOM_LAMBDA_DRIFT_MAX = 0.25  # dominant wavelength drift over the window
# F2 = multi-bead but not (yet) satisfying the F3 regularity/persistence part.


def find_peaks_periodic(h, min_dist_bins, prominence):
    """Local maxima with prominence, periodic in the array."""
    n = h.size
    idx = [i for i in range(n)
           if h[i] > h[(i - 1) % n] and h[i] >= h[(i + 1) % n] and h[i] > 0]
    kept = []
    for i in sorted(idx, key=lambda k: -h[k]):
        if all(min(abs(i - j), n - abs(i - j)) >= min_dist_bins for j in kept):
            kept.append(i)
    out = []
    for i in sorted(kept):
        # prominence relative to the ridge between this peak and its neighbours
        lo = max(0.0, min(h[(i - 1) % n], h[(i + 1) % n]))
        if h[i] - lo >= prominence:
            out.append(i)
    return out


def frame_metrics(h, z_grid, R0, Lz, mmax=12, min_dist_sigma=2.0):
    rec = {}
    n = h.size
    dz = Lz / n
    mean_h = float(np.mean(h))
    rec["mean_h"] = mean_h
    rec["min_h"] = float(np.min(h))
    rec["max_h"] = float(np.max(h))
    A = M.fourier_project(h - mean_h, Lz, mmax)
    rec["A_dom"] = max(A.values()) if A else 0.0
    rec["dominant_m"] = max(A, key=A.get) if A else 0
    rec["dominant_lambda"] = Lz / rec["dominant_m"] if rec["dominant_m"] else Lz
    rec["spectrum"] = ";".join("%d:%.4f" % (m, A[m]) for m in sorted(A))

    amp = float(np.max(h) - np.min(h))
    prom = max(0.02, 0.12 * amp)
    peaks = find_peaks_periodic(h, max(1, int(min_dist_sigma / dz)), prom)
    rec["bead_count"] = len(peaks)
    if len(peaks) >= 1:
        rec["bead_z"] = ";".join("%.3f" % (zi * dz) for zi in peaks)
        positions = np.array([zi * dz for zi in peaks])
        if len(peaks) >= 2:
            sp = np.diff(np.concatenate([positions, [positions[0] + Lz]]))
            rec["mean_spacing"] = float(np.mean(sp))
            rec["CV_spacing"] = float(np.std(sp) / np.mean(sp)) if np.mean(sp) else np.nan
        else:
            rec["mean_spacing"] = np.nan
            rec["CV_spacing"] = np.nan
        excess = np.maximum(h[peaks] - mean_h, 0.0)
        rec["largest_fraction"] = float(np.max(excess) / np.sum(excess)) if np.sum(excess) > 0 else 1.0
    else:
        rec["bead_z"] = ""
        rec["mean_spacing"] = np.nan
        rec["CV_spacing"] = np.nan
        rec["largest_fraction"] = 1.0
    rec["modulation"] = rec["A_dom"] / mean_h if mean_h > 0 else np.nan
    rec["class"] = classify(rec)
    return rec


def classify(r):
    n = r["bead_count"]
    mod = r["modulation"]
    cv = r["CV_spacing"]
    lf = r["largest_fraction"]
    if mod < 0.05 and n <= 1:
        return "F0"
    if n <= 1:
        return "F5" if mod >= 0.35 else "F1"
    if n == 2 and mod >= 0.30:
        return "F4"
    if n >= 4 and np.isfinite(cv) and cv <= CV_SPACING_MAX and lf <= LARGEST_FRAC_MAX \
            and mod >= MODULATION_MIN:
        return "F3"
    if n >= 3:
        return "F2"
    return "F1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=80.0)
    ap.add_argument("--nbins", type=int, default=0)
    ap.add_argument("--mmax", type=int, default=12)
    ap.add_argument("--t-lo", type=float, default=0.0)
    ap.add_argument("--t-hi", type=float, default=1e9)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nbins = a.nbins or max(48, int(round(a.Lz * 2.0)))
    files = M.frame_list(a.run_dir)
    times = M.frame_times(a.run_dir, files)
    rows = []
    for f, tf in zip(files, times):
        if tf < a.t_lo or tf > a.t_hi:
            continue
        _t, z, r = M.read_vtp(f)
        for tag, hfun in (("A", M.h_profile_slabmax), ("B", M.h_profile_outerlayer)):
            h = hfun(z, r, a.R0, a.Lz, nbins)
            if np.all(np.isnan(h)):
                continue
            rec = frame_metrics(h, None, a.R0, a.Lz, a.mmax)
            rec.update(time=float(tf), method=tag, n=int(z.size),
                       file=os.path.basename(f))
            rows.append(rec)
    if not rows:
        raise SystemExit("no frames in the requested window")
    out = a.out or os.path.join(a.run_dir, "m15_beads.csv")
    order = ["time", "method", "file", "n", "class", "bead_count", "mean_spacing",
             "CV_spacing", "largest_fraction", "dominant_m", "dominant_lambda",
             "A_dom", "modulation", "mean_h", "min_h", "max_h", "bead_z", "spectrum"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=order, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summ = summarize(rows, a)
    sfile = os.path.join(os.path.dirname(out), "m15_summary.csv")
    with open(sfile, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys()))
        w.writeheader()
        w.writerow(summ)
    for k in ("method", "n_frames", "t_window", "modal_class", "class_fractions",
              "median_bead_count", "median_CV_spacing", "median_largest_fraction",
              "median_modulation", "median_dominant_lambda", "lambda_drift",
              "max_F3_hold_time", "periodic_structure"):
        print("%-24s %s" % (k, summ[k]))
    print("per-frame -> %s\nsummary -> %s" % (out, sfile))


def summarize(rows, a):
    out = {}
    for tag in ("A", "B"):
        rr = [r for r in rows if r["method"] == tag]
        if not rr:
            continue
        cls = [r["class"] for r in rr]
        t = np.array([r["time"] for r in rr])
        out["method"] = tag
        out["n_frames"] = len(rr)
        out["t_window"] = "%.1f..%.1f" % (t.min(), t.max())
        vals, cnts = np.unique(cls, return_counts=True)
        out["class_fractions"] = ";".join("%s=%.2f" % (v, c / len(cls))
                                          for v, c in zip(vals, cnts))
        order = ["F0", "F1", "F2", "F3", "F4", "F5"]
        modal = max(order, key=lambda c: cls.count(c))
        out["modal_class"] = modal
        out["median_bead_count"] = float(np.median([r["bead_count"] for r in rr]))
        out["median_CV_spacing"] = float(np.nanmedian([r["CV_spacing"] for r in rr]))
        out["median_largest_fraction"] = float(np.nanmedian([r["largest_fraction"] for r in rr]))
        out["median_modulation"] = float(np.nanmedian([r["modulation"] for r in rr]))
        out["median_dominant_lambda"] = float(np.nanmedian([r["dominant_lambda"] for r in rr]))
        lam = np.array([r["dominant_lambda"] for r in rr], float)
        out["lambda_drift"] = float((lam.max() - lam.min()) / np.median(lam)) if np.median(lam) else np.nan
        # longest contiguous F3 run in time
        best = 0.0
        run = 0.0
        prev = None
        for c, tt in zip(cls, t):
            if c == "F3":
                run = run + (tt - prev if prev is not None else 0.0)
                best = max(best, run)
            else:
                run = 0.0
            prev = tt
        out["max_F3_hold_time"] = best
        frac_F3 = cls.count("F3") / len(cls)
        out["periodic_structure"] = bool(frac_F3 >= 0.5 and best >= HOLD_TIME_MIN)
        break
    return out


if __name__ == "__main__":
    main()
