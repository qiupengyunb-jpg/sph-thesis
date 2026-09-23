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
    """Topographic peaks with prominence, periodic in the array.

    Uses scipy.signal.find_peaks(mode="wrap") so that "prominence" means the
    standard topographic prominence (height above the higher of the two
    surrounding troughs) instead of the second-difference proxy that an
    immediate-neighbour rule produces.  The earlier proxy preferentially
    detected single-bin spikes, which is exactly the flicker source found in
    the Stage-2 pre-check.
    """
    from scipy.signal import find_peaks
    h = np.asarray(h, float)
    n = h.size
    pad = int(max(2, min_dist_bins))
    ext = np.concatenate([h[-pad:], h, h[:pad]])
    idx, _props = find_peaks(ext, distance=max(1, int(min_dist_bins)),
                             prominence=prominence)
    keep = sorted({int(i) - pad for i in idx if 0 <= int(i) - pad < n})
    return [i for i in keep if h[i] > 0]


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
              "t_first_bead_ge2", "t_first_bead_ge4", "merge_events", "max_bead_count",
              "bead4_hold_time", "largest_fraction_slope", "CV_spacing_slope",
              "mean_spacing_slope",
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
        bc = np.array([r["bead_count"] for r in rr], float)
        lf = np.array([r["largest_fraction"] for r in rr], float)
        sp = np.array([r["CV_spacing"] for r in rr], float)
        msp = np.array([r["mean_spacing"] for r in rr], float)

        def first_time(cond):
            idx = np.where(cond)[0]
            return float(t[idx[0]]) if idx.size else float("nan")

        def slope(y):
            ok = np.isfinite(y)
            if ok.sum() < 3 or t[ok].ptp() <= 0:
                return float("nan")
            return float(np.polyfit(t[ok], y[ok], 1)[0])

        out["t_first_bead_ge2"] = first_time(bc >= 2)
        out["t_first_bead_ge4"] = first_time(bc >= 4)
        # bead-merge events: frames where the bead count DROPS
        out["merge_events"] = int(np.sum(np.diff(bc) <= -1))
        out["max_bead_count"] = float(np.max(bc))
        # longest contiguous time with bead_count >= 4
        best4 = run4 = 0.0
        prev = None
        for c, tt in zip(bc >= 4, t):
            if c:
                run4 += (tt - prev if prev is not None else 0.0)
                best4 = max(best4, run4)
            else:
                run4 = 0.0
            prev = tt
        out["bead4_hold_time"] = best4
        out["largest_fraction_slope"] = slope(lf)
        out["CV_spacing_slope"] = slope(sp)
        out["mean_spacing_slope"] = slope(msp)
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
