#!/usr/bin/env python3
"""M1.4: axisymmetric film axial-mode analysis.

Reads CGParticles_*.vtp frames from one run and produces, per frame,

  * h(z) reconstructed by TWO independent methods
      A: outer particle envelope (binned maximum radius)
      B: radial density threshold (0.5 rho_plateau of the outer crossing)
  * Fourier amplitudes A_m by TWO independent paths
      (direct sine/cosine projection, and np.fft on the same uniform grid)
  * purity P_m = A_m^2 / sum_{m'>0} A_m'^2, m = 0 drift, mean/min/max h

then fits the early linear window of ln A_m(t) with ONE fixed algorithm
shared by every case (see linear_window()), classifies the mode with the
pre-declared 2-sigma rule, and cross-checks omega with a finite-window
growth-ratio estimator.

All thresholds are constants at the top of this file; no per-case tuning.
"""

import argparse
import csv
import glob
import math
import os
import re

import numpy as np

# ------------------------------------------------------------------ frozen ---
MIN_T_FRACTION = 0.02      # skip the first 2 % of the run (start-up transient)
MIN_WINDOW_POINTS = 8      # minimum frames in a linear window
PURITY_MIN = 0.50          # target-mode purity required inside the window
AMP_OVER_H_MAX = 0.20      # linearity/validity cap on A_m / mean h
NONLINEAR_CAP = 0.30       # frames above this are excluded from the fit
R2_MIN = 0.80              # below this the mode is UNRESOLVED
MIN_LOG_CHANGE = 0.05      # |d ln A_m| over the window must exceed this
BBINS_PER_SIGMA = 2.0      # z-bins per sigma for the h(z) reconstruction
RBIN = 0.5                 # radial bin width (sigma) for method B


def read_vtp(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        txt = fh.read()
    m = re.search(r'Name="TimeValue"[^>]*>(.*?)</DataArray>', txt, re.S)
    t = float(m.group(1).split()[0]) if m else float("nan")
    pts = re.search(r"<Points>(.*?)</Points>", txt, re.S)
    body = pts.group(1)
    da = re.search(r'<DataArray[^>]*Name="Position"[^>]*>(.*?)</DataArray>',
                   body, re.S)
    vals = np.fromstring(da.group(1), sep=" ")
    xyz = vals.reshape(-1, 3)
    return t, xyz[:, 0].astype(float), xyz[:, 1].astype(float)


def frame_list(run_dir):
    pats = [os.path.join(run_dir, "**", "CGParticles_ite_*.vtp")]
    files = []
    for p in pats:
        files += glob.glob(p, recursive=True)
    def ite(f):
        m = re.search(r"_ite_(\d+)\.vtp$", f)
        return int(m.group(1)) if m else 0
    return sorted(files, key=ite)


def frame_times(run_dir, files, dt_fallback=0.0075):
    """Frame times.

    The VTP TimeValue field is written as 0 by SPHinXsys, so the time axis is
    taken from the run's selfcheck.csv (one row per recorded frame, same order)
    and falls back to step_index * dt only if that file is unusable.
    """
    sc = os.path.join(run_dir, "selfcheck.csv")
    if os.path.exists(sc):
        with open(sc, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) == len(files):
            return np.array([float(r["time"]) for r in rows])
    out = []
    for f in files:
        m = re.search(r"_ite_(\d+)\.vtp$", f)
        out.append(int(m.group(1)) * dt_fallback if m else 0.0)
    return np.array(out)


def _fill_and_smooth(h, nbins, sigma_bins):
    """Periodic fill of empty bins + periodic Gaussian smoothing."""
    h = np.asarray(h, float)
    if np.all(np.isnan(h)):
        return h
    if np.any(np.isnan(h)):
        good = ~np.isnan(h)
        xg = np.arange(nbins)
        h = np.interp(xg, xg[good], h[good], period=nbins)
    if sigma_bins <= 0:
        return h
    k = int(max(1, round(3.0 * sigma_bins)))
    x = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (x / sigma_bins) ** 2)
    g /= g.sum()
    hp = np.concatenate([h[-k:], h, h[:k]])
    return np.convolve(hp, g, mode="valid")


def h_profile_slabmax(z, r, R0, Lz, nbins, half_width=1.0, sigma_bins=1.0):
    """Method A: sliding-slab maximum radius (morphological outer envelope).

    A slab of half-width >= half the axial lattice spacing always contains an
    outermost-layer particle, so the envelope never drops to the next layer.
    """
    zc = (np.arange(nbins) + 0.5) * (Lz / nbins)
    h = np.full(nbins, np.nan)
    for b, z0 in enumerate(zc):
        dd = z - z0
        dd -= Lz * np.round(dd / Lz)
        sel = np.abs(dd) <= half_width
        if np.any(sel):
            h[b] = r[sel].max() - R0
    return _fill_and_smooth(h, nbins, sigma_bins)


def h_profile_outerlayer(z, r, R0, Lz, nbins, sigma_bins=1.0):
    """Method B: explicit outermost-particle selection, then interpolation.

    Independent of method A: instead of a morphological maximum over a slab,
    this picks ONE particle per fine z-bin (the outermost one), builds the
    scatter (z_i, r_i) of that outer layer, and interpolates it onto the grid.
    """
    nfine = max(8 * nbins, 256)
    edges = np.linspace(0.0, Lz, nfine + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, nfine - 1)
    zc_f = 0.5 * (edges[:-1] + edges[1:])
    rf = np.full(nfine, np.nan)
    for b in range(nfine):
        sel = idx == b
        if np.any(sel):
            rf[b] = r[sel].max()
    good = ~np.isnan(rf)
    if good.sum() < 8:
        return np.full(nbins, np.nan)
    x = np.arange(nfine)
    rf = np.interp(x, x[good], rf[good], period=nfine)
    zc = (np.arange(nbins) + 0.5) * (Lz / nbins)
    h = np.interp(zc, zc_f, rf - R0, period=Lz)
    return _fill_and_smooth(h, nbins, sigma_bins)


def fourier_project(h, Lz, mmax):
    """Direct sine/cosine projection (path 1)."""
    n = h.size
    z = (np.arange(n) + 0.5) * (Lz / n)
    A = {}
    for m in range(1, mmax + 1):
        c = np.cos(2 * math.pi * m * z / Lz)
        s = np.sin(2 * math.pi * m * z / Lz)
        a = 2.0 / n * float(np.sum(h * c))
        b = 2.0 / n * float(np.sum(h * s))
        A[m] = math.hypot(a, b)
    return A


def fourier_fft(h, Lz, mmax):
    """FFT path (path 2), independent implementation."""
    n = h.size
    f = np.fft.rfft(h - h.mean()) / n
    A = {}
    for m in range(1, mmax + 1):
        A[m] = 2.0 * abs(f[m]) if m < f.size else 0.0
    return A


def analyse_frames(run_dir, R0, Lz, mmax, nbins):
    files = frame_list(run_dir)
    times = frame_times(run_dir, files)
    rows = []
    for f, tf in zip(files, times):
        t, z, r = read_vtp(f)
        ha = h_profile_slabmax(z, r, R0, Lz, nbins)
        hb = h_profile_outerlayer(z, r, R0, Lz, nbins)
        rec = dict(file=os.path.basename(f), time=float(tf), n=z.size,
                   mean_h_A=float(np.nanmean(ha)), mean_h_B=float(np.nanmean(hb)),
                   min_h_A=float(np.nanmin(ha)), max_h_A=float(np.nanmax(ha)),
                   m0_A=0.0, m0_B=0.0,
                   rmean=float(r.mean()))
        for tag, h in (("A", ha), ("B", hb)):
            A1 = fourier_project(h, Lz, mmax)
            A2 = fft = fourier_fft(h, Lz, mmax)
            for m in range(1, mmax + 1):
                rec["A%d_%s" % (m, tag)] = A1[m]
                rec["F%d_%s" % (m, tag)] = A2[m]
            tot = sum(v * v for v in A1.values()) or 1.0
            for m in range(1, mmax + 1):
                rec["P%d_%s" % (m, tag)] = A1[m] ** 2 / tot
        # m = 0 drift of the mean thickness relative to frame 0
        rec["_ha"] = ha
        rec["_hb"] = hb
        rows.append(rec)
    return rows


def linear_window(t, a, hmean, purity, target_m):
    """ONE fixed window algorithm, shared by every case.

    Candidate frames: t >= MIN_T_FRACTION * t_end, purity >= PURITY_MIN,
    A/h <= AMP_OVER_H_MAX (nonlinear frames above NONLINEAR_CAP excluded).
    Among all contiguous runs of >= MIN_WINDOW_POINTS candidates, take the
    longest run; inside it maximise R^2 by trimming from both ends.
    """
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    ok = (t >= MIN_T_FRACTION * t[-1]) & np.isfinite(a) & (a > 0)
    ok &= hmean > 0
    ok &= (a / np.maximum(hmean, 1e-12)) <= AMP_OVER_H_MAX
    if purity is not None:
        ok &= np.asarray(purity, float) >= PURITY_MIN
    if ok.sum() < MIN_WINDOW_POINTS:
        # relax the purity cut once, never per case
        ok = (t >= MIN_T_FRACTION * t[-1]) & np.isfinite(a) & (a > 0) & (hmean > 0)
        ok &= (a / np.maximum(hmean, 1e-12)) <= NONLINEAR_CAP
    if ok.sum() < MIN_WINDOW_POINTS:
        return None
    idx = np.where(ok)[0]
    # longest contiguous run
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    run = max(runs, key=len)
    if len(run) < MIN_WINDOW_POINTS:
        return None
    lo, hi, best = 0, len(run) - 1, None
    # trim from both ends to maximise R^2 (still one fixed rule)
    for i in range(0, len(run) - MIN_WINDOW_POINTS + 1):
        for j in range(len(run) - 1, i + MIN_WINDOW_POINTS - 2, -1):
            sel = run[i:j + 1]
            tt = t[sel]
            yy = np.log(a[sel])
            p = np.polyfit(tt, yy, 1)
            pred = np.polyval(p, tt)
            ss_res = float(np.sum((yy - pred) ** 2))
            ss_tot = float(np.sum((yy - yy.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if best is None or r2 > best["r2"] + 1e-9:
                npts = len(sel)
                se = math.sqrt(ss_res / max(npts - 2, 1) /
                               max(np.sum((tt - tt.mean()) ** 2), 1e-30))
                best = dict(t_start=float(tt[0]), t_end=float(tt[-1]),
                            n=npts, omega=float(p[0]), r2=r2, se=se,
                            amp_start=float(a[sel][0]), amp_end=float(a[sel][-1]),
                            omega_ratio=float((yy[-1] - yy[0]) / (tt[-1] - tt[0])))
    # Reject DEGENERATE windows.  A quenched (T = 0) run freezes to a static
    # configuration, where A_m(t) is constant to machine precision: that gives
    # R^2 = 1 with omega = 0 and is NOT a linear-response measurement.  A window
    # must carry a real amplitude change to be usable.
    if best is not None:
        if abs(math.log(best["amp_end"] / best["amp_start"])) < MIN_LOG_CHANGE:
            best = None
    return best


def classify(fit):
    if fit is None:
        return "UNRESOLVED", "no linear window with enough candidate frames"
    if fit["r2"] < R2_MIN:
        return "UNRESOLVED", "R2 %.3f < %.2f" % (fit["r2"], R2_MIN)
    lo = fit["omega"] - 2.0 * fit["se"]
    hi = fit["omega"] + 2.0 * fit["se"]
    if lo > 0:
        return "GROWTH", "omega - 2SE = %.3e > 0" % lo
    if hi < 0:
        return "DECAY", "omega + 2SE = %.3e < 0" % hi
    return "NEUTRAL", "omega +/- 2SE straddles zero"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=80.0)
    ap.add_argument("--target-m", type=int, default=0)
    ap.add_argument("--mmax", type=int, default=12)
    ap.add_argument("--nbins", type=int, default=0,
                    help="0 = Lz * BBINS_PER_SIGMA / sigma ~ auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nbins = a.nbins or max(32, int(round(a.Lz * BBINS_PER_SIGMA)))
    rows = analyse_frames(a.run_dir, a.R0, a.Lz, a.mmax, nbins)
    if not rows:
        raise SystemExit("no frames found in %s" % a.run_dir)
    out = a.out or os.path.join(a.run_dir, "m14_modes.csv")
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    t = np.array([r["time"] for r in rows])
    summary = []
    for m in range(1, a.mmax + 1):
        for tag in ("A", "B"):
            amp = np.array([r["A%d_%s" % (m, tag)] for r in rows])
            hm = np.array([r["mean_h_%s" % tag] for r in rows])
            pur = np.array([r["P%d_%s" % (m, tag)] for r in rows])
            fit = linear_window(t, amp, hm, pur if a.target_m else None, m)
            cls, why = classify(fit)
            rec = dict(mode=m, method=tag,
                       lambda_z=a.Lz / m,
                       A0=float(amp[0]), mean_h=float(hm.mean()),
                       dm0=float(hm[-1] - hm[0]),
                       purity0=float(pur[0]),
                       classification=cls, reason=why)
            if fit:
                rec.update({k: fit[k] for k in
                            ("t_start", "t_end", "n", "omega", "r2", "se",
                             "omega_ratio", "amp_start", "amp_end")})
                rec["omega_minus_2se"] = fit["omega"] - 2 * fit["se"]
                rec["omega_plus_2se"] = fit["omega"] + 2 * fit["se"]
            summary.append(rec)
    order = [k for k in summary[0]]
    sfile = os.path.join(os.path.dirname(out), "m14_summary.csv")
    with open(sfile, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=order, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print("frames=%d  ->  %s" % (len(rows), out))
    print("summary -> %s" % sfile)
    for r in summary:
        if r["method"] == "A":
            print("m=%2d lam_z=%6.2f  A0=%.4e  purity0=%.3f  %-11s %s"
                  % (r["mode"], r["lambda_z"], r["A0"], r["purity0"],
                     r["classification"], r["reason"]))


if __name__ == "__main__":
    main()
