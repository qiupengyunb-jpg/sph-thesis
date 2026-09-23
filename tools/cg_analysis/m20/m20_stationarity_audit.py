#!/usr/bin/env python3
"""Stage 10R: statistical-stationarity audit of the T=1 A=0 background.

Analysis only, on the existing Stage-10 probes.  It replaces the Stage-10
extrema criteria with block statistics of

  * T_kin                (from selfcheck.csv)
  * <h>(t)               (Method D)
  * complex Fourier modes H_m = Re_m + i Im_m, m = 1..4, and their power |H_m|^2
  * frozen-detector morphology (bead_count, modulation, largest fraction)

and decides whether any >= MIN_WINDOW time-unit window is statistically
stationary (the criterion structure is pre-declared in the report).
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m14"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m17"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m15"))
import m14_axisym_modes as M  # noqa: E402
import m17_recon as R  # noqa: E402
import m15_beads as B  # noqa: E402

BLOCK = 25.0          # time units per statistical block
MIN_WINDOW = 50.0     # minimum window for a stationarity verdict
NDOF = 532            # N * dim


def load(run_dir, R0=5.0, Lz=160.0, nbins=320):
    sc = list(csv.DictReader(open(os.path.join(run_dir, "selfcheck.csv"),
                                  newline="", encoding="utf-8")))
    tk = {float(r["time"]): float(r["kinetic_temperature"]) for r in sc}
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    rows = []
    for f, tt in zip(files, times):
        _t, z, r = M.read_vtp(f)
        h = R.method_D(z, r, R0, Lz, nbins)
        hm = h - np.nanmean(h)
        rec = dict(time=tt, mean_h=float(np.nanmean(h)),
                   tkin=tk.get(tt, np.nan))
        zc = (np.arange(nbins) + 0.5) * (Lz / nbins)
        amp = float(np.nanmax(h) - np.nanmin(h))
        peaks = B.find_peaks_periodic(h, max(1, int(2.0 / (Lz / nbins))),
                                      max(B.PROM_ABS_SIGMA, 0.25 * amp))
        rec["bead_count"] = len(peaks)
        A = M.fourier_project(hm, Lz, 4)
        rec["modulation"] = A[1] / rec["mean_h"] if rec["mean_h"] else np.nan
        exc = np.maximum(h[peaks] - np.nanmean(h), 0.0) if len(peaks) else np.array([0.0])
        rec["largest_fraction"] = float(exc.max() / exc.sum()) if exc.sum() > 0 else 1.0
        for m in range(1, 5):
            c = np.cos(2 * np.pi * m * zc / Lz)
            s = np.sin(2 * np.pi * m * zc / Lz)
            re = 2.0 / nbins * float(np.sum(hm * c))
            im = 2.0 / nbins * float(np.sum(hm * s))
            rec["Re%d" % m] = re
            rec["Im%d" % m] = im
            rec["P%d" % m] = re * re + im * im
        rows.append(rec)
    return rows


def autocorr_time(x, dt):
    x = x - np.mean(x)
    n = len(x)
    if n < 8 or np.allclose(x, 0):
        return np.nan
    ac = np.correlate(x, x, "full")[n - 1:] / (np.var(x) * n)
    for k in range(1, n):
        if ac[k] < 1.0 / np.e:
            return k * dt
    return np.nan


def slope_sig(t, y):
    """least-squares slope with its standard error"""
    if len(t) < 5:
        return np.nan, np.nan
    p, cov = np.polyfit(t, y, 1, cov=True)
    return float(p[0]), float(np.sqrt(cov[0, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.run_dir)
    t = np.array([r["time"] for r in rows])
    dt = float(np.median(np.diff(t)))
    print("frames=%d  t=[%.1f,%.1f]  dt_out=%.3f" % (len(rows), t[0], t[-1], dt))

    print("\n[theory] T_kin estimator with ndof=%d: sigma(T_kin)/<T_kin> = sqrt(2/%d) = %.3f"
          % (NDOF, NDOF, np.sqrt(2.0 / NDOF)))
    tk = np.array([r["tkin"] for r in rows])
    print("[measured] T_kin mean=%.4f  std=%.4f (%.2f%%)  min=%.3f max=%.3f"
          % (np.nanmean(tk), np.nanstd(tk), 100 * np.nanstd(tk) / np.nanmean(tk),
             np.nanmin(tk), np.nanmax(tk)))
    print("           autocorrelation time = %.2f t" % autocorr_time(tk, dt))
    print("           Stage-10 extrema rule (0.9..1.1 over a 20 t window) would need"
          " |dev| < 10%% = %.2f sigma of the *instantaneous* estimator"
          % (0.10 / (np.nanstd(tk) / np.nanmean(tk))))

    print("\n[blocks of %.0f t]" % BLOCK)
    print(" block        T_kin(mean+-std)      <h> slope(sig)   |H1|^2  |H2|^2  |H3|^2  |H4|^2   beads")
    nb = int((t[-1] - t[0]) // BLOCK)
    blocks = []
    for i in range(nb):
        s = (t >= t[0] + i * BLOCK) & (t < t[0] + (i + 1) * BLOCK)
        if s.sum() < 5:
            continue
        tkb = tk[s]
        sl, se = slope_sig(t[s], np.array([r["mean_h"] for r in rows])[s])
        pw = [np.mean([r["P%d" % m] for r in np.array(rows, dtype=object)[s]])
              for m in range(1, 5)]
        bc = np.mean([r["bead_count"] for r in np.array(rows, dtype=object)[s]])
        blocks.append(dict(t0=float(t[s][0]), tkin=float(np.nanmean(tkb)),
                           tstd=float(np.nanstd(tkb)), h_slope=sl, h_se=se,
                           p1=pw[0], p2=pw[1], p3=pw[2], p4=pw[3], bead=bc))
        print(" [%5.1f,%5.1f]  %.3f +- %.3f      %+.2e (%.1e)  %.4f %.4f %.4f %.4f   %.1f"
              % (blocks[-1]["t0"], blocks[-1]["t0"] + BLOCK, blocks[-1]["tkin"],
                 blocks[-1]["tstd"], sl, se, pw[0], pw[1], pw[2], pw[3], bc))

    with open(os.path.join(a.out, "stage10R_blocks.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(blocks[0].keys()))
        w.writeheader()
        for b in blocks:
            w.writerow(b)

    # spectral power trend over the second half (statistical stationarity)
    print("\n[spectral power trend over t >= %.0f]" % (0.4 * t[-1]))
    s = t >= 0.4 * t[-1]
    for m in range(1, 5):
        y = np.array([r["P%d" % m] for r in rows])[s]
        sl, se = slope_sig(t[s], y)
        rel = sl * (t[s][-1] - t[s][0]) / np.mean(y) if np.mean(y) > 0 else np.nan
        print("  |H%d|^2: mean=%.4f  slope=%+.2e +- %.1e  ->  relative change over %.0f t = %+.1f%%  %s"
              % (m, np.mean(y), sl, se, t[s][-1] - t[s][0], 100 * rel,
                 "SIGNIFICANT" if abs(sl) > 2 * se else "not significant"))

    print("\n[morphology trend over t >= %.0f]" % (0.4 * t[-1]))
    for k in ("bead_count", "modulation", "largest_fraction"):
        y = np.array([r[k] for r in rows])[s]
        sl, se = slope_sig(t[s], y)
        print("  %-16s mean=%.3f  slope=%+.2e +- %.1e  %s" %
              (k, np.mean(y), sl, se,
               "SIGNIFICANT DRIFT" if abs(sl) > 2 * se else "no significant drift"))
    return


if __name__ == "__main__":
    main()
