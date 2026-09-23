#!/usr/bin/env python3
"""Stage 7: packing -> initial-relaxation / Fourier-background pre-check.

For one T=0 (noise-off) run of a cylindrical film this measures

  * the per-mode Fourier background A_m (m = 1..12) of h(z) after the initial
    deterministic relaxation  -> "mode noise floor";
  * the one-off relaxation of the mean thickness (amplitude and time);
  * the outer-surface ripple after relaxation;
  * the APPARENT (pseudo) growth rate of A_m during the relaxation window,
    which is the quantity that must shrink if a cleaner initial lattice really
    restores a measurable linear window.

It reuses the M1.4 h(z) reconstruction (no detector thresholds are involved,
so nothing here depends on the m15 peak criteria).

Usage: python m16_packing_noise.py <run_dir> --R0 5 --Lz 160 --label phi0.63
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m14"))
import m14_axisym_modes as M  # noqa: E402

SMOOTH_BINS = 2.0     # frozen reconstruction smoothing (same as the m15 frozen config)
MMAX = 12


def analyse(run_dir, R0, Lz, t_lo, t_hi, nbins):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    zc = (np.arange(nbins) + 0.5) * (Lz / nbins)
    rows = []
    for f, tf in zip(files, times):
        if tf < t_lo or tf > t_hi:
            continue
        _t, z, r = M.read_vtp(f)
        h = M.h_profile_slabmax(z, r, R0, Lz, nbins)
        if np.all(np.isnan(h)):
            continue
        h = M._fill_and_smooth(h, nbins, SMOOTH_BINS)
        A = M.fourier_project(h - np.mean(h), Lz, MMAX)
        rec = dict(time=float(tf), mean_h=float(np.mean(h)),
                   min_h=float(np.min(h)), max_h=float(np.max(h)),
                   ripple_std=float(np.std(h)), ripple_ptp=float(np.max(h) - np.min(h)))
        for m in range(1, MMAX + 1):
            rec["A%d" % m] = A[m]
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=160.0)
    ap.add_argument("--nbins", type=int, default=0)
    ap.add_argument("--t-lo", type=float, default=0.0)
    ap.add_argument("--t-hi", type=float, default=1e9)
    ap.add_argument("--label", default="")
    ap.add_argument("--relax-window", type=float, default=12.0)
    a = ap.parse_args()
    nbins = a.nbins or max(64, int(round(a.Lz * 2.0)))
    rows = analyse(a.run_dir, a.R0, a.Lz, a.t_lo, a.t_hi, nbins)
    if len(rows) < 6:
        raise SystemExit("need >= 6 frames")
    t = np.array([r["time"] for r in rows])
    mh = np.array([r["mean_h"] for r in rows])
    rp = np.array([r["ripple_std"] for r in rows])
    A = {m: np.array([r["A%d" % m] for r in rows]) for m in range(1, MMAX + 1)}

    # relaxation: end value, one-off drop, time to settle
    h_end = float(np.median(mh[t > 0.7 * t[-1]]))
    relax_drop = float(mh[0] - h_end)
    settled = np.where(np.abs(mh - h_end) < 0.02)[0]
    relax_time = float(t[settled[0]]) if settled.size else float("nan")

    # post-relaxation noise floor per mode
    post = t >= max(relax_time, 0.0) + a.relax_window
    if post.sum() < 4:
        post = t > 0.5 * t[-1]
    med = {m: float(np.median(A[m][post])) for m in range(1, MMAX + 1)}
    floor_mean = float(np.mean(list(med.values())))
    floor_max = float(np.max(list(med.values())))
    dom_m = int(max(med, key=med.get))

    # apparent (pseudo) growth during the relaxation window
    win = (t <= min(relax_time if np.isfinite(relax_time) else 10.0, 10.0)) & (t >= 0.0)
    if win.sum() < 4:
        win = t <= 0.25 * t[-1]
    pseudo = {}
    for m in range(1, MMAX + 1):
        y = A[m][win]
        ok = y > 0
        if ok.sum() >= 3 and np.ptp(t[win][ok]) > 0:
            pseudo[m] = float(np.polyfit(t[win][ok], np.log(y[ok]), 1)[0])
    pseudo_max = float(max(pseudo.values())) if pseudo else float("nan")
    pseudo_arg = int(max(pseudo, key=pseudo.get)) if pseudo else 0

    out_csv = os.path.join(a.run_dir, "m16_frames.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summ = dict(label=a.label, t_start=float(t[0]), t_end=float(t[-1]), n_frames=len(rows),
                mean_h_start=float(mh[0]), mean_h_end=h_end, relax_drop=relax_drop,
                relax_time=relax_time,
                ripple_after=float(np.median(rp[post])),
                noise_floor_mean=floor_mean, noise_floor_max=floor_max,
                noise_floor_dominant_m=dom_m,
                pseudo_omega_max=pseudo_max, pseudo_omega_argm=pseudo_arg,
                noise_floor_M1_12=";".join("%d:%.4f" % (m, med[m]) for m in range(1, MMAX + 1)))
    s_csv = os.path.join(a.run_dir, "m16_summary.csv")
    with open(s_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys()))
        w.writeheader()
        w.writerow(summ)
    for k, v in summ.items():
        print("%-22s %s" % (k, v))
    print("-> %s" % s_csv)


if __name__ == "__main__":
    main()
