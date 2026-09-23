#!/usr/bin/env python3
"""Stage 8B: does Method D expose a separable DYNAMIC linear window?

Frozen analysis (no tuning allowed): Method D from Stage 8A with its frozen
parameters (window = 3 sigma, softness lambda = 0.30 sigma, 2-bin smoothing),
Fourier amplitudes A_m for m = 1..8 by direct projection.

For every target mode the injected case is compared with the SAME mode of the
A = 0 control trajectory (never with an average over modes):

    SNR_m(t)      = A_m^inj(t) / A_m^ctrl(t)
    linear window : contiguous frames with SNR_m > 3, R^2 > 0.8 on ln A_m,
                    at least MIN_POINTS points, and not yet saturated.

Usage:
  python m18_dynamic_window.py --ctrl <D0 dir> --inj m2=<D2 dir> m3=<D3 dir> m4=<D4 dir> \
      --R0 5 --Lz 160 --out <dir>
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "m14"))
sys.path.insert(0, os.path.join(HERE, "..", "m17"))
import m14_axisym_modes as M  # noqa: E402
import m17_recon as R  # noqa: E402

MMAX = 8
SNR_MIN = 3.0
R2_MIN = 0.80
MIN_POINTS = 8


def series(run_dir, R0, Lz, nbins):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    out = []
    for f, tf in zip(files, times):
        _t, z, r = M.read_vtp(f)
        h = R.method_D(z, r, R0, Lz, nbins)
        A = M.fourier_project(h - np.nanmean(h), Lz, MMAX)
        rec = dict(time=float(tf), mean_h=float(np.nanmean(h)))
        for m in range(1, MMAX + 1):
            rec["A%d" % m] = A[m]
        out.append(rec)
    return out


def find_window(t, a_inj, a_ctrl):
    """Fixed algorithm shared by every mode: contiguous frames with SNR > 3,
    then the longest run; inside it maximise R^2 of ln A_inj vs t with >=8 pts."""
    snr = a_inj / np.maximum(a_ctrl, 1e-12)
    ok = snr > SNR_MIN
    idx = np.where(ok)[0]
    if idx.size < MIN_POINTS:
        return None, snr
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    best = None
    for run in runs:
        if len(run) < MIN_POINTS:
            continue
        for i in range(0, len(run) - MIN_POINTS + 1):
            for j in range(len(run) - 1, i + MIN_POINTS - 2, -1):
                sel = run[i:j + 1]
                tt, yy = t[sel], np.log(a_inj[sel])
                p = np.polyfit(tt, yy, 1)
                pred = np.polyval(p, tt)
                ss = float(np.sum((yy - pred) ** 2))
                st = float(np.sum((yy - yy.mean()) ** 2))
                r2 = 1.0 - ss / st if st > 0 else 0.0
                se = float(np.sqrt(ss / max(len(tt) - 2, 1) /
                                   max(np.sum((tt - tt.mean()) ** 2), 1e-30)))
                if best is None or r2 > best["r2"] + 1e-9:
                    best = dict(t_start=float(tt[0]), t_end=float(tt[-1]),
                                n=int(len(sel)), omega=float(p[0]), r2=r2, se=se,
                                snr_med=float(np.median(snr[sel])))
    return best, snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--inj", nargs="+", required=True, help="label=dir (label = m2/m3/m4)")
    ap.add_argument("--R0", type=float, default=5.0)
    ap.add_argument("--Lz", type=float, default=160.0)
    ap.add_argument("--nbins", type=int, default=320)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ctrl = series(a.ctrl, a.R0, a.Lz, a.nbins)
    tc = np.array([x["time"] for x in ctrl])

    rows, growth = [], []
    cases = {}
    for spec in a.inj:
        label, d = spec.split("=", 1)
        cases[label] = (d, int(label[1:]))
    for label, (d, m) in cases.items():
        s = series(d, a.R0, a.Lz, a.nbins)
        t = np.array([x["time"] for x in s])
        ai = np.array([x["A%d" % m] for x in s])
        ac = np.array([x["A%d" % m] for x in ctrl])
        n = min(len(t), len(tc))
        fit, snr = find_window(t[:n], ai[:n], ac[:n])
        # control growth on the identical window
        g_ctrl = None
        if fit:
            sel = (tc[:n] >= fit["t_start"]) & (tc[:n] <= fit["t_end"])
            if sel.sum() >= 3:
                p = np.polyfit(tc[:n][sel], np.log(np.maximum(ac[:n][sel], 1e-12)), 1)
                yy = np.log(np.maximum(ac[:n][sel], 1e-12))
                pred = np.polyval(p, tc[:n][sel])
                st = float(np.sum((yy - yy.mean()) ** 2))
                r2c = 1.0 - float(np.sum((yy - pred) ** 2)) / st if st > 0 else 0.0
                g_ctrl = dict(omega=float(p[0]), r2=r2c, n=int(sel.sum()))
        for k in range(n):
            rows.append(dict(case=label, mode=m, time=float(t[k]),
                             A_inj=float(ai[k]) if k < len(ai) else np.nan,
                             A_ctrl=float(ac[k]), SNR=float(snr[k]) if k < len(snr) else np.nan))
        rec = dict(mode=m, case=label,
                   SNR_median=float(np.median(snr)), SNR_max=float(np.max(snr)),
                   frac_SNR_gt3=float(np.mean(snr > SNR_MIN)),
                   A0_inj=float(ai[0]), A0_ctrl=float(ac[0]))
        if fit:
            rec.update(t_start=fit["t_start"], t_end=fit["t_end"], n=fit["n"],
                       omega_inj=fit["omega"], r2_inj=fit["r2"], se_inj=fit["se"],
                       snr_in_window=fit["snr_med"])
            if g_ctrl:
                rec.update(omega_ctrl=g_ctrl["omega"], r2_ctrl=g_ctrl["r2"],
                           delta_omega=fit["omega"] - g_ctrl["omega"])
        growth.append(rec)
    with open(os.path.join(a.out, "stage8B_mode_amplitudes.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(a.out, "stage8B_growth_comparison.csv"), "w", newline="",
              encoding="utf-8") as fh:
        keys = sorted({k for r in growth for k in r})
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in growth:
            w.writerow(r)
    for r in growth:
        print("m=%d SNR med=%.2f max=%.2f frac>3=%.2f | %s" %
              (r["mode"], r["SNR_median"], r["SNR_max"], r["frac_SNR_gt3"],
               ("window t=[%.1f,%.1f] n=%d omega_inj=%.4f R2=%.3f snr_win=%.2f "
                "omega_ctrl=%s d_omega=%s" %
                (r.get("t_start", np.nan), r.get("t_end", np.nan), r.get("n", 0),
                 r.get("omega_inj", np.nan), r.get("r2_inj", np.nan),
                 r.get("snr_in_window", np.nan),
                 "%.4f" % r["omega_ctrl"] if "omega_ctrl" in r else "n/a",
                 "%.4f" % r["delta_omega"] if "delta_omega" in r else "n/a"))
               if "omega_inj" in r else "NO WINDOW"))

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for m in sorted({r["mode"] for r in growth}):
            fig, ax = plt.subplots(figsize=(7, 3))
            sel = [r for r in rows if r["mode"] == m]
            ax.plot([r["time"] for r in sel], [r["A_inj"] for r in sel], lw=1,
                    label="injected m=%d" % m)
            ax.plot([r["time"] for r in sel], [r["A_ctrl"] for r in sel], lw=1,
                    label="A=0 control (same mode)")
            ax.set_yscale("log")
            ax.set_xlabel("t")
            ax.set_ylabel("A_m (sigma)")
            ax.set_title("A%d(t): D%d vs D0 control (Method D)" % (m, m))
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "fig_A%d_vs_ctrl.png" % m), dpi=130)
        fig, ax = plt.subplots(figsize=(7, 3))
        for m in sorted({r["mode"] for r in growth}):
            sel = [r for r in rows if r["mode"] == m]
            ax.plot([r["time"] for r in sel], [r["SNR"] for r in sel], lw=1, label="m=%d" % m)
        ax.axhline(3, ls="--", c="k", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("SNR = A_m^inj / A_m^ctrl")
        ax.set_title("dynamic SNR(t), Method D")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_dynamic_SNR.png"), dpi=130)
    except Exception as exc:
        print("plot skipped: %s" % exc)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
