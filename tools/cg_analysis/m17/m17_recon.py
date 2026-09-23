#!/usr/bin/env python3
"""Stage 8: interface-reconstruction resolution verification (analysis-only).

Keeps the existing reconstructions as BASELINE and adds two continuous ones:

  Method A (baseline, m14) : sliding-slab maximum radius
  Method B (baseline, m14) : outermost-particle selection + interpolation
  Method C (new)           : local KERNEL-WEIGHTED QUADRATIC FIT of r(z) over
                             the outer-layer candidates in a periodic window
  Method D (new)           : kernel-weighted SOFT-UPPER ENVELOPE, i.e. a
                             normalised exponential (soft-max) estimator

        D: r_eff(z0) = sum_i w_i r_i exp((r_i - r_max_loc)/lam)
                       / sum_i w_i exp((r_i - r_max_loc)/lam)
           with w_i a periodic Gaussian kernel in z and lam a softness scale.

Both new methods are continuous in z and never let a single outermost particle
determine h(z0) locally.  Every method is evaluated on exactly the same VTP.

Outputs in <out_dir>:
  m17_frames_<label>_<method>.csv   per-frame A_m and h statistics
  stage8_reconstruction_comparison.csv   band noise floors + recovery + SNR
  stage8_mode_resolution.csv             points per wavelength and band class
  fig_*.png                              the five required figures
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m14"))
import m14_axisym_modes as M  # noqa: E402

MMAX = 12
SMOOTH_BINS = 2.0        # same smoothing the frozen m15 configuration uses
DEFAULT_WINDOW = 3.0     # sigma; Method C/D neighbourhood half width
SOFT_LAM = 0.30          # sigma; Method D soft-max scale


def method_C(z, r, R0, Lz, nbins, window=DEFAULT_WINDOW):
    """Local kernel-weighted quadratic fit of the outer-layer radius."""
    zc = (np.arange(nbins) + 0.5) * (Lz / nbins)
    r_out = r.max()
    sel = r > r_out - 2.2          # outer-layer candidates (2.2 sigma thick shell)
    zs, rs = z[sel], r[sel]
    h = np.empty(nbins)
    for k, z0 in enumerate(zc):
        dz = zs - z0
        dz -= Lz * np.round(dz / Lz)
        w = np.exp(-0.5 * (dz / (window / 2.0)) ** 2)
        m = (np.abs(dz) <= window) & (w > 1e-6)
        if m.sum() < 4:
            h[k] = np.nan
            continue
        # quadratic fit r = a0 + a1 dz + a2 dz^2 (weighted least squares)
        A = np.vstack([np.ones(m.sum()), dz[m], dz[m] ** 2]).T
        W = np.diag(w[m])
        coef = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ rs[m], rcond=None)[0]
        h[k] = coef[0] - R0
    return M._fill_and_smooth(h, nbins, SMOOTH_BINS)


def method_D(z, r, R0, Lz, nbins, window=DEFAULT_WINDOW, lam=SOFT_LAM):
    """Kernel-weighted soft-upper-envelope estimator."""
    zc = (np.arange(nbins) + 0.5) * (Lz / nbins)
    h = np.empty(nbins)
    for k, z0 in enumerate(zc):
        dz = z - z0
        dz -= Lz * np.round(dz / Lz)
        w = np.exp(-0.5 * (dz / (window / 2.0)) ** 2)
        m = (np.abs(dz) <= window) & (w > 1e-6)
        if m.sum() < 3:
            h[k] = np.nan
            continue
        r_sel, w_sel = r[m], w[m]
        rm = r_sel.max()
        e = np.exp((r_sel - rm) / lam)
        h[k] = float(np.sum(w_sel * r_sel * e) / np.sum(w_sel * e)) - R0
    return M._fill_and_smooth(h, nbins, SMOOTH_BINS)


METHODS = ("A", "B", "C", "D")


def profile(z, r, R0, Lz, nbins, method):
    if method == "A":
        return M.h_profile_slabmax(z, r, R0, Lz, nbins)
    if method == "B":
        return M.h_profile_outerlayer(z, r, R0, Lz, nbins)
    if method == "C":
        return method_C(z, r, R0, Lz, nbins)
    return method_D(z, r, R0, Lz, nbins)


def analyse_case(run_dir, R0, Lz, nbins, t_lo=0.0, t_hi=1e9):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    out = {m: [] for m in METHODS}
    for f, tf in zip(files, times):
        if tf < t_lo or tf > t_hi:
            continue
        _t, z, r = M.read_vtp(f)
        for meth in METHODS:
            h = profile(z, r, R0, Lz, nbins, meth)
            if np.all(np.isnan(h)):
                continue
            A = M.fourier_project(h - np.nanmean(h), Lz, MMAX)
            rec = dict(time=float(tf), method=meth, mean_h=float(np.nanmean(h)),
                       ripple=float(np.nanstd(h)))
            for m in range(1, MMAX + 1):
                rec["A%d" % m] = A[m]
            out[meth].append(rec)
    return out


def band_stats(rows, lo, hi):
    vals = []
    for r in rows:
        for m in range(lo, hi + 1):
            vals.append(r["A%d" % m])
    v = np.array(vals)
    return float(v.mean()), float(np.median(v)), float(v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg63", required=True, help="Stage 7 A=0 packing 0.63 run dir")
    ap.add_argument("--bg70", required=True, help="Stage 7 A=0 packing 0.70 run dir")
    ap.add_argument("--inj", nargs="+", required=True,
                    help="label=dir[:mode] entries, e.g. m2=<dir>:2")
    ap.add_argument("--R0", type=float, default=5.0)
    ap.add_argument("--Lz", type=float, default=160.0)
    ap.add_argument("--nbins", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    nbins = a.nbins or max(64, int(round(a.Lz * 2.0)))
    os.makedirs(a.out, exist_ok=True)

    dz = a.Lz / nbins
    # effective PARTICLE sampling of the outer layer (the physically meaningful
    # resolution): count the particles within 1.5 sigma of the outer radius.
    _tf, _z0, _r0v = M.read_vtp(M.frame_list(a.bg63)[0])
    n_outer = int(np.sum(_r0v > _r0v.max() - 0.7))   # half the radial layer gap
    dz_eff = a.Lz / max(n_outer, 1)
    res_rows = []
    for m in range(1, MMAX + 1):
        lam = a.Lz / m
        ppw = lam / dz_eff
        res_rows.append(dict(mode=m, lambda_z=lam, dz_grid=dz, dz_eff=dz_eff,
                             n_outer_layer=n_outer, points_per_wavelength=ppw,
                             class_=("SAFE" if ppw >= 20 else
                                     "ACCEPTABLE" if ppw >= 10 else "UNRESOLVED")))
    with open(os.path.join(a.out, "stage8_mode_resolution.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res_rows[0].keys()))
        w.writeheader()
        for r in res_rows:
            w.writerow(r)

    cases = {"bg63": a.bg63, "bg70": a.bg70}
    inj_mode = {}
    for spec in a.inj:
        label, rest = spec.split("=", 1)
        if ":" in rest:
            d, mm = rest.rsplit(":", 1)
            inj_mode[label] = int(mm)
        else:
            d = rest
        cases[label] = d

    data = {}
    for label, d in cases.items():
        data[label] = analyse_case(d, a.R0, a.Lz, nbins)
        for meth in METHODS:
            rows = data[label][meth]
            with open(os.path.join(a.out, "m17_frames_%s_%s.csv" % (label, meth)),
                      "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow(r)

    # ---- comparison table ----
    comp = []
    for meth in METHODS:
        row = dict(method=meth)
        for label in ("bg63", "bg70"):
            rows = data[label][meth]
            for band, (lo, hi) in (("low", (1, 4)), ("mid", (5, 8)), ("high", (9, 12))):
                mean, med, mx = band_stats(rows, lo, hi)
                row["%s_%s_mean" % (label, band)] = mean
                row["%s_%s_median" % (label, band)] = med
                row["%s_%s_max" % (label, band)] = mx
        # recovery of the injected modes (frame 0 = pristine injected state)
        rec = {}
        for label, mm in inj_mode.items():
            rows = data[label][meth]
            if not rows:
                continue
            r0 = rows[0]
            amp = r0["A%d" % mm]
            # phase of the target mode at frame 0 (expected: pure sin => 90 deg)
            _tf, zz, rr = M.read_vtp(M.frame_list(cases[label])[0])
            h = profile(zz, rr, a.R0, a.Lz, nbins, meth)
            zc = (np.arange(nbins) + 0.5) * (a.Lz / nbins)
            c = np.cos(2 * np.pi * mm * zc / a.Lz)
            s = np.sin(2 * np.pi * mm * zc / a.Lz)
            aa = 2.0 / nbins * float(np.sum((h - np.nanmean(h)) * c))
            bb = 2.0 / nbins * float(np.sum((h - np.nanmean(h)) * s))
            ph = np.degrees(np.arctan2(bb, aa))
            rec["rec_%s_phase_deg" % label] = ph
            rec["rec_%s_phase_err_deg" % label] = abs(ph - 90.0)
            src = rows  # purity from the same frame
            tot = sum(src[0]["A%d" % k] ** 2 for k in range(1, MMAX + 1)) or 1.0
            # expected amplitude: (h_last - h_in) * A with A = 0.05
            expected = 1.0391 * 0.05
            rec["rec_%s_ratio" % label] = amp / expected
            rec["rec_%s_err" % label] = abs(amp / expected - 1.0)
            rec["rec_%s_purity" % label] = amp ** 2 / tot
            rec["snr_%s" % label] = amp / max(row["bg63_low_mean"], 1e-12)
        row.update(rec)
        comp.append(row)
    with open(os.path.join(a.out, "stage8_reconstruction_comparison.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(comp[0].keys()))
        w.writeheader()
        for r in comp:
            w.writerow(r)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f0 = M.frame_list(cases["bg63"])[0]
        _t, z, r = M.read_vtp(f0)
        fig, ax = plt.subplots(figsize=(7, 3))
        for meth in METHODS:
            h = profile(z, r, a.R0, a.Lz, nbins, meth)
            ax.plot((np.arange(nbins) + 0.5) * dz, h, lw=1, label=meth)
        ax.set_xlabel("z (sigma)")
        ax.set_ylabel("h(z) (sigma)")
        ax.set_title("h(z) overlay, same frame (A=0, packing 0.63)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig1_hz_overlay.png"), dpi=130)

        fig, ax = plt.subplots(figsize=(7, 3))
        for meth in METHODS:
            rows = data["bg63"][meth]
            y = [np.median([x["A%d" % m] for x in rows]) for m in range(1, MMAX + 1)]
            ax.plot(range(1, MMAX + 1), y, marker="o", ms=3, label=meth)
        ax.axhline(0.05, ls=":", c="k", lw=0.8)
        ax.set_xlabel("mode m")
        ax.set_ylabel("median A_m (sigma)")
        ax.set_title("A=0 background spectrum (packing 0.63, noise-off)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig2_bg_spectrum.png"), dpi=130)

        lab3 = [l for l in inj_mode if inj_mode[l] == 3]
        if lab3:
            fig, ax = plt.subplots(figsize=(7, 3))
            for meth in METHODS:
                rows = data[lab3[0]][meth]
                y = [rows[0]["A%d" % m] for m in range(1, MMAX + 1)]
                ax.plot(range(1, MMAX + 1), y, marker="o", ms=3, label=meth)
            ax.axvline(3, ls=":", c="k", lw=0.8)
            ax.set_xlabel("mode m")
            ax.set_ylabel("A_m at frame 0 (sigma)")
            ax.set_title("injected m=3, A=0.05: recovered spectrum")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, "fig3_injected_m3.png"), dpi=130)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot([r["mode"] for r in res_rows],
                [r["points_per_wavelength"] for r in res_rows], marker="o", ms=3)
        ax.axhline(20, ls="--", c="g", lw=0.8)
        ax.axhline(10, ls="--", c="r", lw=0.8)
        ax.set_xlabel("mode m")
        ax.set_ylabel("samples per wavelength")
        ax.set_title("Nyquist budget, Lz=%.0f, dz=%.3f sigma" % (a.Lz, dz))
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig4_samples_per_wavelength.png"), dpi=130)

        fig, ax = plt.subplots(figsize=(6, 3))
        labels = list(METHODS)
        low = [next(r for r in comp if r["method"] == m)["bg63_low_mean"] for m in labels]
        ax.bar(labels, low)
        ax.set_ylabel("low-mode (m=1..4) mean A_m (sigma)")
        ax.set_title("reconstruction method vs low-mode noise floor")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig5_method_vs_lowmode.png"), dpi=130)
    except Exception as exc:      # plotting is optional
        print("plot skipped: %s" % exc)

    hdr = ["method", "bg63_low_mean", "bg63_low_median", "bg63_mid_mean",
           "bg63_high_mean", "bg70_low_mean"]
    print(",".join(hdr))
    for r in comp:
        print(",".join("%.5f" % r.get(h, float("nan")) if isinstance(r.get(h), float)
                       else str(r.get(h, "")) for h in hdr))
    for k in [k for k in comp[0] if k.startswith(("rec_", "snr_"))]:
        print("%-18s %s" % (k, "  ".join("%s=%.3f" % (r["method"], r[k]) for r in comp)))
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
