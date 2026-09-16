"""Is the healthy control actually exhibiting PR growth, or just relaxing?

Measures the interface perturbation amplitude A(T) = |Fourier coefficient of
the outer film radius at the domain fundamental| for the equal-radius periodic
case (one wavelength per domain), plus the mean/max outer radius.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from run_metrics import load, frames, BIN


def outer_profile(x, r, xlo, xhi, nbins=180):
    edges = np.linspace(xlo, xhi, nbins + 1)
    idx = np.digitize(x, edges) - 1
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.sum():
            prof[b] = r[m].max()
    return prof


def main():
    base = Path(r"E:\sphmethod\SPH_results_center\healthy_control_20260914"
                r"\uniform_r24_lam18_T80")
    liq = max([d for d in base.glob("output*") if d.is_dir()],
              key=lambda d: len(list(d.glob("LiquidFilmHalf_*.vtp"))))
    fs = frames(liq, 1e9)
    print("healthy control: %d frames T=%.0f..%.0f" % (len(fs), fs[0][1],
                                                       fs[-1][1]))
    xlo, xhi = -9.0, 9.0            # domain length 18a, centred on 0
    xc = 0.5 * (xlo + xhi)
    rows = []
    A0 = None
    for p, T in fs:
        a = load(p)
        x, r = a["Position"][:, 0], a["Position"][:, 1]
        prof = outer_profile(x - xc, r, xlo - xc, xhi - xc)
        ok = ~np.isnan(prof)
        if ok.sum() < 20:
            continue
        xx = np.linspace(xlo - xc, xhi - xc, len(prof))[ok]
        yy = prof[ok] - np.nanmean(prof[ok])
        # one wavelength across the domain -> fundamental is the sin/cos pair
        k = 2.0 * np.pi / (xhi - xlo)
        c1 = 2.0 / ok.sum() * np.sum(yy * np.cos(k * xx))
        s1 = 2.0 / ok.sum() * np.sum(yy * np.sin(k * xx))
        A = float(np.hypot(c1, s1))
        if A0 is None:
            A0 = A
        rows.append((T, A, A / A0, float(np.nanmean(prof)),
                     float(np.nanmax(prof))))
    print("%-7s %12s %10s %10s %10s" % ("T", "A", "A/A0", "r_mean", "r_max"))
    for T, A, ratio, rm, rx in rows[::max(1, len(rows) // 20)]:
        print("%-7.1f %12.6f %10.4f %10.5f %10.5f" % (T, A, ratio, rm, rx))
    print()
    print("A/A0 at last frame = %.4f  (T=%.0f)" % (rows[-1][2], rows[-1][0]))
    print("r_mean drift  = %.6f -> %.6f" % (rows[0][3], rows[-1][3]))
    with open(r"E:\哈哈\_pr_diag\out\healthy\amplitude.csv", "w",
              encoding="utf-8") as fh:
        fh.write("T,A,A_over_A0,r_mean,r_max\n")
        for row in rows:
            fh.write("%.6g,%.8g,%.8g,%.8g,%.8g\n" % row)


if __name__ == "__main__":
    main()
