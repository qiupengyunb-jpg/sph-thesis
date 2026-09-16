"""delta(t) for the same-program / same-resolution healthy control.

Definition identical to calib_delta.py (see its module docstring):
    S_i = W(0) + sum over LIQUID neighbours within 2h of W(|x_i-x_j|)
    delta_i(T) = 1 - S_i(T)/S_i(0),  tracked by OriginalID
Cohort: particles that are interior (Indicator == 0) at T = 0.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from calib_delta import s_values, THRESHOLDS
from run_metrics import load, frames, DX

RUN = Path(r"E:\sphmethod\SPH_results_center\healthy_control_20260914"
           r"\uniform_r24_lam18_T80\output_uniform_r24_lam18_T80")
OUT = Path(r"E:\哈哈\_pr_diag\out\healthy")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fs = frames(RUN, 1e9)
    print("healthy control (v2 program, reconstructed path, r=24, uniform,"
          " lambda/a=18): %d frames, T=%.0f..%.0f"
          % (len(fs), fs[0][1], fs[-1][1]))

    a0 = load(fs[0][0])
    id0 = a0["OriginalID"].astype(np.int64)
    ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
    s0, nbr0 = s_values(a0["Position"][:, :2], DX)
    interior = ind0 == 0
    ref = dict(zip(id0[interior], s0[interior]))
    print("interior particles at T=0: %d of %d"
          % (interior.sum(), len(id0)))

    rows = []
    flips = {}
    prev = {}
    for p, T in fs:
        a = load(p)
        ids = a["OriginalID"].astype(np.int64)
        ind = np.where(a["Indicator"] > 0.5, 1, 0)
        s, nbr = s_values(a["Position"][:, :2], DX)
        cur = {int(i): (sv, nv, iv)
               for i, sv, nv, iv in zip(ids, s, nbr, ind)}
        d = np.array([1.0 - cur[i][0] / ref[i] for i in ref])
        row = dict(T=T, mean=float(np.nanmean(d)))
        for q in (50, 90, 95, 99):
            row["p%d" % q] = float(np.nanpercentile(d, q))
        row["max"] = float(np.nanmax(d))
        for th in THRESHOLDS:
            row["n_%g" % th] = int((d > th).sum())
        for i in ref:
            if i in cur and cur[i][2] == 1 and i not in flips:
                flips[i] = (T, prev.get(i, np.nan))
        rows.append(row)
        prev = {int(i): v for i, v in zip(ref.keys(), d)}
        print("  T=%5.1f  p90=%.5f  p95=%.5f  p99=%.5f  max=%.5f  "
              "n(>0.02)=%d  n(>0.03)=%d  n(>0.05)=%d  flipped=%d"
              % (T, row["p90"], row["p95"], row["p99"], row["max"],
                 row["n_0.02"], row["n_0.03"], row["n_0.05"], len(flips)))

    keys = list(rows[0].keys())
    with open(OUT / "healthy_rows.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("%.8g" % r[k] for k in keys) + "\n")

    mx = max(r["max"] for r in rows)
    print("\nmax delta over the whole run      = %.5f" % mx)
    print("delta_thr=0.03 margin             = %.1fx" % (0.03 / mx))
    for th in (0.02, 0.03, 0.05):
        n = max(r["n_%g" % th] for r in rows)
        first = next((r["T"] for r in rows if r["n_%g" % th] > 0), None)
        print("  threshold %.2f: worst count=%d, first T=%s" % (th, n, first))
    if flips:
        ft = np.array([v[0] for v in flips.values()])
        fd = np.array([v[1] for v in flips.values()])
        print("indicator flips: %d, first T=%.1f, median T=%.1f"
              % (len(flips), ft.min(), np.median(ft)))
        print("  delta before flipping: p50=%.4f p90=%.4f max=%.4f"
              % (np.nanpercentile(fd, 50), np.nanpercentile(fd, 90),
                 np.nanmax(fd)))
    else:
        print("indicator flips: none")
    print("\nwrote", OUT / "healthy_rows.csv")


if __name__ == "__main__":
    main()
