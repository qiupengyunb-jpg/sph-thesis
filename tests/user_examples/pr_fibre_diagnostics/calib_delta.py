"""2C calibration: is the self-referenced kernel-completeness deficit separable?

DEFINITION USED HERE (S), stated once and used everywhere:

    S_i = W(0) + Σ_{j in liquid, |x_i-x_j| < 2h} W(|x_i-x_j|)

  * kernel      : KernelWendlandC2, h = 1.3*dx, cutoff = 2h
  * self term   : W(0) included (same convention as the library's
                  DensitySummation and as the Shepard sum used by the
                  wetting classes)
  * neighbours  : LIQUID-LIQUID only (no fibre, no axis-symmetry support body)
  * no mass / volume weighting and no gradient -> this is a kernel
    completeness (number density) measure.

This is deliberately NOT:
  * PositionDivergence  = -Σ ∇W_ij·V_j·r_ij   (a gradient-based quantity,
                          and the quantity that defines Indicator)
  * summation density   = S/σ0                 (same S, but used as a density;
                          its ABSOLUTE value carries the free-surface
                          truncation baseline, e.g. 0.98 for a healthy
                          surface particle, which is why it must not be
                          used with an absolute threshold)

    δ_i(T) = 1 - S_i(T)/S_i(0)

Only the ratio is used, so the free-surface baseline cancels by construction.
This is exactly what distinguishes δ from "using the summation density".
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
import v1io
from run_metrics import load as load_v2

LAMBDAS = [12, 16, 18, 20, 24, 10, 14, 22, 26, 30]
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10]


def w2d(r, h):
    q = r / h
    out = np.zeros_like(q)
    m = q < 2.0
    qq = q[m]
    out[m] = 7.0 / (4.0 * np.pi * h * h) * (1 - 0.5 * qq) ** 4 * (1 + 2 * qq)
    return out


def s_values(pos, dx):
    """Self-inclusive liquid-liquid kernel sum for every particle."""
    h = 1.3 * dx
    w0 = 7.0 / (4.0 * np.pi * h * h)
    tree = cKDTree(pos)
    pairs = tree.query_ball_point(pos, 2.0 * h, return_sorted=False)
    s = np.full(len(pos), w0)
    nbr = np.zeros(len(pos), dtype=np.int64)
    for i, js in enumerate(pairs):
        js = np.asarray(js, dtype=np.int64)
        js = js[js != i]
        if js.size == 0:
            continue
        d = np.linalg.norm(pos[js] - pos[i], axis=1)
        s[i] += w2d(d, h).sum()
        nbr[i] = js.size
    return s, nbr


def analyse(frame_list, dx, region=None, label=""):
    """frame_list = [(path, T, loader)]; region = function(x0) -> bool mask."""
    p0, T0, ld0 = frame_list[0]
    a0 = ld0(p0)
    pos0 = a0["Position"][:, :2]
    id0 = a0["OriginalID"].astype(np.int64)
    ind0 = np.where(a0["Indicator"] > 0.5, 1, 0)
    s0, nbr0 = s_values(pos0, dx)
    keep = np.ones(len(id0), dtype=bool)
    if region is not None:
        keep = region(pos0)
    interior = keep & (ind0 == 0)
    ref = dict(zip(id0[interior], s0[interior]))
    n_int = int(interior.sum())

    flips = {}      # id -> (T_flip, delta_prev, nbr_prev)
    prev_delta = {}
    rows = []
    for path, T, ld in frame_list:
        a = ld(path)
        ids = a["OriginalID"].astype(np.int64)
        s, nbr = s_values(a["Position"][:, :2], dx)
        ind = np.where(a["Indicator"] > 0.5, 1, 0)
        cur = {int(i): (sv, nv, iv) for i, sv, nv, iv in zip(ids, s, nbr, ind)}
        delta = np.array([1.0 - cur[i][0] / ref[i] if i in ref else np.nan
                          for i in ref])
        row = dict(T=T)
        for q in (50, 90, 95, 99):
            row["p%d" % q] = float(np.nanpercentile(delta, q))
        row["max"] = float(np.nanmax(delta))
        row["mean"] = float(np.nanmean(delta))
        for th in THRESHOLDS:
            row["frac_%g" % th] = float((delta > th).sum()) / max(n_int, 1)
            row["n_%g" % th] = int((delta > th).sum())
        # Indicator flips of the initially-interior cohort
        newly = 0
        for i in ref:
            if i in cur and cur[i][2] == 1 and i not in flips:
                flips[i] = (T, prev_delta.get(i, np.nan), 0)
                newly += 1
        row["new_flips"] = newly
        row["n_flipped_total"] = len(flips)
        rows.append(row)
        prev_delta = {int(i): d for i, d in zip(np.asarray(list(ref.keys())),
                                                delta)}
        print("  %-14s T=%5.1f  p95=%.4f  p99=%.4f  max=%.4f  "
              "n(>0.05)=%d  flipped=%d"
              % (label, T, row["p95"], row["p99"], row["max"],
                 row["n_0.05"], len(flips)))
    return rows, flips, n_int


def main():
    out = Path(r"E:\哈哈\_pr_diag\out\calib")
    out.mkdir(parents=True, exist_ok=True)

    print("=== A. equal-radius periodic PR cases (healthy) ===")
    summary = []
    per_case = {}
    for lam in LAMBDAS:
        try:
            fs = v1io.frames(lam)
        except FileNotFoundError as exc:
            print("  lambda=%s: %s" % (lam, exc))
            continue
        fl = [(p, T, v1io.load) for p, T in fs]
        rows, flips, n_int = analyse(fl, 1.0 / 16.0, label="lam%g" % lam)
        per_case[lam] = (rows, flips, n_int)
        # worst over time: the quantity that would drive a gate
        for th in THRESHOLDS:
            key = "frac_%g" % th
            summary.append((lam, th, max(r[key] for r in rows),
                            max(r["n_%g" % th] for r in rows),
                            max(r["p99"] for r in rows),
                            max(r["max"] for r in rows), n_int))
    with open(out / "equal_radius_summary.csv", "w", encoding="utf-8") as fh:
        fh.write("lambda,threshold,worst_frac,worst_n,worst_p99,worst_max,"
                 "n_interior\n")
        for row in summary:
            fh.write("%g,%.3g,%.6f,%d,%.6f,%.6f,%d\n" % row)
    np.savez_compressed(
        out / "equal_radius_rows.npz",
        **{"lam%g_%s" % (lam, k): np.array([r[k] for r in rows])
           for lam, (rows, _, _) in per_case.items()
           for k in rows[0]})

    print("\n=== B. variable-radius r=24 broadband case (pathological) ===")
    base = Path(r"E:\sphmethod\SPH_results_center\recon_tapered_20260914"
                r"\r24_broadband_T80\output_recon_r24_T80")
    fs = []
    for p in base.glob("LiquidFilmHalf_*.vtp"):
        import xml.etree.ElementTree as ET
        t = float(ET.parse(p).getroot()
                  .find('.//FieldData/DataArray[@Name="TimeValue"]').text)
        fs.append((p, t / 0.67082039325))
    fs.sort(key=lambda it: it[1])
    fl = [(p, T, load_v2) for p, T in fs]
    rows, flips, n_int = analyse(
        fl, 1.0 / 24.0, region=lambda p: np.abs(p[:, 0]) < 6.0,
        label="varradial")
    with open(out / "variable_rows.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            fh.write(",".join("%.6g" % r[k] for k in rows[0]) + "\n")
    flip_T = np.array([v[0] for v in flips.values()])
    flip_d = np.array([v[1] for v in flips.values()])
    print("  flipped interior particles: %d" % len(flips))
    if len(flips):
        print("  first flip at T=%.1f ; median flip T=%.1f"
              % (flip_T.min(), np.median(flip_T)))
        print("  delta just before flipping: p50=%.4f p90=%.4f max=%.4f"
              % (np.nanpercentile(flip_d, 50), np.nanpercentile(flip_d, 90),
                 np.nanmax(flip_d)))
    np.savez_compressed(out / "variable_rows.npz",
                        **{k: np.array([r[k] for r in rows]) for k in rows[0]},
                        flip_T=flip_T, flip_delta=flip_d)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
