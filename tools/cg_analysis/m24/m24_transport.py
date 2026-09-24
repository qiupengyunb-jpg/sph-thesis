#!/usr/bin/env python3
"""Stage 17B: quantify particle transport ("liquid capability") from VTP only.

Outputs per case (in the run directory):
  m24_msd.csv          MSD_z and MSD_r versus lag, overall and per layer
  m24_exchange.csv     neighbour-exchange fraction X(t) and tau_50
  m24_clusters.csv     cluster-count / split-event time series
  m24_summary.csv      one row: D_z, D_r, D_ratio, tau_50, splits, class

Particles are aligned by OriginalID, so cell-list re-sorting cannot corrupt the
trajectories.  z is periodic; layers are defined from the INITIAL radius:
wall (h<1.0 sigma), inner (1.0<=h<1.8), outer (h>=1.8).
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "m14"))
import m14_axisym_modes as M  # noqa: E402

NB_CUT = 2.8      # neighbour-definition cutoff (kernel support), sigma
BOND = 1.6        # bond cutoff for cluster labelling, sigma


def load_frames(run_dir):
    files = M.frame_list(run_dir)
    times = M.frame_times(run_dir, files)
    pos, ids = [], []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        pts = re.search(r"<Points>(.*?)</Points>", txt, re.S).group(1)
        p = np.fromstring(re.search(r'Name="Position"[^>]*>(.*?)</DataArray>',
                                    pts, re.S).group(1), sep=" ").reshape(-1, 3)
        i = np.fromstring(re.search(r'Name="OriginalID"[^>]*>(.*?)</DataArray>',
                                    txt, re.S).group(1), sep=" ").astype(int)
        o = np.argsort(i)
        pos.append(p[o, :2])
        ids.append(i[o])
    return times, pos, ids


def neighbours(z, r, Lz, cut):
    """periodic-z neighbour lists via a tripled-z KD-tree"""
    z3 = np.concatenate([z - Lz, z, z + Lz])
    r3 = np.tile(r, 3)
    idx3 = np.tile(np.arange(len(z)), 3)
    tree = cKDTree(np.column_stack([z3, r3]))
    return tree.query_ball_point(np.column_stack([z, r]), cut), idx3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--Lz", type=float, default=160.0)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-origins", type=int, default=60)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    t, pos, ids = load_frames(a.run_dir)
    n = len(pos)
    P = np.array(pos)                      # (n, N, 2)
    if not all(np.array_equal(ids[0], x) for x in ids):
        print("WARNING: OriginalID sets differ between frames")
    dt = float(np.median(np.diff(t)))
    z0, r0 = P[0, :, 0], P[0, :, 1]
    h0 = r0 - a.R0
    layers = dict(wall=h0 < 1.0, inner=(h0 >= 1.0) & (h0 < 1.8), outer=h0 >= 1.8)

    # ---------------- MSD with multiple time origins ----------------
    lag_frames = np.arange(1, min(n // 4, 200), a.stride)
    origins = np.linspace(0, n - lag_frames.max() - 1, min(a.max_origins, n - lag_frames.max())).astype(int)
    rows = []
    for lag in lag_frames:
        ia = P[(origins + lag)][..., 0]             # (n_origins, N)
        ib = P[origins][..., 0]
        dz = ia - ib
        dz -= a.Lz * np.round(dz / a.Lz)
        dr = P[(origins + lag)][..., 1] - P[origins][..., 1]
        rec = dict(lag=float(lag * dt),
                   msd_z=float(np.mean(dz ** 2)), msd_r=float(np.mean(dr ** 2)))
        for name, sel in layers.items():
            if sel.sum() >= 3:
                rec["msd_z_" + name] = float(np.mean(dz[:, sel] ** 2))
                rec["msd_r_" + name] = float(np.mean(dr[:, sel] ** 2))
        rows.append(rec)
    with open(os.path.join(a.run_dir, "m24_msd.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    fitsub = (np.array([r["lag"] for r in rows]) < 0.4 * t[-1])
    lag_a = np.array([r["lag"] for r in rows])[fitsub]
    D_z = float(np.polyfit(lag_a, [r["msd_z"] for r in rows][:len(lag_a)], 1)[0] / 2.0) if fitsub.sum() > 2 else np.nan
    D_r = float(np.polyfit(lag_a, [r["msd_r"] for r in rows][:len(lag_a)], 1)[0] / 2.0) if fitsub.sum() > 2 else np.nan

    # ---------------- neighbour exchange ----------------
    nb0 = neighbours(z0, r0, a.Lz, NB_CUT)[0]
    nb0 = [set() for _ in range(len(z0))] if nb0 is None else nb0
    nbs, idx3 = neighbours(z0, r0, a.Lz, NB_CUT)
    nb0 = [set(idx3[n] % len(z0)) for n in nbs]
    ex_rows = []
    sample = np.arange(0, len(z0), max(1, len(z0) // 60))
    for k in range(0, n, max(1, n // 40)):
        zt, rt = P[k, :, 0], P[k, :, 1]
        nbt = neighbours(zt, rt, a.Lz, NB_CUT)[0]
        frac = []
        for i in sample:
            if not nb0[i]:
                continue
            keep = sum(1 for j in nbt[i] if (idx3[j] % len(z0)) in nb0[i])
            frac.append(1.0 - keep / len(nb0[i]))
        ex_rows.append(dict(time=float(t[k]), X=float(np.mean(frac)) if frac else np.nan))
    with open(os.path.join(a.run_dir, "m24_exchange.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["time", "X"])
        w.writeheader()
        for r in ex_rows:
            w.writerow(r)
    tau50 = next((r["time"] for r in ex_rows if r["X"] >= 0.5), np.nan)
    X_end = ex_rows[-1]["X"]

    # ---------------- cluster labelling / splits ----------------
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    cl_rows, prev_label, splits = [], None, 0
    for k in range(0, n, max(1, n // 40)):
        nbk = neighbours(P[k, :, 0], P[k, :, 1], a.Lz, BOND)[0]
        row, col = [], []
        for i, lst in enumerate(nbk):
            for j in lst:
                jj = idx3[j] % len(z0)
                if jj != i:
                    row.append(i)
                    col.append(jj)
        g = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(z0), len(z0)))
        ncomp, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab)
        big = int(np.sum(sizes >= 3))
        if prev_label is not None:
            for c in np.unique(prev_label):
                m = prev_label[prev_label == c]
                if len(m) >= 6:
                    new = np.unique(lab[m])
                    sub = [int(np.sum(lab[m] == q)) for q in new]
                    if len([s for s in sub if s >= 3]) >= 2:
                        splits += 1
        prev_label = lab
        cl_rows.append(dict(time=float(t[k]), n_cluster=int(ncomp), n_cluster_ge3=big,
                            largest=int(sizes.max()), split_events_cum=splits))
    with open(os.path.join(a.run_dir, "m24_clusters.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cl_rows[0].keys()))
        w.writeheader()
        for r in cl_rows:
            w.writerow(r)

    D0_free = 1.0     # k_BT/gamma with gamma=1
    summ = dict(label=a.label or os.path.basename(a.run_dir.rstrip("\\/")),
                n_particles=len(z0), n_frames=n, t_end=float(t[-1]),
                D_z=D_z, D_r=D_r, D_z_over_free=D_z / D0_free, D_r_over_free=D_r / D0_free,
                D_ratio_rz=(D_r / D_z) if D_z else np.nan,
                tau_50_exchange=tau50, X_end=X_end,
                cluster_splits=splits,
                n_cluster_ge3_end=cl_rows[-1]["n_cluster_ge3"],
                largest_cluster_end=cl_rows[-1]["largest"])
    Dn = D_z + D_r
    if Dn / 2.0 > 0.15:
        summ["state"] = "A liquid film"
    elif Dn / 2.0 > 1e-3:
        summ["state"] = "B viscous liquid"
    else:
        summ["state"] = "C glassy/jammed solid"
    with open(os.path.join(a.run_dir, "m24_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys()))
        w.writeheader()
        w.writerow(summ)
    for k, v in summ.items():
        print("%-22s %s" % (k, v))


if __name__ == "__main__":
    main()
