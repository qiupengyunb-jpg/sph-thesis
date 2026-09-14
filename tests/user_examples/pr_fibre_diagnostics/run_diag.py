"""Full time series of the geometry / support / density diagnostic.

Writes:
  out/frame_metrics.csv     one row per output frame
  out/per_particle.npz      per-particle arrays for the key frames
No solver run and no solver modification is involved.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import (DX, H, CUTOFF, SIGMA0_NUM, frames, particle_fields,
                       neighbor_sums)

OUT = Path(r"E:\哈哈\_pr_diag\out")
OUT.mkdir(parents=True, exist_ok=True)

CENTRAL = 4.0          # |x| < 4a : the lesion region found earlier
BIN = 2.0              # axial bin width in fibre radii (earlier convention)
KEY_T = (0.0, 20.0, 28.0, 34.0, 36.0, 42.0, 50.0, 60.0, 72.0, 80.0)


def gap_per_bin(x, r, x0, width):
    """Max radial gap (dx units) inside the axial bin [x0, x0+width)."""
    sel = (x >= x0) & (x < x0 + width)
    if sel.sum() < 8:
        return np.nan, np.nan, np.nan
    rr = np.sort(r[sel])
    d = np.diff(rr)
    k = int(np.argmax(d))
    return d[k] / DX, rr[k], rr[k + 1]


def bin_gap_map(x, r):
    """Per-particle max radial gap of the 2a bin the particle sits in."""
    gap = np.full(len(x), np.nan)
    for x0 in np.arange(-30.0, 30.0, BIN):
        sel = (x >= x0) & (x < x0 + BIN)
        if sel.sum() < 8:
            continue
        rr = np.sort(r[sel])
        g = np.diff(rr).max() / DX
        gap[sel] = g
    return gap


def connectivity(pos, thr):
    """Largest connected fraction using a simple union-find on a cell list."""
    cell = thr
    key = np.floor(pos / cell).astype(np.int64)
    buckets = {}
    for i, k in enumerate(map(tuple, key)):
        buckets.setdefault(k, []).append(i)
    parent = np.arange(len(pos))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rng = range(-1, 2)
    thr2 = thr * thr
    for (kx, ky), idxs in buckets.items():
        for dkx in rng:
            for dky in rng:
                other = buckets.get((kx + dkx, ky + dky))
                if not other:
                    continue
                for a_i in idxs:
                    a = pos[a_i]
                    for b_i in other:
                        if b_i <= a_i:
                            continue
                        d = pos[b_i] - a
                        if d[0] * d[0] + d[1] * d[1] <= thr2:
                            union(a_i, b_i)
    roots = np.array([find(i) for i in range(len(pos))])
    _, counts = np.unique(roots, return_counts=True)
    counts = np.sort(counts)[::-1]
    return len(counts), counts[0] / len(pos)


def main():
    fs = frames()
    base = None
    ref = {}
    rows = []
    store = {}

    for path, T in fs:
        f = particle_fields(path)
        s_num, s_mass, nbr, d_min = neighbor_sums(f["pos"], f["mass"])
        rho_s = s_num / SIGMA0_NUM
        gap = bin_gap_map(f["x"], f["r"])

        if base is None:
            base = f["id"].copy()
            ref["s_num"] = dict(zip(f["id"], s_num))

        central = np.abs(f["x"]) < CENTRAL
        g_central = np.nanmax(gap[central])

        # support ratio for the same particles, relative to the initial frame
        s0 = np.array([ref["s_num"][i] for i in f["id"]])
        ratio = s_num / s0

        # lesion set: worst axial gaps inside the central region
        worst = central & (gap > 1.5)
        if worst.sum() < 20:
            worst = central

        n_blocks, biggest = connectivity(f["pos"], 1.5 * DX)

        rows.append(dict(
            T=T,
            max_gap_central=g_central,
            max_gap_interior=np.nanmax(gap[np.abs(f["x"]) < 26.0]),
            min_support_ratio_central=ratio[central].min(),
            p5_support_ratio_central=np.percentile(ratio[central], 5),
            mean_support_ratio_central=ratio[central].mean(),
            min_neighbors_central=nbr[central].min(),
            mean_neighbors_central=nbr[central].mean(),
            max_dmin_central=(d_min[central] / DX).max(),
            rho_c_min_central=f["rho_c"][central].min(),
            rho_c_mean_central=f["rho_c"][central].mean(),
            rho_s_min_central=rho_s[central].min(),
            rho_s_mean_central=rho_s[central].mean(),
            lesion_n=int(worst.sum()),
            lesion_rho_c_mean=float(f["rho_c"][worst].mean()),
            lesion_rho_s_mean=float(rho_s[worst].mean()),
            lesion_support_ratio_mean=float(ratio[worst].mean()),
            lesion_gap_mean=float(np.nanmean(gap[worst])),
            blocks=n_blocks,
            biggest_block_frac=biggest,
        ))

        if T in KEY_T:
            store["T%d" % int(T)] = dict(
                x=f["x"], r=f["r"], id=f["id"], rho_c=f["rho_c"], p=f["p"],
                rho_s=rho_s, s_num=s_num, s_mass=s_mass, gap=gap,
                nbr=nbr, d_min=d_min, ratio=ratio, mass=f["mass"],
                indicator=f["indicator"])
        print("T=%6.2f  gap=%5.2f  support_min=%.4f  rho_c=[%.5f,%.5f]  "
              "rho_s=[%.4f,%.4f]  blocks=%d"
              % (T, g_central, ratio[central].min(),
                 f["rho_c"][central].min(), f["rho_c"][central].max(),
                 rho_s[central].min(), rho_s[central].max(), n_blocks))

    keys = sorted(rows[0].keys())
    with open(OUT / "frame_metrics.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join("%.10g" % row[k] if isinstance(row[k], float)
                              else str(row[k]) for k in keys) + "\n")
    np.savez_compressed(OUT / "per_particle.npz", **store)
    print("\nwrote", OUT / "frame_metrics.csv")
    print("wrote", OUT / "per_particle.npz", "frames:", list(store))


if __name__ == "__main__":
    main()
