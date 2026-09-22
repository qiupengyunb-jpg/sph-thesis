"""M1.2 independent verification of the axisymmetric film initial state.

Reads ONLY the frame-0 VTP written by the solver and re-derives every I1-I8
quantity from scratch, so the solver's own log line is not taken on trust.

  I2  min(r - R0) over all particles            must be > 0
  I3  min(r - R0)                               must equal h_in = 0.5 sigma
  I4  true min pair distance (z min image)      must exceed 2^(1/6) sigma
  I5  per-layer z spacing incl. the wrap seam   must equal dz
  I6  particles per radial layer                must all equal n, no empty layer
  I8  spread of r inside one layer              must be output precision only

Usage:
  python m12_film_check.py --dir <run dir> --tag <output tag> --R0 5 --Lz 80
"""

import argparse
import os
import re

import numpy as np


def read_frame0(path):
    with open(path, "r") as fh:
        txt = fh.read()
    pos = re.search(r'<DataArray Name="Position"[^>]*>(.*?)</DataArray>', txt, re.S)
    return np.fromstring(pos.group(1), sep=" ").reshape(-1, 3)


def min_image_z(delta, lz):
    out = delta.copy()
    out[:, 0] -= lz * np.round(out[:, 0] / lz)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--R0", type=float, default=5.0)
    ap.add_argument("--Lz", type=float, default=80.0)
    ap.add_argument("--h_in", type=float, default=0.5)
    ap.add_argument("--tol_r", type=float, default=1.0e-4,
                    help="layer grouping tolerance; VTP stores Float32")
    ap.add_argument("--out", default=None, help="optional CSV summary path")
    args = ap.parse_args()

    path = os.path.join(args.dir, "output_" + args.tag,
                        "CGParticles_ite_0000000000.vtp")
    p = read_frame0(path)
    z, r = p[:, 0], p[:, 1]
    n = len(p)
    wca = 2.0 ** (1.0 / 6.0)

    # ---- I2 / I3: solid exclusion and innermost layer -----------------------
    h = r - args.R0
    h_min = float(h.min())
    i2 = h_min > 0.0
    i3 = abs(h_min - args.h_in) < 1.0e-5

    # ---- I4: true minimum pair distance with the z minimum image ------------
    d_all = min_image_z(p[:, None, :] - p[None, :, :], args.Lz)
    dist = np.sqrt((d_all ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = float(dist.min())
    i4 = min_pair > wca

    # ---- I6 / I8: radial layers --------------------------------------------
    order = np.argsort(r, kind="stable")
    rs = r[order]
    layers, start = [], 0
    for k in range(1, len(rs) + 1):
        if k == len(rs) or (rs[k] - rs[start]) > args.tol_r:
            layers.append(order[start:k])
            start = k
    counts = [len(L) for L in layers]
    spreads = [float(r[L].max() - r[L].min()) for L in layers]
    i6 = (len(counts) > 0 and min(counts) == max(counts))
    i8 = max(spreads) <= args.tol_r

    # ---- I5: z spacing inside each layer, including the wrap seam ----------
    worst_seam = 0.0
    worst_gap = 0.0
    dz_est = []
    for L in layers:
        zz = np.sort(z[L])
        gaps = np.diff(zz)
        seam = (args.Lz - zz[-1]) + zz[0]          # wrap-around gap
        dz_est.append(float(np.mean(gaps)))
        worst_gap = max(worst_gap, float(np.abs(gaps - gaps.mean()).max()))
        worst_seam = max(worst_seam, abs(seam - float(gaps.mean())))
    i5 = worst_seam < 1.0e-4 and worst_gap < 1.0e-4

    print("frame0 particles          : %d" % n)
    print("I2  min(r-R0)             : %.8f   (>0)                => %s"
          % (h_min, "PASS" if i2 else "FAIL"))
    print("I3  min(r-R0) == h_in     : %.8f vs %.8f          => %s"
          % (h_min, args.h_in, "PASS" if i3 else "FAIL"))
    print("I4  min pair distance     : %.8f   (WCA %.6f)  => %s"
          % (min_pair, wca, "PASS" if i4 else "FAIL"))
    print("I5  z seam-free           : seam dev %.2e, gap dev %.2e  => %s"
          % (worst_seam, worst_gap, "PASS" if i5 else "FAIL"))
    print("I6  per-layer counts      : %d layers, counts %s  => %s"
          % (len(counts), sorted(set(counts)), "PASS" if i6 else "FAIL"))
    print("I8  r spread per layer    : max %.2e                   => %s"
          % (max(spreads), "PASS" if i8 else "FAIL"))
    print("    (dz per layer         : %s)" % ", ".join("%.6f" % v for v in dz_est))
    print("    layer radii           : %s"
          % ", ".join("%.6f" % float(r[L].mean()) for L in layers))

    if args.out:
        with open(args.out, "w") as fh:
            fh.write("check,value,verdict\n")
            fh.write("I2_min_r_minus_R0,%.10f,%s\n" % (h_min, i2))
            fh.write("I3_min_r_minus_R0_equals_h_in,%.10f,%s\n" % (h_min, i3))
            fh.write("I4_min_pair_distance,%.10f,%s\n" % (min_pair, i4))
            fh.write("I4_wca_contact_distance,%.10f,reference\n" % wca)
            fh.write("I5_worst_seam_deviation,%.3e,%s\n" % (worst_seam, i5))
            fh.write("I6_layer_count_min,%d,%s\n" % (min(counts), i6))
            fh.write("I6_layer_count_max,%d,%s\n" % (max(counts), i6))
            fh.write("I6_number_of_layers,%d,%s\n" % (len(layers), i6))
            fh.write("I8_max_r_spread,%.3e,%s\n" % (max(spreads), i8))
        print("\nwrote " + args.out)


if __name__ == "__main__":
    main()
