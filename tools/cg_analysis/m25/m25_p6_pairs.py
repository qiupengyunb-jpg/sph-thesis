#!/usr/bin/env python3
"""Stage 19: P6 metastable two-bead audit (analysis only).

Per frame (Method D reconstruction of h(z), frozen parameters):
  * bead count, bead centres z_i and bead masses m_i, where a bead is a local
    maximum of h plus its catchment basin down to the neighbouring minima;
    mass = Int (h - h_min_local) dz over the basin (sigma^2 units);
  * for frames with exactly two beads: distance d (periodic) and m1/m2.

Then: P(d) histogram over all 2-bead frames (normalised by the sampling range),
F_eff(d) = -k_B T ln P(d) (relative, kT = 1), the approach speed just before
each 2->1 fusion event, and the fraction of time beads spend approaching vs
holding.

Usage: python m25_p6_pairs.py <run_dir> --R0 8 --Lz 160 --label P6_s16
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("../m14", "../m17", "../m15"):
    sys.path.insert(0, os.path.join(HERE, sub))
import m14_axisym_modes as M  # noqa: E402
import m17_recon as R  # noqa: E402


def beads_from_profile(h, Lz, amp_frac=0.25):
    n = h.size
    dz = Lz / n
    # FROZEN detector rule (same as m15): standard topographic prominence with
    # the absolute floor PROM_ABS_SIGMA and distance 2 sigma.
    import m15_beads as B
    amp = float(np.max(h) - np.min(h))
    peaks = B.find_peaks_periodic(h, max(1, int(2.0 / dz)),
                                  max(B.PROM_ABS_SIGMA, amp_frac * amp))
    if not peaks:
        return []
    # catchment basin between the two neighbouring local minima
    mins = [i for i in range(n)
            if h[i] < h[i - 1] and h[i] <= h[(i + 1) % n]]
    out = []
    for p in peaks:
        left = max([m for m in mins if m < p], default=None)
        right = min([m for m in mins if m > p], default=None)
        if left is None:
            left = mins[-1] - n if mins else 0
        if right is None:
            right = mins[0] + n if mins else n
        idx = np.arange(left, right + 1) % n
        base = min(h[left % n], h[right % n])
        mass = float(np.sum(np.maximum(h[idx] - base, 0.0)) * dz)
        out.append(dict(pos=(p * dz) % Lz, mass=mass))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--R0", type=float, default=8.0)
    ap.add_argument("--Lz", type=float, default=160.0)
    ap.add_argument("--nbins", type=int, default=320)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    files = M.frame_list(a.run_dir)
    times = M.frame_times(a.run_dir, files)
    rows = []
    for f, tf in zip(files, times):
        _t, z, r = M.read_vtp(f)
        h = R.method_D(z, r, a.R0, a.Lz, a.nbins)
        bs = beads_from_profile(h, a.Lz)
        bs = sorted(bs, key=lambda b: -b["mass"])[:4]
        rec = dict(time=float(tf), n_bead=len(bs))
        if len(bs) == 2:
            d = abs(bs[0]["pos"] - bs[1]["pos"]) % a.Lz
            rec["d"] = min(d, a.Lz - d)
            rec["m1"] = bs[0]["mass"]
            rec["m2"] = bs[1]["mass"]
            rec["m_ratio"] = bs[0]["mass"] / bs[1]["mass"] if bs[1]["mass"] > 0 else np.nan
        rows.append(rec)
    out = os.path.join(a.run_dir, "m25_pairs.csv")
    keys = ["time", "n_bead", "d", "m1", "m2", "m_ratio"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    two = [r for r in rows if r["n_bead"] == 2 and np.isfinite(r.get("d", np.nan))]
    n2 = len(two)
    frac2 = n2 / max(len(rows), 1)
    if n2 >= 10:
        d = np.array([r["d"] for r in two])
        t = np.array([r["time"] for r in two])
        # P(d) over the geometrically allowed range 0..Lz/2, 16 bins
        edges = np.linspace(0, a.Lz / 2, 17)
        cnt, _ = np.histogram(d, bins=edges)
        P = cnt / max(cnt.sum(), 1)
        with np.errstate(divide="ignore"):
            F = -np.log(np.maximum(P, 1e-12))
        F = F - np.nanmin(F)
        nb = int(np.argmin(F))
        d_pref = 0.5 * (edges[nb] + edges[nb + 1])
        # approach vs hold: sign of dd/dt over consecutive 2-bead frames
        dd = np.diff(d)
        dt = np.diff(t)
        v = dd / np.maximum(dt, 1e-9)
        approaching = float(np.mean(v < -0.005))
        holding = float(np.mean(np.abs(v) <= 0.005))
        # fusion events: 2-bead frame followed by 1-bead frame
        fus = []
        for i in range(1, len(rows)):
            if rows[i - 1]["n_bead"] == 2 and rows[i]["n_bead"] == 1:
                k = i - 1
                j = k
                while j > 0 and rows[j]["n_bead"] == 2:
                    j -= 1
                j += 1
                fus.append(dict(t_fuse=rows[i]["time"], d_before=rows[k]["d"],
                                d_min=float(np.min([r["d"] for r in rows[j:k + 1]])),
                                approach_speed=float(np.mean(v[max(0, j - 1):k]) if k > j else np.nan)))
    else:
        d = np.array([])
        d_pref, approaching, holding, fus = np.nan, np.nan, np.nan, []
    summ = dict(label=a.label or os.path.basename(a.run_dir.rstrip("\\/")),
                n_frames=len(rows), t_end=float(times[-1]),
                frac_two_bead=frac2, n_two_frames=n2,
                d_median=float(np.median(d)) if d.size else np.nan,
                d_min=float(np.min(d)) if d.size else np.nan,
                d_max=float(np.max(d)) if d.size else np.nan,
                d_preferred=d_pref,
                frac_approaching=approaching, frac_holding=holding,
                n_fusion_events=len(fus),
                m_ratio_median=float(np.median([r["m_ratio"] for r in two])) if n2 else np.nan)
    with open(os.path.join(a.run_dir, "m25_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ.keys()))
        w.writeheader()
        w.writerow(summ)
    if fus:
        with open(os.path.join(a.run_dir, "m25_fusions.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fus[0].keys()))
            w.writeheader()
            for r in fus:
                w.writerow(r)
    for k, v in summ.items():
        print("%-20s %s" % (k, v))


if __name__ == "__main__":
    main()
