#!/usr/bin/env python3
"""Compare the solver-side moving_j (Scheme A) and difference_energy (Scheme E)
static force dumps produced by --sg-force-dump=1.

Usage:  python m13r_solver_cmp.py <run_dir_A> <run_dir_B> [--h 2.8] [--interior-pad 2]
"""
import argparse
import csv
import os

import numpy as np


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            pass
    return out


def stats(d, pad, h):
    r = d["r"]
    z = d["z"]
    Fr = d["Fr_total"]
    Fz = d["Fz_total"]
    interior = (r > r.min() + pad * h) & (r < r.max() - pad * h)
    res = {}
    for name, sel in (("all", np.ones_like(r, bool)), ("interior", interior)):
        if not np.any(sel):
            continue
        k = name + "_"
        res[k + "N"] = int(sel.sum())
        res[k + "mean_Fr"] = float(np.mean(Fr[sel]))
        res[k + "rms_Fr"] = float(np.sqrt(np.mean(Fr[sel] ** 2)))
        res[k + "mean_Fz"] = float(np.mean(Fz[sel]))
        res[k + "max_Fz"] = float(np.max(np.abs(Fz[sel])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA")
    ap.add_argument("dirB")
    ap.add_argument("--h", type=float, default=2.8)
    ap.add_argument("--pad", type=float, default=2.0)
    a = ap.parse_args()
    dA = load(os.path.join(a.dirA, "sg_force_split.csv"))
    dB = load(os.path.join(a.dirB, "sg_force_split.csv"))
    sA = stats(dA, a.pad, a.h)
    sB = stats(dB, a.pad, a.h)
    print("%-10s %-26s %-26s" % ("scope", "moving_j (Scheme A)", "difference_energy (E)"))
    for scope in ("all", "interior"):
        if scope + "_N" not in sA:
            continue
        for key, fmt in (("N", "%d"), ("mean_Fr", "%.6e"), ("rms_Fr", "%.6e"),
                         ("mean_Fz", "%.6e"), ("max_Fz", "%.6e")):
            k = scope + "_" + key
            print("%-10s %-26s %-26s" % (scope + "." + key,
                                         fmt % sA[k], fmt % sB[k]))
    print("\nScheme A explicit-branch mean (all particles) = %.6e"
          % float(np.mean(dA["Fr_explicit_J"])))
    print("Scheme E explicit-branch mean (all particles) = %.6e"
          % float(np.mean(dB["Fr_explicit_J"])))
    g2 = dA.get("grad2")
    if g2 is not None:
        pred = -(12.0 / 0.88 / (2.0 * 5.0)) * float(np.mean(g2))
        print("N3 prediction -(lam V0/2R0)<|G|^2> (all)      = %.6e" % pred)
    print("ratio E_mean_Fr / A_mean_Fr (all)             = %.4f"
          % (sB["all_mean_Fr"] / sA["all_mean_Fr"] if sA["all_mean_Fr"] else float("nan")))


if __name__ == "__main__":
    main()
