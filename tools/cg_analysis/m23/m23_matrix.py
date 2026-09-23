#!/usr/bin/env python3
"""Stage 13: aggregate m23_events.csv of many cases into one parameter-effect map.

Usage: python m23_matrix.py <matrix_root> --out table.csv
"""

import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.root, "**", "m23_events.csv"), recursive=True)):
        r = list(csv.DictReader(open(p, newline="", encoding="utf-8")))[0]
        r["dir"] = os.path.dirname(p)
        rows.append(r)
    if not rows:
        raise SystemExit("no m23_events.csv under %s" % a.root)
    keys = ["label", "t_end", "N_max", "t_bead2", "t_bead4", "t_Nmax", "t_coarsen",
            "T_F2", "bead_density_max", "coarsen_rate_N", "L_spec_slope",
            "P_low_end", "dynamics_class", "dir"]
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(",".join(keys[:-1]))
    for r in rows:
        print(",".join(str(r.get(k, "")) for k in keys[:-1]))
    print("-> %s (%d cases)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
