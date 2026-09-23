#!/usr/bin/env python3
"""Stage 2: merge per-run m15_beads + m15_beadtrack summaries into one table.

Usage: python m15_collect.py --out table.csv label=dir=R0 [label=dir=R0 ...]
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("specs", nargs="+", help="label=dir=R0")
    a = ap.parse_args()
    rows = []
    for spec in a.specs:
        label, d, R0 = spec.split("=")
        b = os.path.join(d, "m15_summary.csv")
        t = os.path.join(d, "m15_track_summary.csv")
        if not (os.path.exists(b) and os.path.exists(t)):
            continue
        bs = list(csv.DictReader(open(b, newline="", encoding="utf-8")))[0]
        ts = list(csv.DictReader(open(t, newline="", encoding="utf-8")))[0]
        frac = dict(x.split("=") for x in bs["class_fractions"].split(";") if "=" in x)
        row = dict(parameter=label, R0=R0,
                   t_first_bead_ge2=bs["t_first_bead_ge2"],
                   t_first_bead_ge4=bs["t_first_bead_ge4"],
                   max_bead_count=ts["max_bead_count"],
                   birth_count=ts["birth_count"], R_birth=ts["R_birth"],
                   merge_events=ts["merge_events"], R_merge=ts["R_merge"],
                   median_bead_lifetime=ts["median_bead_lifetime"],
                   mean_migration_speed=ts["mean_migration_speed"],
                   median_migration_speed=ts["median_migration_speed"],
                   mean_spacing_slope=bs["mean_spacing_slope"],
                   CV_spacing_slope=bs["CV_spacing_slope"],
                   largest_fraction_slope=bs["largest_fraction_slope"],
                   median_CV_spacing=bs["median_CV_spacing"],
                   median_modulation=bs["median_modulation"],
                   modal_class=bs["modal_class"],
                   F0=frac.get("F0", "0"), F1=frac.get("F1", "0"),
                   F2=frac.get("F2", "0"), F3=frac.get("F3", "0"),
                   F4=frac.get("F4", "0"), F5=frac.get("F5", "0"),
                   n_flicker=ts["n_flicker"],
                   median_bead4_hold=bs["bead4_hold_time"])
        rows.append(row)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    hdr = ["parameter", "max_bead_count", "birth_count", "R_birth", "merge_events",
           "median_bead_lifetime", "median_migration_speed", "t_first_bead_ge2",
           "t_first_bead_ge4", "mean_spacing_slope", "largest_fraction_slope",
           "median_CV_spacing", "median_modulation", "modal_class", "F2", "F3", "F4", "F5"]
    print(",".join(hdr))
    for r in rows:
        print(",".join(str(r[h])[:8] for h in hdr))
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
