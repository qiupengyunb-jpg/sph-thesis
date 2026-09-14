"""Population view: how many particles lose kernel support, and when."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import DX, SIGMA0_NUM, frames, particle_fields, neighbor_sums

OUT = Path(r"E:\哈哈\_pr_diag\out")
WANT = (0.0, 20.0, 28.0, 36.0, 42.0, 50.0, 56.0, 60.0, 62.0, 70.0, 80.0)
REGION = 6.0          # |x| < 6a : central neck / bulge section


def gap_of_particle(x, r):
    gap = np.full(len(x), np.nan)
    for x0 in np.arange(-30.0, 30.0, 2.0):
        sel = (x >= x0) & (x < x0 + 2.0)
        if sel.sum() < 8:
            continue
        rr = np.sort(r[sel])
        gap[sel] = np.diff(rr).max() / DX
    return gap


def main():
    allf = frames()
    data = {}
    for want in WANT:
        path, T = min(allf, key=lambda item: abs(item[1] - want))
        f = particle_fields(path)
        s_num, s_mass, nbr, d_min = neighbor_sums(f["pos"], f["mass"])
        f.update(s_num=s_num, s_mass=s_mass, nbr=nbr, d_min=d_min,
                 rho_s=s_num / SIGMA0_NUM, gap=gap_of_particle(f["x"], f["r"]))
        data[want] = f
        print("loaded T=%5.1f" % want)

    ref = data[0.0]
    ref_map = dict(zip(ref["id"], ref["s_num"]))

    # cohort is chosen once, on the worst late frame, then tracked backwards
    late = data[62.0]
    late_ratio = late["s_num"] / np.array([ref_map[i] for i in late["id"]])
    late_reg = np.abs(late["x"]) < REGION
    worst_ids = set(np.asarray(
        late["id"][late_reg])[np.argsort(late_ratio[late_reg])[:200]].tolist())

    rows = []
    for T in sorted(data):
        f = data[T]
        s0 = np.array([ref_map[i] for i in f["id"]])
        ratio = f["s_num"] / s0
        reg = np.abs(f["x"]) < REGION
        rr = ratio[reg]
        sel = np.array([i in worst_ids for i in f["id"]])
        rows.append(dict(
            T=T,
            min_ratio=rr.min(),
            n_lt_095=int((rr < 0.95).sum()),
            n_lt_090=int((rr < 0.90).sum()),
            n_lt_080=int((rr < 0.80).sum()),
            n_lt_070=int((rr < 0.70).sum()),
            n_lt_060=int((rr < 0.60).sum()),
            region_n=int(reg.sum()),
            cohort_support=float(ratio[sel].mean()),
            cohort_nbr=float(f["nbr"][sel].mean()),
            cohort_dmin=float((f["d_min"][sel] / DX).mean()),
            cohort_gap=float(np.nanmean(f["gap"][sel])),
            cohort_rho_c=float(f["rho_c"][sel].mean()),
            cohort_rho_s=float(f["rho_s"][sel].mean()),
            cohort_ratio_rho=float((f["rho_s"][sel] / f["rho_c"][sel]).mean()),
        ))

    keys = list(rows[0].keys())
    with open(OUT / "support_stats.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join("%.10g" % row[k] for k in keys) + "\n")
    print("\nwrote", OUT / "support_stats.csv")

    print("\n| x |<%g   particle counts by support ratio S(T)/S(0)" % REGION)
    print("%-6s %8s %8s %8s %8s %8s %8s" %
          ("T", "<0.95", "<0.90", "<0.80", "<0.70", "<0.60", "min"))
    for row in rows:
        print("%-6.1f %8d %8d %8d %8d %8d %8.3f"
              % (row["T"], row["n_lt_095"], row["n_lt_090"], row["n_lt_080"],
                 row["n_lt_070"], row["n_lt_060"], row["min_ratio"]))

    print("\nworst-200 cohort (fixed id set chosen at T=62)")
    print("%-6s %8s %8s %8s %8s %10s %10s %8s"
          % ("T", "support", "nbr", "dmin/dx", "gap/dx", "rho_c", "rho_s",
             "rho_s/c"))
    for row in rows:
        print("%-6.1f %8.4f %8.2f %8.3f %8.2f %10.6f %10.5f %8.4f"
              % (row["T"], row["cohort_support"], row["cohort_nbr"],
                 row["cohort_dmin"], row["cohort_gap"], row["cohort_rho_c"],
                 row["cohort_rho_s"], row["cohort_ratio_rho"]))

    flat = {}
    for T in sorted(data):
        tag = "T%d" % int(T)
        for name in ("id", "x", "r", "rho_c", "rho_s", "s_num", "nbr",
                     "d_min", "p", "gap", "mass", "indicator"):
            flat["%s_%s" % (tag, name)] = data[T][name]
    np.savez_compressed(OUT / "support_frames.npz", **flat)


if __name__ == "__main__":
    main()
