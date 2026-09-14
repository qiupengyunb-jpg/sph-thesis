"""Particle cohort tracking: geometry loss vs kernel support vs density."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:\哈哈\_pr_diag")
from diag_core import DX, SIGMA0_NUM, frames, particle_fields, neighbor_sums

OUT = Path(r"E:\哈哈\_pr_diag\out")
WANT = (0.0, 20.0, 28.0, 36.0, 42.0, 50.0, 56.0, 60.0, 62.0, 70.0, 80.0)


def bin_gaps(x, r):
    """Return {bin start: (max gap in dx, r_low, r_high)} for the central bins."""
    out = {}
    for x0 in np.arange(-30.0, 30.0, 2.0):
        sel = (x >= x0) & (x < x0 + 2.0)
        if sel.sum() < 8:
            continue
        rr = np.sort(r[sel])
        d = np.diff(rr)
        k = int(np.argmax(d))
        out[round(x0, 1)] = (d[k] / DX, rr[k], rr[k + 1])
    return out


def main():
    allf = frames()
    fs = []
    for want in WANT:                       # match by nearest available frame
        p, T = min(allf, key=lambda item: abs(item[1] - want))
        if abs(T - want) > 0.5:
            raise SystemExit("no frame near T=%.1f" % want)
        fs.append((p, want))                # key by the nominal T
    data = {}
    for path, T in fs:
        f = particle_fields(path)
        s_num, s_mass, nbr, d_min = neighbor_sums(f["pos"], f["mass"])
        f.update(s_num=s_num, s_mass=s_mass, nbr=nbr, d_min=d_min,
                 rho_s=s_num / SIGMA0_NUM)
        data[T] = f
        print("loaded T=%5.1f" % T)

    ref = data[0.0]
    ref_map = dict(zip(ref["id"], ref["s_num"]))
    ref_dmin = dict(zip(ref["id"], ref["d_min"]))

    # ---- cohort: particles inside the four central neck bins at the late frame
    late = data[62.0]
    g = bin_gaps(late["x"], late["r"])
    print("\ncentral bins at T=62 (gap in dx):")
    for k in sorted(g):
        if -4.1 <= k < 4.0:
            print("   x=[%+.0f,%+.0f]  gap=%5.2f  r=%.3f->%.3f"
                  % (k, k + 2, g[k][0], g[k][1], g[k][2]))
    hot_bins = [k for k, v in g.items() if v[0] > 3.0 and abs(k) < 6]
    mask = np.zeros(len(late["x"]), dtype=bool)
    for k in hot_bins:
        mask |= (late["x"] >= k) & (late["x"] < k + 2.0)
    cohort_ids = set(late["id"][mask].tolist())
    print("cohort = %d particles from bins %s" % (len(cohort_ids), hot_bins))

    # ---- time series for that cohort, matched by OriginalID
    rows = []
    for T in sorted(data):
        f = data[T]
        sel = np.array([i in cohort_ids for i in f["id"]])
        s0 = np.array([ref_map[i] for i in f["id"]])
        ratio = f["s_num"] / s0
        gapmax = max(v[0] for v in bin_gaps(f["x"], f["r"]).values())
        rows.append(dict(
            T=T,
            cohort_n=int(sel.sum()),
            cohort_gap_bin_mean=float(np.mean([bin_gaps(f["x"], f["r"])[k][0]
                                               for k in hot_bins
                                               if k in bin_gaps(f["x"], f["r"])])) if hot_bins else np.nan,
            cohort_support_ratio_mean=float(ratio[sel].mean()),
            cohort_support_ratio_min=float(ratio[sel].min()),
            cohort_nbr_mean=float(f["nbr"][sel].mean()),
            cohort_nbr_min=float(f["nbr"][sel].min()),
            cohort_dmin_mean=float((f["d_min"][sel] / DX).mean()),
            cohort_rho_c_mean=float(f["rho_c"][sel].mean()),
            cohort_rho_c_min=float(f["rho_c"][sel].min()),
            cohort_rho_s_mean=float(f["rho_s"][sel].mean()),
            cohort_rho_s_min=float(f["rho_s"][sel].min()),
            all_gap_max=float(gapmax),
        ))

    keys = list(rows[0].keys())
    with open(OUT / "cohort.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join("%.10g" % row[k] for k in keys) + "\n")
    print("\nwrote", OUT / "cohort.csv")

    print("\n%-6s %8s %8s %8s %8s %10s %10s"
          % ("T", "gap", "supp_rt", "nbr", "dmin/dx", "rho_c", "rho_s"))
    for row in rows:
        print("%-6.1f %8.2f %8.4f %8.2f %8.3f %10.6f %10.5f"
              % (row["T"], row["all_gap_max"], row["cohort_support_ratio_mean"],
                 row["cohort_nbr_mean"], row["cohort_dmin_mean"],
                 row["cohort_rho_c_mean"], row["cohort_rho_s_mean"]))

    np.savez_compressed(
        OUT / "cohort_particles.npz",
        **{"T%d" % int(T): dict(
            id=data[T]["id"], x=data[T]["x"], r=data[T]["r"],
            rho_c=data[T]["rho_c"], rho_s=data[T]["rho_s"],
            s_num=data[T]["s_num"], nbr=data[T]["nbr"], d_min=data[T]["d_min"],
            p=data[T]["p"], indicator=data[T]["indicator"],
            mass=data[T]["mass"]) for T in sorted(data)})


if __name__ == "__main__":
    main()
